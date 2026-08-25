#!/usr/bin/env python3
"""
Benchmark Ewald under pdb vs. `python -m pkdb`, one execution space at a time.

Neither debugger is interactive here. Each child is fed a breakpoint on the script's first line,
then `continue` and `quit` on stdin; pdb otherwise restarts the program when it finishes.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from helpers import (
    Runner,
    average,
    override_flag,
    parse_spaces,
    resolve_output_paths,
    system_and_suffix,
    write_checkpoint,
)

DEFAULT_SPACES = ["DebugOpenMP", "DebugCuda"]

KERNEL_KEYS = ["P2P", "P2G", "FFT", "CNV", "IFFT", "G2P"]


def parse_walltimes(stdout: str) -> dict[str, float | None] | None:
    """Last "Ewald Walltimes" section: TOTAL plus each KERNEL_KEYS component's "tot" value."""
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in stdout.splitlines()]

    start = None
    for i in range(len(lines) - 1, -1, -1):
        if "Ewald Walltimes" in lines[i]:
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if "Ewald Cost Model" in lines[i]:
            end = i
            break

    result: dict[str, float | None] = {
        "TOTAL": None,
        **{key: None for key in KERNEL_KEYS},
    }
    component = None
    for line in lines[start + 1 : end]:
        header = re.match(r"^(P2P|P2G|FFT|CNV|IFFT|G2P)\s*:", line)
        if header:
            component = header.group(1)
            continue

        m_total = re.match(r"^TOTAL:\s*([0-9eE+\-.]+)", line)
        if m_total:
            try:
                result["TOTAL"] = float(m_total.group(1))
            except ValueError:
                pass
            component = None
            continue

        if component is not None:
            m_tot = re.search(r"['\"]?tot['\"]?\s*:\s*([0-9eE+\-.]+)", line)
            if m_tot:
                try:
                    result[component] = float(m_tot.group(1))
                except ValueError:
                    pass
                component = None

    return result

# Atom counts passed to Ewald via --atoms. Every run here pays Python-level tracing on top of the
# kernels, so the wall clock is far higher than run_ewald.py's own timing.
ATOM_SIZES = [
    4_000,
    32_000,
    108_000,
    256_000,
    300_000,
    320_000,
    370_000,
    400_000,
    420_000,
    450_000,
    470_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    7_000_000,
    10_000_000,
]

N_RUNS = 3
TIMEOUT = 3600  # seconds per Ewald run

BENCHMARKS_DIR = Path(__file__).resolve().parent
PREFIX = "ewald_debuggers"

# One entry per JSON column; the plain runner is used only to precompile the pk_cpp bundles.
DEBUGGERS = [("pdb", ["-m", "pdb"]), ("pkdb", ["-m", "pkdb"])]
WARMUP_RUNNER = Runner(timeout=TIMEOUT, space_flag="--device")


def debugger_stdin(script: Path) -> str:
    """Break at the script's first line, run to the end, then leave the debugger."""
    code = compile(script.read_text(encoding="utf-8"), str(script), "exec")
    first = min(ln for _, _, ln in code.co_lines() if ln)
    return f"break {script}:{first}\ncontinue\nquit\n"


def run_n_times(
    runner: Runner,
    label: str,
    script: Path,
    space: str,
    script_args: list[str],
    log_path: Path,
    runs: int,
) -> tuple[float | None, float | None]:
    """Average (wall seconds, Ewald-reported TOTAL seconds) over `runs` runs; failed runs are skipped."""
    argv = [sys.executable, *runner.launcher, str(script)]
    if runner.space_flag:
        argv += [runner.space_flag, space]
    argv += script_args
    print(" ".join(argv))
    walls: list[float] = []
    kernels: list[float] = []
    for run in range(1, runs + 1):
        err = ""
        print(f"  {label} @ {space} run {run}/{runs} ...", end=" ", flush=True)
        started = time.perf_counter()
        try:
            code, out, err = runner.run_once(script, space, script_args)
            status = "OK" if code == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            code, out, status = -1, "TIMEOUT", "TIMEOUT"
        wall = time.perf_counter() - started
        walltimes = parse_walltimes(out) if code == 0 else None
        seconds = walltimes["TOTAL"] if walltimes is not None else None
        if code == 0:
            walls.append(wall)
            if seconds is not None:
                kernels.append(seconds)
            print(f"{status}  wall {wall:.2f}s" + (f"  ewald {seconds:.2f}s" if seconds is not None else ""))
        else:
            print(status)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"run {run} {label} {space}: wall {wall:.3f}s\n{out}\n")
            if err:
                f.write(f"stderr:\n{err}\n")
    return average(walls), average(kernels)


def warmup_spaces(spaces: list[str]) -> list[str]:
    """Every pk_cpp bundle the timed runs can reach: pkdb runs a plain <X> as Debug<X>."""
    reachable: list[str] = []
    for space in spaces:
        wanted = [space] if space.startswith("Debug") else [space, f"Debug{space}"]
        reachable += [s for s in wanted if s not in reachable]
    return reachable


def parse_atom_sizes(arg: str) -> list[int]:
    return [int(s.strip()) for s in arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "script",
        type=Path,
        nargs="?",
        default=BENCHMARKS_DIR / "ewald" / "stokes1p.py",
    )
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES), metavar="SPACES")
    parser.add_argument("--atom-sizes", default=",".join(str(n) for n in ATOM_SIZES), metavar="SIZES")
    parser.add_argument("--runs", type=int, default=N_RUNS, metavar="N")
    parser.add_argument("-o", "--output", "--output-file", type=Path, default=None, metavar="PATH")
    parser.add_argument("--log-file", type=Path, default=None, metavar="PATH")
    parser.add_argument("--script-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    script = args.script.resolve()
    if not script.exists():
        sys.exit(f"Error: script not found: {script}")
    spaces = parse_spaces(args.spaces)
    if not spaces:
        sys.exit("Error: --spaces is empty (no spaces to run).")
    atom_sizes = parse_atom_sizes(args.atom_sizes)
    if not atom_sizes:
        sys.exit("Error: --atom-sizes is empty (no sizes to run).")

    runs = args.runs
    stdin_text = debugger_stdin(script)
    debuggers = [
        (label, Runner(timeout=TIMEOUT, space_flag="--device", launcher=lc, stdin_text=stdin_text))
        for label, lc in DEBUGGERS
    ]

    system_info, suf = system_and_suffix()
    output, log_path = resolve_output_paths(args, PREFIX, suf)
    log_path.write_text("")  # fresh log each invocation

    # Compilation is debugger-independent, so precompile without one: pdb would pay the same
    # cost at Python-tracing speed.
    warmup_args = override_flag(args.script_args, "--atoms", str(min(atom_sizes)))
    WARMUP_RUNNER.warmup(script, warmup_args, warmup_spaces(spaces))

    print(f"\nEwald pdb vs. pkdb wall-clock timing, {runs} runs per configuration and per atom size:")
    for atoms in atom_sizes:
        print(f"\n=== atoms = {atoms} ===")
        size_args = override_flag(args.script_args, "--atoms", str(atoms))
        rows = []
        for space in spaces:
            avgs = {
                label: run_n_times(runner, label, script, space, size_args, log_path, runs)
                for label, runner in debuggers
            }
            row: dict = {"backend": space, "runs": runs}
            for label, (wall, kernel) in avgs.items():
                row[f"{label}_avg"] = round(wall, 4) if wall is not None else None
                row[f"{label}_kernel_avg"] = round(kernel, 4) if kernel is not None else None
            rows.append(row)

            wall_pdb, kern_pdb = avgs["pdb"]
            wall_pkdb, kern_pkdb = avgs["pkdb"]
            if wall_pdb and wall_pkdb:
                line = f"    -> {space}: wall pdb {wall_pdb:.2f}s  pkdb {wall_pkdb:.2f}s  {wall_pkdb / wall_pdb:.2f}x"
                if kern_pdb and kern_pkdb:
                    line += f"   (ewald {kern_pdb:.2f}s / {kern_pkdb:.2f}s  {kern_pkdb / kern_pdb:.2f}x)"
                print(line)
            else:
                for label, (wall, _) in avgs.items():
                    if wall is not None:
                        print(f"    -> {space}: {label} wall {wall:.2f}s")

        write_checkpoint(
            output,
            [{"atoms": atoms, "backends": rows}],
            x_key="atoms",
            test=PREFIX,
            system=system_info,
        )
        print(f"  checkpoint: wrote atoms={atoms} -> {output.name}")

    print(f"\nResults at {output}")
    print(f"Log at {log_path}")


if __name__ == "__main__":
    main()
