#!/usr/bin/env python3
"""
Benchmark ExaMiniMD under pdb vs. `python -m pkdb`, one execution space at a time.

Neither debugger is interactive here. Each child is fed a breakpoint on the script's first line,
then `continue` and `quit` on stdin; pdb otherwise restarts the program when it finishes.
"""

import argparse
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
from run_examinimd import parse_total_seconds

DEFAULT_SPACES = ["DebugOpenMP", "DebugCuda"]

# Atom counts passed to ExaMiniMD via --atoms. Small next to run_examinimd.py's list: every
# run here pays Python-level tracing on top of the kernels, so the wall clock is far higher.
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
TIMEOUT = 3600  # seconds per ExaMiniMD run

BENCHMARKS_DIR = Path(__file__).resolve().parent
# Renamed with the pdb baseline: write_checkpoint folds into any file it finds.
PREFIX = "examinimd_debuggers"

# One entry per JSON column; the plain runner is used only to precompile the pk_cpp bundles.
DEBUGGERS = [("pdb", ["-m", "pdb"]), ("pkdb", ["-m", "pkdb"])]
WARMUP_RUNNER = Runner(timeout=TIMEOUT)


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
    """Average (wall seconds, kernel-profile seconds) over `runs` runs; failed runs are skipped."""
    argv = [sys.executable, *runner.launcher, str(script), *script_args]
    print(f"PK_EXEC_SPACE=PK_EXA_SPACE={space}  {' '.join(argv)}")
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
        seconds = parse_total_seconds(out) if code == 0 else None
        if code == 0:
            walls.append(wall)
            if seconds is not None:
                kernels.append(seconds)
            print(f"{status}  wall {wall:.2f}s" + (f"  kernels {seconds:.2f}s" if seconds is not None else ""))
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
        default=BENCHMARKS_DIR / "ExaMiniMD" / "src" / "main.py",
    )
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES), metavar="SPACES")
    parser.add_argument("--atom-sizes", default=",".join(str(n) for n in ATOM_SIZES), metavar="COUNTS")
    parser.add_argument("--runs", type=int, default=N_RUNS, metavar="N")
    parser.add_argument("-o", "--output", "--output-file", type=Path, default=None, metavar="PATH")
    parser.add_argument("--log-file", type=Path, default=None, metavar="PATH")
    parser.add_argument("--script-args", nargs=argparse.REMAINDER, default=["-il", "../input/in.lj", "--fill"])
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
    debuggers = [(label, Runner(timeout=TIMEOUT, launcher=lc, stdin_text=stdin_text)) for label, lc in DEBUGGERS]

    system_info, suf = system_and_suffix()
    output, log_path = resolve_output_paths(args, PREFIX, suf)
    log_path.write_text("")  # fresh log each invocation

    # Compilation is debugger-independent, so precompile without one: pdb would pay the same
    # cost at Python-tracing speed.
    warmup_args = override_flag(args.script_args, "--atoms", str(min(atom_sizes)))
    WARMUP_RUNNER.warmup(script, warmup_args, warmup_spaces(spaces))

    print(f"\nExaMiniMD pdb vs. pkdb wall-clock timing, {runs} runs per configuration and per atom size:")
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
                    line += f"   (kernels {kern_pdb:.2f}s / {kern_pkdb:.2f}s  {kern_pkdb / kern_pdb:.2f}x)"
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
