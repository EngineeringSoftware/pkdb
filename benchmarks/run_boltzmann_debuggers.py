#!/usr/bin/env python3
"""
Benchmark Boltzmann under pdb vs. `python -m pkdb`, one execution space at a time.

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
from run_boltzmann import parse_total_time_s

DEFAULT_SPACES = ["DebugOpenMP", "DebugCuda"]

# Step counts passed to boltzmann/main.py via -s. Every run here pays Python-level tracing on top
# of the kernels, so the wall clock is far higher than run_boltzmann.py's own timing.
STEP_COUNTS = [
    # 10,
    # 100,
    # 400,
    # 1_000,
    # 5_000,
    10_000,
]

N_RUNS = 3
TIMEOUT = 3600  # seconds per boltzmann run

BENCHMARKS_DIR = Path(__file__).resolve().parent
# Renamed with the pdb baseline: write_checkpoint folds into any file it finds.
PREFIX = "boltzmann_debuggers"

# One entry per JSON column; the plain runner is used only to precompile the pk_cpp bundles.
DEBUGGERS = [("pdb", ["-m", "pdb"]), ("pkdb", ["-m", "pkdb"])]
WARMUP_RUNNER = Runner(timeout=TIMEOUT, space_flag="-space")


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
    """Average (wall seconds, boltzmann-reported seconds) over `runs` runs; failed runs are skipped."""
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
        seconds = parse_total_time_s(out) if code == 0 else None
        if code == 0:
            walls.append(wall)
            if seconds is not None:
                kernels.append(seconds)
            print(f"{status}  wall {wall:.2f}s" + (f"  boltzmann {seconds:.2f}s" if seconds is not None else ""))
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


def parse_step_counts(arg: str) -> list[int]:
    return [int(s.strip()) for s in arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "script",
        type=Path,
        nargs="?",
        default=BENCHMARKS_DIR / "boltzmann" / "main.py",
    )
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES), metavar="SPACES")
    parser.add_argument("--step-counts", default=",".join(str(n) for n in STEP_COUNTS), metavar="COUNTS")
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
    step_counts = parse_step_counts(args.step_counts)
    if not step_counts:
        sys.exit("Error: --step-counts is empty (no sizes to run).")

    runs = args.runs
    stdin_text = debugger_stdin(script)
    debuggers = [
        (label, Runner(timeout=TIMEOUT, space_flag="-space", launcher=lc, stdin_text=stdin_text))
        for label, lc in DEBUGGERS
    ]

    system_info, suf = system_and_suffix()
    output, log_path = resolve_output_paths(args, PREFIX, suf)
    log_path.write_text("")  # fresh log each invocation

    # Compilation is debugger-independent, so precompile without one: pdb would pay the same
    # cost at Python-tracing speed.
    warmup_args = override_flag(args.script_args, "-s", str(min(step_counts)))
    WARMUP_RUNNER.warmup(script, warmup_args, warmup_spaces(spaces))

    print(f"\nBoltzmann pdb vs. pkdb wall-clock timing, {runs} runs per configuration and per step count:")
    for steps in step_counts:
        print(f"\n=== steps = {steps} ===")
        size_args = override_flag(args.script_args, "-s", str(steps))
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
                    line += f"   (boltzmann {kern_pdb:.2f}s / {kern_pkdb:.2f}s  {kern_pkdb / kern_pdb:.2f}x)"
                print(line)
            else:
                for label, (wall, _) in avgs.items():
                    if wall is not None:
                        print(f"    -> {space}: {label} wall {wall:.2f}s")

        write_checkpoint(
            output,
            [{"steps": steps, "backends": rows}],
            x_key="steps",
            test=PREFIX,
            system=system_info,
        )
        print(f"  checkpoint: wrote steps={steps} -> {output.name}")

    print(f"\nResults at {output}")
    print(f"Log at {log_path}")


if __name__ == "__main__":
    main()
