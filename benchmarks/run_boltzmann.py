#!/usr/bin/env python3
"""
Benchmark Boltzmann across execution spaces (normal vs. pkdb debug).

For every step count in STEP_COUNTS, boltzmann/main.py is run N_RUNS times per
selected execution space via -space; a run's time is the last "total time =
<float>" line it prints (main.py runs an internal warmup + "RESET" first, so
the last such line is the timed run). Kernel timings come from the pprint of
stack_profiler's profile dict that follows that line (ast.literal_eval; no
modifications inside boltzmann). A space and its Debug partner (e.g. Cuda and
DebugCuda) share one results row as normal_avg / pkdb_debug_avg.

Every run is a fresh subprocess, matching run_examinimd.py. Timing and
kernel-profile JSON are checkpointed after each step count, folding into any
existing file at the same path.
"""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

from helpers import (
    Runner,
    average,
    override_flag,
    parse_spaces,
    results_path,
    system_and_suffix,
    timing_plan,
    write_checkpoint,
)

# A space named Debug<X> is the pkdb debug variant of space <X>; when both are
# selected they form one results row (normal_avg vs. pkdb_debug_avg).
DEFAULT_SPACES = ["OpenMP", "DebugOpenMP", "Cuda", "DebugCuda"]

# Timestep counts passed to boltzmann/main.py via -s/--steps.
STEP_COUNTS = [10, 100, 400, 1_000, 5_000, 10_000]

N_RUNS = 10
TIMEOUT = 3600  # seconds per boltzmann run

BENCHMARKS_DIR = Path(__file__).resolve().parent
PREFIX = "boltzmann"


def parse_total_time_s(stdout: str) -> float | None:
    """Last "total time = <float>" line (warmup + RESET produce an earlier one)."""
    matches = list(
        re.finditer(
            r"^total time = ([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*$",
            stdout,
            re.MULTILINE,
        )
    )
    return float(matches[-1].group(1)) if matches else None


def _balanced_brace_slice(s: str, start: int) -> str | None:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def parse_profile_dict(stdout: str) -> dict | None:
    """dict literal following the last "total time =" line (pprint of stack_profiler.calls)."""
    matches = list(re.finditer(r"^total time = ", stdout, re.MULTILINE))
    if not matches:
        return None
    tail = stdout[matches[-1].end() :]
    start = tail.find("{")
    if start == -1:
        return None
    literal = _balanced_brace_slice(tail, start)
    if literal is None:
        return None
    try:
        d = ast.literal_eval(literal)
    except (ValueError, SyntaxError):
        return None
    return d if isinstance(d, dict) else None


def profile_to_kernel_rows(profile: dict) -> list[dict]:
    """stack_profiler {(parent, func, lineno) | (func, lineno) | key: (time_s, calls)} -> sorted rows."""
    rows = []
    for key, val in profile.items():
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            continue
        time_s, calls = float(val[0]), int(val[1])
        if isinstance(key, tuple) and len(key) == 3:
            parent, function, lineno = key
        elif isinstance(key, tuple) and len(key) == 2:
            parent, function, lineno = None, key[0], key[1]
        else:
            parent, function, lineno = None, str(key), None
        rows.append(
            {
                "parent": parent,
                "function": function,
                "lineno": lineno,
                "time_s": round(time_s, 6),
                "calls": calls,
            }
        )
    rows.sort(key=lambda r: -r["time_s"])
    return rows


RUNNER = Runner(timeout=TIMEOUT, space_flag="-space")


def run_n_times(
    main_py: Path, space: str, script_args: list[str], log_path: Path
) -> tuple[float | None, list[dict] | None]:
    """Run N_RUNS times for one space; return (avg wall time, kernel rows from the last successful parse)."""
    print(f"{space}: {sys.executable} {main_py.name} -space {space} {' '.join(script_args)}")
    times: list[float] = []
    last_rows: list[dict] | None = None
    for run in range(1, N_RUNS + 1):
        show = run in (1, N_RUNS)
        if show:
            print(f"  {space} run {run}/{N_RUNS} ...", end=" ", flush=True)
        err = ""
        try:
            code, out, err = RUNNER.run_once(main_py, space, script_args)
            status = "OK" if code == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            code, out, status = -1, "TIMEOUT", "TIMEOUT"
        if show:
            print(status)
        if code == 0:
            t = parse_total_time_s(out)
            if t is not None:
                times.append(t)
            profile = parse_profile_dict(out)
            if profile is not None:
                last_rows = profile_to_kernel_rows(profile)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"run {run} {space}:\n{out}\n")
            if err:
                f.write(f"stderr:\n{err}\n")
    return average(times), last_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "main_py",
        type=Path,
        nargs="?",
        default=BENCHMARKS_DIR / "boltzmann" / "main.py",
    )
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES), metavar="SPACES")
    parser.add_argument(
        "-o", "--output", "--output-file", type=Path, default=None, metavar="PATH"
    )
    parser.add_argument("--kernels-output", type=Path, default=None, metavar="PATH")
    parser.add_argument("--log-file", type=Path, default=None, metavar="PATH")
    parser.add_argument("--script-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    main_py = args.main_py.resolve()
    if not main_py.exists():
        sys.exit(f"Error: main.py not found: {main_py}")
    spaces = parse_spaces(args.spaces)
    if not spaces:
        sys.exit("Error: --spaces is empty.")

    system_info, suf = system_and_suffix()
    output = results_path(args.output or Path(f"results_{suf}"), PREFIX, ".json")
    kernels_output = results_path(
        args.kernels_output or Path(f"kernels_{suf}"), PREFIX, ".json"
    )
    log_path = results_path(args.log_file or Path("out"), PREFIX, ".log")
    for path in (output, kernels_output, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # fresh log each invocation

    plan = timing_plan(spaces)
    smallest = min(STEP_COUNTS)
    # First run per space compiles the pk_cpp modules; smallest step count keeps it quick.
    RUNNER.warmup(main_py, override_flag(args.script_args, "-s", str(smallest)), spaces)

    config = {
        "main_py": str(main_py),
        "script_args": args.script_args,
        "step_counts": list(STEP_COUNTS),
        "n_runs": N_RUNS,
        "timeout_s": TIMEOUT,
    }

    print(
        f"\nBoltzmann timing, {N_RUNS} runs per configuration and per step count; STEP_COUNTS={STEP_COUNTS}"
    )
    for steps in STEP_COUNTS:
        print(f"\n=== steps = {steps} ===")
        size_args = override_flag(args.script_args, "-s", str(steps))
        timing_rows, kernel_rows = [], []

        for space, debug in plan:
            avg_normal, rows_normal = run_n_times(main_py, space, size_args, log_path)
            avg_debug, rows_debug = (
                run_n_times(main_py, debug, size_args, log_path)
                if debug
                else (None, None)
            )
            timing_rows.append(
                {
                    "backend": space,
                    "runs": N_RUNS,
                    "normal_avg": (
                        round(avg_normal, 4) if avg_normal is not None else None
                    ),
                    "pkdb_debug_avg": (
                        round(avg_debug, 4) if avg_debug is not None else None
                    ),
                }
            )
            kernel_rows.append(
                {
                    "backend": space,
                    "normal_space": space,
                    "debug_space": debug,
                    "kernels_normal": rows_normal,
                    "kernels_debug": rows_debug,
                }
            )
            if avg_normal and avg_debug:
                print(
                    f"    -> {space}: {avg_normal:.3f}s  {debug}: {avg_debug:.3f}s  "
                    f"slowdown: {avg_debug / avg_normal:.2f}x"
                )
            elif avg_normal is not None:
                print(f"    -> {space}: {avg_normal:.3f}s")

        write_checkpoint(
            output,
            [{"steps": steps, "backends": timing_rows}],
            x_key="steps",
            test="boltzmann",
            system=system_info,
            extra_fields={
                "kernels_output_file": str(kernels_output),
                "log_file": str(log_path),
                "config": config,
            },
        )
        write_checkpoint(
            kernels_output,
            [{"steps": steps, "backends": kernel_rows}],
            x_key="steps",
            test="boltzmann_kernels",
            system=system_info,
            extra_fields={"config": config},
        )
        print(
            f"  checkpoint: wrote steps={steps} -> {output.name}, {kernels_output.name}"
        )

    print(f"\nResults at {output}")
    print(f"Kernel timings at {kernels_output}")
    print(f"Log at {log_path}")


if __name__ == "__main__":
    main()
