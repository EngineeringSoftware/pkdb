#!/usr/bin/env python3
"""
Benchmark ExaMiniMD across PyKokkos execution spaces (normal vs. pkdb debug).

For every atom count in ATOM_SIZES, ExaMiniMD is run N_RUNS times per selected
execution space; a run's time is the sum of the Total(s) column of the kernel
profile table that ExaMiniMD prints. A space and its Debug partner (e.g. Cuda
and DebugCuda) share one results row as normal_avg / pkdb_debug_avg.

Every run is a fresh subprocess: pykokkos caches compiled-module bindings per
process without keying them by execution space (mixing spaces in one process
breaks), and fresh processes keep timings independent. Timing results are
checkpointed to JSON after each atom size, folding into any existing file at
the same path.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from helpers import (
    Runner,
    average,
    override_flag,
    parse_spaces,
    resolve_output_paths,
    system_and_suffix,
    timing_plan,
    write_checkpoint,
)

# A space named Debug<X> is the pkdb debug variant of space <X>; when both are
# selected they form one results row (normal_avg vs. pkdb_debug_avg).
DEFAULT_SPACES = ["OpenMP", "DebugOpenMP", "Cuda", "DebugCuda"]

# Atom counts passed to ExaMiniMD via --atoms.
ATOM_SIZES = [
    1_000,
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
    # Sizes above 10M are omitted from the paper figures; append here to extend
]

N_RUNS = 10
TIMEOUT = 1800  # seconds per ExaMiniMD run

BENCHMARKS_DIR = Path(__file__).resolve().parent
PREFIX = "examinimd"


def parse_total_seconds(stdout: str) -> float | None:
    """Sum the Total(s) column of the kernel profile table ExaMiniMD prints."""
    total = 0.0
    found = False
    in_table = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if not in_table:
            in_table = stripped.startswith("Kernel") and "Total(s)" in stripped and "Avg(ms)" in stripped
            continue
        if not stripped:
            break
        parts = stripped.split()
        # Data rows look like: <name> <Count> <Total(s)> <Avg(ms)> <Percent%>
        if len(parts) >= 5 and parts[-1].endswith("%") and parts[-4].isdigit():
            try:
                total += float(parts[-3])
                found = True
            except ValueError:
                pass
    return total if found else None


RUNNER = Runner(timeout=TIMEOUT)


def run_n_times(script: Path, space: str, script_args: list[str], log_path: Path) -> float | None:
    print(f"PK_EXEC_SPACE=PK_EXA_SPACE={space}  {' '.join([sys.executable, str(script), *script_args])}")
    times: list[float] = []
    for run in range(1, N_RUNS + 1):
        show = run in (1, N_RUNS)
        if show:
            print(f"  {space} run {run}/{N_RUNS} ...", end=" ", flush=True)
        err = ""
        try:
            code, out, err = RUNNER.run_once(script, space, script_args)
            status = "OK" if code == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            code, out, status = -1, "TIMEOUT", "TIMEOUT"
        if show:
            print(status)
        if code == 0:
            seconds = parse_total_seconds(out)
            if seconds is not None:
                times.append(seconds)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"run {run} {space}:\n{out}\n")
            if err:
                f.write(f"stderr:\n{err}\n")
    return average(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "script",
        type=Path,
        nargs="?",
        default=BENCHMARKS_DIR / "ExaMiniMD" / "src" / "main.py",
    )
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES), metavar="SPACES")
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

    system_info, suf = system_and_suffix()
    output, log_path = resolve_output_paths(args, PREFIX, suf)
    log_path.write_text("")  # fresh log each invocation

    plan = timing_plan(spaces)
    smallest = min(ATOM_SIZES)
    # First run per space compiles the pk_cpp modules; smallest atom count keeps it quick.
    RUNNER.warmup(script, override_flag(args.script_args, "--atoms", str(smallest)), spaces)

    print(f"\nExaMiniMD timing, {N_RUNS} runs per configuration and per atom size:")
    for atoms in ATOM_SIZES:
        print(f"\n=== atoms = {atoms} ===")
        size_args = override_flag(args.script_args, "--atoms", str(atoms))
        rows = []
        for space, debug in plan:
            avg_normal = run_n_times(script, space, size_args, log_path)
            avg_debug = run_n_times(script, debug, size_args, log_path) if debug else None
            rows.append(
                {
                    "backend": space,
                    "runs": N_RUNS,
                    "normal_avg": round(avg_normal, 4) if avg_normal is not None else None,
                    "pkdb_debug_avg": round(avg_debug, 4) if avg_debug is not None else None,
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
            [{"atoms": atoms, "backends": rows}],
            x_key="atoms",
            test="examinimd",
            system=system_info,
        )
        print(f"  checkpoint: wrote atoms={atoms} -> {output.name}")

    print(f"\nResults at {output}")
    print(f"Log at {log_path}")


if __name__ == "__main__":
    main()
