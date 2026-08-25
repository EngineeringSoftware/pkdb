#!/usr/bin/env python3
"""
Benchmark Ewald across execution spaces (normal vs. pkdb debug).

Unlike examinimd/boltzmann, Ewald's main.py is not subprocessed: it is executed
fresh in-process as __main__ for every run (see run_once). For
every atom count in ATOM_SIZES, it is run N_RUNS times per selected execution
space via --device; a run's time is TOTAL from the last "Ewald Walltimes"
breakdown it prints. A space and its Debug partner (e.g. Cuda and DebugCuda)
share one results row as normal_avg / pkdb_debug_avg.

Per-kernel (P2P/P2G/FFT/CNV/IFFT/G2P) times are only sampled at the smallest
atom count (cheapest to profile) and averaged across those runs. Timing and
kernel-profile JSON are checkpointed as soon as they're available, folding
into any existing file at the same path.
"""

import argparse
import io
import json
import os
import re
import runpy
import sys
from contextlib import redirect_stdout, redirect_stderr
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

# Atom counts passed to Ewald via --atoms.
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

N_RUNS = 1
KERNEL_KEYS = ["P2P", "P2G", "FFT", "CNV", "IFFT", "G2P"]

BENCHMARKS_DIR = Path(__file__).resolve().parent
PREFIX = "ewald"


def parse_walltimes(stdout: str) -> dict[str, float | None] | None:
    """Last "Ewald Walltimes" section: TOTAL plus each KERNEL_KEYS component's "tot" value."""
    lines = [
        re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in stdout.splitlines()
    ]

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


def aggregate_kernels(walltimes: list[dict]) -> dict:
    """Average each KERNEL_KEYS component across per-run parse_walltimes() dicts."""
    by_kernel: dict[str, list[float]] = {}
    for wt in walltimes:
        for key in KERNEL_KEYS:
            val = wt.get(key)
            if val is not None:
                by_kernel.setdefault(key, []).append(val)
    kernels = [
        {"name": key, "total_time": round(average(vals), 6)}
        for key, vals in by_kernel.items()
    ]
    return {"kernels": kernels, "profile_runs": len(walltimes)}


class EwaldRunner(Runner):
    """
    Ewald's script is executed in-process as __main__ (via runpy with a patched
    sys.argv) instead of subprocessed (unlike boltzmann/examinimd), so only
    run_once needs overriding here; warmup() is inherited as-is.
    """

    def run_once(
        self, main_py: Path, space: str, script_args: list[str]
    ) -> tuple[int, str, str]:
        os.environ["PK_EXEC_SPACE"] = os.environ["PK_EXA_SPACE"] = space

        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        returncode = 0
        saved_argv = sys.argv
        sys.argv = [str(main_py), *script_args, "--device", space]
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                runpy.run_path(str(main_py), run_name="__main__")
        except SystemExit as e:
            returncode = e.code if isinstance(e.code, int) else 0
        except Exception as e:
            returncode = 1
            stderr_buf.write(str(e))
        finally:
            sys.argv = saved_argv

        return returncode, stdout_buf.getvalue(), stderr_buf.getvalue()


RUNNER = EwaldRunner()


def run_n_times(
    main_py: Path, space: str, script_args: list[str], log_path: Path
) -> tuple[float | None, list[dict]]:
    """Run N_RUNS times for one space; return (avg TOTAL wall time, per-run walltime dicts)."""
    print(f"PK_EXEC_SPACE=PK_EXA_SPACE={space}  {main_py.name} {' '.join(script_args)}")
    times: list[float] = []
    walltimes: list[dict] = []
    for run in range(1, N_RUNS + 1):
        show = run in (1, N_RUNS)
        if show:
            print(f"  {space} run {run}/{N_RUNS} ...", end=" ", flush=True)
        code, out, _ = RUNNER.run_once(main_py, space, script_args)
        if show:
            print("OK" if code == 0 else "FAIL")
        if code == 0:
            wt = parse_walltimes(out)
            if wt is not None and wt["TOTAL"] is not None:
                times.append(wt["TOTAL"])
                walltimes.append(wt)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"run {run} {space}:\n{out}\n")
    return average(times), walltimes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES), metavar="SPACES")
    parser.add_argument(
        "-o", "--output", "--output-file", type=Path, default=None, metavar="PATH"
    )
    parser.add_argument("--kernels-output", type=Path, default=None, metavar="PATH")
    parser.add_argument("--log-file", type=Path, default=None, metavar="PATH")
    parser.add_argument("--script-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    main_py = args.script.resolve()
    if not main_py.exists():
        sys.exit(f"Error: script not found: {main_py}")
    spaces = parse_spaces(args.spaces)
    if not spaces:
        sys.exit("Error: --spaces is empty.")

    system_info, suf = system_and_suffix()
    output = results_path(args.output or Path(suf), PREFIX, ".json")
    kernels_output = results_path(
        args.kernels_output or Path(f"kernel_profile_{suf}"), PREFIX, ".json"
    )
    log_path = results_path(args.log_file or Path("out"), PREFIX, ".log")
    for path in (output, kernels_output, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")  # fresh log each invocation

    plan = timing_plan(spaces)
    smallest = min(ATOM_SIZES)
    # First run per space compiles the pk_cpp modules; smallest atom count keeps it quick.
    RUNNER.warmup(
        main_py, override_flag(args.script_args, "--atoms", str(smallest)), spaces
    )

    print(f"\nEwald timing, {N_RUNS} run(s) per configuration and per atom size:")
    for atoms in ATOM_SIZES:
        print(f"\n=== atoms = {atoms} ===")
        size_args = override_flag(args.script_args, "--atoms", str(atoms))
        timing_rows = []
        kernel_walltimes: dict[str, list[dict]] = {}

        for space, debug in plan:
            avg_normal, wt_normal = run_n_times(main_py, space, size_args, log_path)
            avg_debug, wt_debug = (
                run_n_times(main_py, debug, size_args, log_path)
                if debug
                else (None, [])
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
            if atoms == smallest:
                kernel_walltimes[space] = wt_normal
                if debug:
                    kernel_walltimes[debug] = wt_debug

            if avg_normal and avg_debug:
                print(
                    f"    -> {space}: {avg_normal:.3f}s  {debug}: {avg_debug:.3f}s  "
                    f"slowdown: {avg_debug / avg_normal:.2f}x"
                )
            elif avg_normal is not None:
                print(f"    -> {space}: {avg_normal:.3f}s")

        write_checkpoint(
            output,
            [{"atoms": atoms, "backends": timing_rows}],
            x_key="atoms",
            test="ewald",
            system=system_info,
        )
        print(f"  checkpoint: wrote atoms={atoms} -> {output.name}")

        if atoms == smallest:
            aggregated = {
                space: aggregate_kernels(wt) for space, wt in kernel_walltimes.items()
            }
            kernels_output.write_text(
                json.dumps(aggregated, indent=2), encoding="utf-8"
            )
            print(f"  kernel profiles written to {kernels_output.name}")

    print(f"\nResults at {output}")
    print(f"Kernel profiles at {kernels_output}")
    print(f"Log at {log_path}")


if __name__ == "__main__":
    main()
