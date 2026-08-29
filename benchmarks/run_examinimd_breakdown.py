#!/usr/bin/env python3
"""
Collect `python -m pkdb --breakdown` phase timings for ExaMiniMD across atom sizes.

Unlike run_examinimd_debuggers.py (which times pdb vs. pkdb as opaque wall clock), this
parses pkdb's own `[pkdb] Timing breakdown:` report out of each run's stdout, so the JSON
records where the wall clock actually went.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from helpers import Runner, override_flag, parse_spaces, resolve_output_paths, system_and_suffix, write_checkpoint

# Import the label text straight from phase_timing.py rather than hand-copying it, so this
# parser can't silently drift out of sync with what `--breakdown` actually prints again (it
# already had once: it was written against an older phase_timing.py whose top-level labels
# were "Metadata-hook overhead" / "Execute-anchor overhead" / "App execution", and silently
# dropped every field whose label had since changed, e.g. the top-level "bdb_line_tracing"
# phase and the "detail" for "metadata_hook"/"execute_hook" - see benchmarks/results/
# examinimd_breakdown_out.log for output captured against that older version).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pkdb.core import phase_timing as pt  # noqa: E402

DEFAULT_SPACES = ["DebugCuda"]
DEFAULT_ATOM_SIZES = [
    4_000,
    5_000,
    20_000,
    10_000,
    32_000,
    108_000,
    256_000,
    300_000,
    320_000,
    370_000,
]
TIMEOUT = 600  # seconds per run

BENCHMARKS_DIR = Path(__file__).resolve().parent
PREFIX = "examinimd_breakdown"

# "  <label>  <seconds>s  (n=<count>  <pct>%)" - the App execution line omits "n=", and
# Total wall clock omits the trailing "(...)" entirely.
_PHASE_RE = re.compile(
    r"^\s*(?P<label>.+?)\s+(?P<seconds>\d+\.\d+)s(?:\s+\(\s*(?:n=(?P<count>\d+)\s+)?(?P<pct>[\d.]+)%\))?\s*$"
)


def _bare_label(label: str) -> str:
    """Strip a trailing parenthetical, e.g. "Startup (process launch -> ...)" -> "Startup"."""
    return re.sub(r"\s*\(.*\)\s*$", "", label.strip()).strip()


# Label -> key, one dict per nesting level in print_report()'s output (top-level phases;
# "detail" nested under "bdb_line_tracing"; "detail" nested under "execute_hook" in turn).
# Built from phase_timing.py's own label dicts rather than copied by hand.
_TOP_LEVEL_KEYS = {_bare_label(v): k for k, v in pt.PHASE_LABELS.items()}
_TOP_LEVEL_KEYS[_bare_label(pt.UNATTRIBUTED_LABEL)] = pt.UNATTRIBUTED_KEY
_TOP_LEVEL_KEYS[_bare_label(pt.TOTAL_WALL_LABEL)] = pt.TOTAL_WALL_KEY
_DETAIL_KEYS = {_bare_label(v): k for k, v in pt.DETAIL_LABELS.items()}
_SUBDETAIL_KEYS = {_bare_label(v): k for k, v in pt.SUBDETAIL_LABELS.items()}


def parse_breakdown(stdout: str) -> dict | None:
    """Parse pkdb's `[pkdb] Timing breakdown:` report."""
    if "[pkdb] Timing breakdown:" not in stdout:
        return None

    phases: dict = {}
    current_top: dict | None = None
    current_detail: dict | None = None
    for line in stdout.splitlines():
        m = _PHASE_RE.match(line)
        if not m:
            continue
        bare_label = _bare_label(m.group("label"))
        seconds = float(m.group("seconds"))
        count = int(m.group("count")) if m.group("count") else None
        pct = float(m.group("pct")) if m.group("pct") else None
        entry: dict = {"seconds": seconds}
        if count is not None:
            entry["count"] = count
        if pct is not None:
            entry["pct"] = pct

        if bare_label in _TOP_LEVEL_KEYS:
            phases[_TOP_LEVEL_KEYS[bare_label]] = entry
            current_top = entry
            current_detail = None
        elif bare_label in _DETAIL_KEYS and current_top is not None:
            current_top.setdefault("detail", {})[_DETAIL_KEYS[bare_label]] = entry
            current_detail = entry
        elif bare_label in _SUBDETAIL_KEYS and current_detail is not None:
            current_detail.setdefault("detail", {})[_SUBDETAIL_KEYS[bare_label]] = entry

    if not phases:
        return None

    return {"phases": phases}


def run_one(runner: Runner, script: Path, space: str, script_args: list[str], log_path: Path, atoms: int) -> dict:
    print(f"  atoms={atoms} @ {space} ...", end=" ", flush=True)
    try:
        code, out, err = runner.run_once(script, space, script_args)
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"atoms={atoms} {space}: TIMEOUT\n")
        return {"backend": space, "error": "timeout"}

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"atoms={atoms} {space}: exit={code}\n{out}\n")
        if err:
            f.write(f"stderr:\n{err}\n")

    if code != 0:
        print(f"FAIL (exit {code})")
        return {"backend": space, "error": f"exit_{code}"}

    parsed = parse_breakdown(out)
    if parsed is None:
        print("FAIL (no breakdown found in output)")
        return {"backend": space, "error": "no_breakdown"}

    total = parsed["phases"].get("total_wall_clock", {}).get("seconds")
    print(f"OK  wall {total:.2f}s" if total is not None else "OK")
    row = {"backend": space, "runs": 1}
    row.update(parsed)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "script",
        type=Path,
        nargs="?",
        default=BENCHMARKS_DIR / "ExaMiniMD" / "src" / "main.py",
    )
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES), metavar="SPACES")
    parser.add_argument("--atom-sizes", default=",".join(str(n) for n in DEFAULT_ATOM_SIZES), metavar="COUNTS")
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
    atom_sizes = [int(s.strip()) for s in args.atom_sizes.split(",") if s.strip()]
    if not atom_sizes:
        sys.exit("Error: --atom-sizes is empty (no sizes to run).")

    stdin_text = "continue\nquit\n"
    runner = Runner(timeout=TIMEOUT, launcher=["-m", "pkdb", "--breakdown"], stdin_text=stdin_text)

    system_info, suf = system_and_suffix()
    output, log_path = resolve_output_paths(args, PREFIX, suf)
    log_path.write_text("")  # fresh log each invocation

    print(f"\nExaMiniMD pkdb --breakdown phase timings, atom sizes {atom_sizes}, spaces {spaces}:")
    for atoms in atom_sizes:
        print(f"\n=== atoms = {atoms} ===")
        size_args = override_flag(args.script_args, "--atoms", str(atoms))
        rows = [run_one(runner, script, space, size_args, log_path, atoms) for space in spaces]

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
