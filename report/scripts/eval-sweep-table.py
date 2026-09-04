#!/usr/bin/env python3
"""
Debug wall-time sweep-range table + macros, derived entirely from input JSON paths.

Reads the `*_debuggers_*.json` shape: one entry per backend per size, holding
`pdb_avg` (plain pdb session) and `pkdb_avg` (pkdb session). The older
`normal_avg`/`debug_avg` shape is still accepted.

Usage:
  # Explicit files:
  python3 eval-sweep-table.py path/to/a.json path/to/b.json

  # Or scan a directory for every matching *.json (default: the repo's
  # benchmarks/results):
  python3 eval-sweep-table.py
  python3 eval-sweep-table.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from report_body import ensure_inputs

FNAME_RE = re.compile(
    r"^(?P<bench>[A-Za-z][A-Za-z0-9]*)_(?:results_)?(?P<platform>.+)$"
)

CPU_ARCH_PREFIXES = ("x86_64_", "x86-64-", "arm64_", "aarch64_")
MARKER_PREFIXES = ("debuggers_", "old_vs_pkdb_", "debug_comparison_")
SKIP_SUBSTRINGS = ("kernel_profile", "hotswap", "old_vs_pkdb", "steps_result")

# Known paper macros for benchmark display names (paper/sc26/macros.tex);
# anything else falls back to a plain capitalized string.
BENCH_DISPLAY = {
    "examinimd": r"\Exa",
    "boltzmann": r"\Boltz",
    "ewald": r"\Ewald",
    "parki": r"\Parki",
}
BENCH_ORDER = ("examinimd", "boltzmann", "ewald", "parki")

BACKEND_DISPLAY = {"openmp": r"\Openmp", "cuda": r"\Cuda", "hip": r"\Hip"}
BACKEND_ORDER = ("openmp", "cuda", "hip")


def _fmt_num(y: float) -> str:
    s = f"{y:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _def_macro(lines: List[str], key: str, value: str) -> None:
    lines.append(f"\\DefMacro{{{key}}}{{{value}}}")


def _slugify(stem: str) -> str:
    s = stem.lower()
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-") or "unknown"


VENDOR_TOKENS = {"nvidia", "amd", "intel"}


def _prettify_token(tok: str) -> str:
    if not tok:
        return tok
    if any(c.isdigit() for c in tok):
        return tok.upper()
    return tok.capitalize()


def _strip_marker(stem: str) -> str:
    low = stem.lower()
    for m in MARKER_PREFIXES:
        if low.startswith(m):
            return stem[len(m) :]
    return stem


def backend_display(backend: str) -> str:
    return BACKEND_DISPLAY.get(backend, backend.capitalize())


def platform_label(platform_stem: str) -> str:
    """Human-readable 'Platform' column text: GPU name only, e.g. 'x86_64_amd_8x(mi300x)' -> '8XMI300X'."""
    s = _strip_marker(platform_stem)
    low = s.lower()
    for pre in CPU_ARCH_PREFIXES:
        if low.startswith(pre):
            s = s[len(pre) :]
            break
    s = s.replace("(", "").replace(")", "")
    tokens = [t for t in s.split("_") if t]
    tokens = [t for t in tokens if t.lower() not in VENDOR_TOKENS] or tokens
    return " ".join(_prettify_token(t) for t in tokens) or platform_stem


def parse_filename(path: Path) -> Optional[Tuple[str, str]]:
    m = FNAME_RE.match(path.stem)
    if not m:
        return None
    return m.group("bench").lower(), _strip_marker(m.group("platform"))


@dataclass
class Platform:
    path: Path
    benchmark: str
    platform_stem: str
    platform_slug: str
    label: str
    x_key: str
    x_values: List[int]
    series: Dict[str, Tuple[List[float], List[float]]]  # backend -> (pdb, pkdb)
    gpu_model: str = "unknown"


def _pair(entry: dict) -> Tuple[Optional[float], Optional[float]]:
    """(pdb, pkdb) times for one backend entry; older JSONs use normal_avg/debug_avg."""
    if "pdb_avg" in entry:
        return entry.get("pdb_avg"), entry.get("pkdb_avg")
    debug = entry.get("debug_avg")
    if debug is None:
        debug = entry.get("pkdb_debug_avg")
    return entry.get("normal_avg"), debug


def _backend_key(name: str) -> str:
    """DebugCuda -> cuda; a debug run always uses the Debug* space."""
    if name.lower().startswith("debug") and len(name) > len("Debug"):
        name = name[len("Debug") :]
    return name.lower()


def _x_key_of(sizes: List[dict]) -> Optional[str]:
    if "atoms" in sizes[0]:
        return "atoms"
    if "steps" in sizes[0]:
        return "steps"
    return None


def _ordered_backends(series: Dict[str, object]) -> List[str]:
    def key(b: str) -> Tuple[int, str]:
        return (BACKEND_ORDER.index(b) if b in BACKEND_ORDER else len(BACKEND_ORDER), b)

    return sorted(series, key=key)


def _load_series(
    data: dict,
) -> Optional[Tuple[str, List[int], Dict[str, Tuple[List[float], List[float]]]]]:
    """Read sizes[] into per-backend (pdb, pkdb) series, stopping at the first gap."""
    sizes = data.get("sizes") or []
    if not sizes:
        return None
    x_key = _x_key_of(sizes)
    if x_key is None:
        return None

    xs: List[int] = []
    series: Dict[str, Tuple[List[float], List[float]]] = {}
    for entry in sizes:
        if x_key not in entry:
            break
        row: Dict[str, Tuple[float, float]] = {}
        for backend_entry in entry.get("backends", []):
            pdb_t, pkdb_t = _pair(backend_entry)
            if pdb_t is None or pkdb_t is None:
                row = {}
                break
            row[_backend_key(backend_entry.get("backend", ""))] = (
                float(pdb_t),
                float(pkdb_t),
            )
        if not row or (series and set(row) != set(series)):
            break
        x_val = int(entry[x_key])
        if x_val in xs:
            # A re-run can leave the same size in the JSON twice.
            continue
        xs.append(x_val)
        for backend, (pdb_t, pkdb_t) in row.items():
            pdb_ys, pkdb_ys = series.setdefault(backend, ([], []))
            pdb_ys.append(pdb_t)
            pkdb_ys.append(pkdb_t)

    if not xs:
        return None
    return x_key, xs, series


def load_platform(path: Path) -> Optional[Platform]:
    parsed = parse_filename(path)
    if parsed is None:
        print(
            f"Skipping {path}: filename doesn't match <benchmark>[_kind]_<platform>.json",
            file=sys.stderr,
        )
        return None
    bench, platform_stem = parsed

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Skipping {path}: {e}", file=sys.stderr)
        return None

    result = _load_series(data)
    if result is None:
        print(f"Skipping {path}: no paired pdb/pkdb sweep found.", file=sys.stderr)
        return None

    x_key, xs, series = result
    gpus = (data.get("system") or {}).get("gpus") or []
    gpu_model = str(gpus[0]["model"]) if gpus else "unknown"

    return Platform(
        path=path,
        benchmark=bench,
        platform_stem=platform_stem,
        platform_slug=_slugify(platform_stem),
        label=platform_label(platform_stem),
        x_key=x_key,
        x_values=xs,
        series=series,
        gpu_model=gpu_model,
    )


def discover_paths(explicit: Sequence[Path], results_dir: Path) -> List[Path]:
    if explicit:
        return list(explicit)
    if not results_dir.is_dir():
        print(f"Warning: results dir not found: {results_dir}", file=sys.stderr)
        return []
    return [
        p
        for p in sorted(results_dir.glob("*.json"))
        if not any(s in p.name for s in SKIP_SUBSTRINGS)
    ]


def _slowdown_stats(pdb_ys: List[float], pkdb_ys: List[float]) -> Dict[str, float]:
    ratios = [p / n for n, p in zip(pdb_ys, pkdb_ys) if n != 0]
    if not ratios:
        return {}
    return {
        "avg": sum(ratios) / len(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "median": statistics.median(ratios),
    }


def emit_macros(lines: List[str], prefix: str, plat: Platform) -> None:
    p = f"{prefix}-{plat.benchmark}-{plat.platform_slug}"
    _def_macro(lines, f"{p}-gpu-model", plat.gpu_model.replace("_", r"\_"))
    _def_macro(lines, f"{p}-x-key", plat.x_key)
    for backend in _ordered_backends(plat.series):
        pdb_ys, pkdb_ys = plat.series[backend]
        bp = f"{p}-{backend}"
        for i, xv in enumerate(plat.x_values):
            _def_macro(lines, f"{bp}-{plat.x_key}-{xv}-pdb", _fmt_num(pdb_ys[i]))
            _def_macro(lines, f"{bp}-{plat.x_key}-{xv}-pkdb", _fmt_num(pkdb_ys[i]))
        _def_macro(lines, f"{bp}-pdb-min", _fmt_num(min(pdb_ys)))
        _def_macro(lines, f"{bp}-pdb-max", _fmt_num(max(pdb_ys)))
        _def_macro(lines, f"{bp}-pkdb-min", _fmt_num(min(pkdb_ys)))
        _def_macro(lines, f"{bp}-pkdb-max", _fmt_num(max(pkdb_ys)))
        for stat_name, val in _slowdown_stats(pdb_ys, pkdb_ys).items():
            _def_macro(lines, f"{bp}-{stat_name}-pkdb-slowdown", _fmt_num(val))


def bench_display(bench: str) -> str:
    return BENCH_DISPLAY.get(bench, bench.capitalize())


def _bench_sort_key(bench: str) -> Tuple[int, str]:
    if bench in BENCH_ORDER:
        return (BENCH_ORDER.index(bench), bench)
    return (len(BENCH_ORDER), bench)


def build_table_body(prefix: str, platforms: List[Platform]) -> List[str]:
    by_bench: Dict[str, List[Platform]] = {}
    for plat in platforms:
        by_bench.setdefault(plat.benchmark, []).append(plat)

    lines: List[str] = []
    for bi, bench in enumerate(sorted(by_bench, key=_bench_sort_key)):
        plats = sorted(by_bench[bench], key=lambda pl: pl.platform_slug)
        n_tot = sum(len(pl.series) for pl in plats)
        if bi > 0:
            lines.append(r"    \midrule")
        bench_open = False
        for plat in plats:
            backends = _ordered_backends(plat.series)
            for ri, backend in enumerate(backends):
                bp = f"{prefix}-{plat.benchmark}-{plat.platform_slug}-{backend}"
                cells: List[str] = []
                if not bench_open:
                    cells.append(
                        f"\\multirow{{{n_tot}}}{{*}}{{{bench_display(bench)}}}"
                    )
                    bench_open = True
                else:
                    cells.append("")
                if len(backends) == 1:
                    cells.append(plat.label if ri == 0 else "")
                elif ri == 0:
                    cells.append(f"\\multirow{{{len(backends)}}}{{*}}{{{plat.label}}}")
                else:
                    cells.append("")
                cells.append(backend_display(backend))
                cells.append(
                    f"{{\\UseMacroRound{{{bp}-pdb-min}}{{1}}--\\UseMacroRound{{{bp}-pdb-max}}{{1}}}}"
                )
                cells.append(
                    f"{{\\UseMacroRound{{{bp}-pkdb-min}}{{1}}--\\UseMacroRound{{{bp}-pkdb-max}}{{1}}}}"
                )
                cells.append(f"\\UseMacroRound{{{bp}-avg-pkdb-slowdown}}{{2}}")
                lines.append("    " + " & ".join(cells) + r" \\")
    return lines


TABLE_TEMPLATE = r"""% Auto-generated by report/scripts/eval-sweep-table.py
% Source JSONs: __SOURCES__
\begin{table}[t]
  \centering
  \caption{%
    Debug wall-time sweep ranges (seconds) for __BENCHES__ on __PLATFORMS__.
    Each range is the min--max over the points in the corresponding
    execution-time sweep; the last column is the mean of
    $t_i^{\ourTool}/t_i^{\text{\PDB}}$ over those same points.}
  \label{tab:eval-sweep-summary}
  \scriptsize
  \setlength{\tabcolsep}{2.5pt}
  \setlength{\extrarowheight}{0.5pt}
  \begin{tabular*}{\columnwidth}{@{}lll@{\extracolsep{\fill}}r@{\extracolsep{\fill}}r@{\extracolsep{\fill}}r@{}}
    \toprule
    \textbf{Benchmark}
      & \textbf{Platform}
      & \textbf{Execution space}
      & \makecell[r]{\textbf{\PDB} \textbf{(s)}}
      & \makecell[r]{\textbf{\ourTool} \textbf{(s)}}
      & \makecell[r]{\textbf{Overhead} ($\times$)} \\
    \midrule
__BODY__
    \bottomrule
  \end{tabular*}
\end{table}
"""


def build_table_tex(prefix: str, platforms: List[Platform]) -> str:
    if not platforms:
        return (
            "% Auto-generated by report/scripts/eval-sweep-table.py\n"
            "% No platforms loaded; table omitted.\n"
        )
    body = "\n".join(build_table_body(prefix, platforms))
    benches = sorted({p.benchmark for p in platforms}, key=_bench_sort_key)
    bench_str = ", ".join(bench_display(b) for b in benches)
    plat_str = ", ".join(sorted({p.label for p in platforms}))
    sources = ", ".join(p.path.name for p in platforms)
    return (
        TABLE_TEMPLATE.replace("__BODY__", body)
        .replace("__BENCHES__", bench_str)
        .replace("__PLATFORMS__", plat_str)
        .replace("__SOURCES__", sources)
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir.parent / "tables"
    default_results_dir = script_dir.parent.parent / "benchmarks" / "results"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_paths",
        nargs="*",
        type=Path,
    )
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--macro-prefix", default="eval-test-sweep")
    args = parser.parse_args()

    paths = discover_paths(args.json_paths, args.results_dir.resolve())
    if not paths:
        print("No input JSON files found.", file=sys.stderr)

    platforms: List[Platform] = []
    for path in paths:
        plat = load_platform(path.resolve())
        if plat is not None:
            platforms.append(plat)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_lines: List[str] = [
        "%% Auto-generated by report/scripts/eval-sweep-table.py",
        "%% \\UseMacro{eval-test-sweep-...} in prose and in the sweep table",
        "",
    ]
    for plat in platforms:
        emit_macros(macro_lines, args.macro_prefix, plat)
        macro_lines.append("")

    macro_out = out_dir / "eval-sweep-macros.tex"
    macro_out.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    table_out = out_dir / "eval-sweep-table.tex"
    table_out.write_text(
        build_table_tex(args.macro_prefix, platforms), encoding="utf-8"
    )

    ensure_inputs(
        script_dir.parent / "body.tex",
        ["tables/eval-sweep-macros", "tables/eval-sweep-table"],
    )

    print(f"Loaded {len(platforms)} platform(s) from {len(paths)} input file(s).")
    print(f"Wrote {macro_out}")
    print(f"Wrote {table_out}")


if __name__ == "__main__":
    main()
