#!/usr/bin/env python3
"""
Debug wall-time (speed) chart + macros, derived entirely from input JSON paths.

Reads the `*_debuggers_*.json` shape: one entry per backend per size, holding
`pdb_avg` (plain pdb session) and `pkdb_avg` (pkdb session). The older
`normal_avg`/`debug_avg` shape is still accepted.

Usage:
  python3 eval-speed-chart.py path/to/a.json path/to/b.json

  # Or scan a directory for every matching *.json (default: the repo's
  # benchmarks/results):
  python3 eval-speed-chart.py
  python3 eval-speed-chart.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from report_body import ensure_inputs

FNAME_RE = re.compile(
    r"^(?P<bench>[A-Za-z][A-Za-z0-9]*)_(?:results_)?(?P<platform>.+)$"
)

CPU_ARCH_PREFIXES = ("x86_64_", "x86-64-", "arm64_", "aarch64_")
VENDOR_TOKENS = {"nvidia", "amd", "intel"}
# Result-kind marker sitting between the benchmark name and the platform.
MARKER_PREFIXES = ("debuggers_", "old_vs_pkdb_", "debug_comparison_")
# Files that parse like a sweep but are not a pdb-vs-pkdb one. `old_vs_pkdb`
# carries a non-accelerated Debug baseline; eval-debug-chart.py owns it.
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

BACKEND_ORDER = ("openmp", "cuda", "hip")
BACKEND_STYLE_PDB = {
    "openmp": "EvalPlotStyleOpenMP",
    "cuda": "EvalPlotStyleCuda",
    "hip": "EvalPlotStyleHIP",
}
BACKEND_STYLE_PKDB = {
    "openmp": "EvalPlotStylePkdbOpenMP",
    "cuda": "EvalPlotStylePkdbCuda",
    "hip": "EvalPlotStylePkdbHIP",
}
BACKEND_DISPLAY = {"openmp": r"\Openmp", "cuda": r"\Cuda", "hip": r"\Hip"}


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


def backend_style(backend: str, *, ours: bool) -> str:
    styles = BACKEND_STYLE_PKDB if ours else BACKEND_STYLE_PDB
    return styles.get(backend, styles["openmp"])


def backend_legend(backend: str, *, ours: bool) -> str:
    tool = r"\ourTool" if ours else r"\PDB"
    return f"{tool}-{backend_display(backend)}"


def platform_label(platform_stem: str) -> str:
    """Human-readable platform text: GPU name only, e.g. 'x86_64_amd_8x(mi300x)' -> '8XMI300X'."""
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
    return Platform(
        path=path,
        benchmark=bench,
        platform_stem=platform_stem,
        platform_slug=_slugify(platform_stem),
        label=platform_label(platform_stem),
        x_key=x_key,
        x_values=xs,
        series=series,
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


def bench_display(bench: str) -> str:
    return BENCH_DISPLAY.get(bench, bench.capitalize())


def _bench_sort_key(bench: str) -> Tuple[int, str]:
    if bench in BENCH_ORDER:
        return (BENCH_ORDER.index(bench), bench)
    return (len(BENCH_ORDER), bench)


def emit_macros(lines: List[str], prefix: str, plat: Platform) -> None:
    p = f"{prefix}-{plat.benchmark}-{plat.platform_slug}"
    for backend in _ordered_backends(plat.series):
        pdb_ys, pkdb_ys = plat.series[backend]
        for tag, ys in (("pdb", pdb_ys), ("pkdb", pkdb_ys)):
            bp = f"{p}-{backend}-{tag}"
            for i, y in enumerate(ys):
                _def_macro(lines, f"{bp}-y-{i + 1}", _fmt_num(y))
            _def_macro(lines, f"{bp}-min", _fmt_num(min(ys)))
            _def_macro(lines, f"{bp}-max", _fmt_num(max(ys)))


def _coordinates(macro_prefix: str, n: int) -> str:
    parts = [f"({i + 1},{{\\UseMacro{{{macro_prefix}-y-{i + 1}}}}})" for i in range(n)]
    return "coordinates { " + " ".join(parts) + " }"


def build_figure(prefix: str, plat: Platform) -> str:
    p = f"{prefix}-{plat.benchmark}-{plat.platform_slug}"
    n = len(plat.x_values)
    backends = _ordered_backends(plat.series)
    all_y = [y for b in backends for ys in plat.series[b] for y in ys]
    ymode = "log" if all_y and min(all_y) > 0 else "linear"
    xlabel = "Atoms" if plat.x_key == "atoms" else "Steps"
    xticklabels = ", ".join(str(x) for x in plat.x_values)
    xtick = ", ".join(str(i + 1) for i in range(n))

    lines: List[str] = []
    lines.append(r"\begin{figure}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tikzpicture}")
    lines.append(r"    \begin{axis}[")
    lines.append(r"        width=\linewidth,")
    lines.append(r"        height=0.65\linewidth,")
    lines.append(f"        xlabel={{{xlabel}}},")
    lines.append(r"        ylabel={Time [s]},")
    lines.append(
        r"        ylabel style={at={(axis description cs:-0.07,0.5)},anchor=south},"
    )
    lines.append(f"        xmin=0.5, xmax={n + 0.5},")
    lines.append(f"        xtick={{{xtick}}},")
    lines.append(f"        xticklabels={{{xticklabels}}},")
    if n > 8:
        # Dense sweeps overlap if printed horizontally at column width; rotate instead.
        lines.append(
            r"        xticklabel style={rotate=60, anchor=near xticklabel, font=\tiny},"
        )
    lines.append(
        r"        legend style={at={(0.02,0.98)},anchor=north west,legend columns=2},"
    )
    lines.append(r"        legend cell align={left},")
    lines.append(f"        ymode={ymode},")
    if ymode == "log":
        lines.append(r"        log basis y={10},")
    lines.append(r"        xmajorgrids=true,")
    lines.append(r"        ymajorgrids=true,")
    lines.append(r"        grid style={dashed,gray!45},")
    lines.append(r"        scaled ticks=false,")
    lines.append(r"      ]")

    for backend in backends:
        lines.append(f"      % {backend} (pdb)")
        lines.append(
            f"      \\addplot[{backend_style(backend, ours=False)}] "
            + _coordinates(f"{p}-{backend}-pdb", n)
            + ";"
        )
        lines.append(f"      \\addlegendentry{{{backend_legend(backend, ours=False)}}}")
        lines.append(f"      % {backend} (pkdb)")
        lines.append(
            f"      \\addplot[{backend_style(backend, ours=True)}] "
            + _coordinates(f"{p}-{backend}-pkdb", n)
            + ";"
        )
        lines.append(f"      \\addlegendentry{{{backend_legend(backend, ours=True)}}}")

    lines.append(r"    \end{axis}")
    lines.append(r"  \end{tikzpicture}")
    cap = (
        f"{bench_display(plat.benchmark)} debug wall time (\\PDB{{}} vs "
        f"\\ourTool{{}}) on {plat.label}."
    )
    lines.append(f"  \\caption{{{cap}}}")
    lines.append(f"  \\label{{fig:eval-speed-{plat.benchmark}-{plat.platform_slug}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def build_chart_tex(prefix: str, platforms: List[Platform]) -> str:
    if not platforms:
        return (
            "% Auto-generated by report/scripts/eval-speed-chart.py\n"
            "% No platforms loaded; chart omitted.\n"
        )
    ordered = sorted(
        platforms, key=lambda pl: (_bench_sort_key(pl.benchmark), pl.platform_slug)
    )
    sources = ", ".join(p.path.name for p in platforms)
    parts = [
        "% Auto-generated by report/scripts/eval-speed-chart.py",
        f"% Source JSONs: {sources}",
        "% Requires: \\usepackage{tikz}, \\usepackage{pgfplots}",
        "",
    ]
    for plat in ordered:
        parts.append(build_figure(prefix, plat))
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_tables_dir = script_dir.parent / "tables"
    default_figures_dir = script_dir.parent / "figures"
    default_results_dir = script_dir.parent.parent / "benchmarks" / "results"

    parser = argparse.ArgumentParser(
        description="Generate a debug wall-time chart + macros from benchmark JSON, no config file needed.",
    )
    parser.add_argument(
        "json_paths",
        nargs="*",
        type=Path,
        help="Explicit benchmark result JSON paths. If omitted, scans --results-dir for *.json.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"Directory to scan for *.json when no paths are given (default: {default_results_dir}).",
    )
    parser.add_argument(
        "--macros-out-dir",
        type=Path,
        default=default_tables_dir,
        help=f"Directory to write the macros .tex into (default: {default_tables_dir}).",
    )
    parser.add_argument(
        "--figures-out-dir",
        type=Path,
        default=default_figures_dir,
        help=f"Directory to write the chart .tex into (default: {default_figures_dir}).",
    )
    parser.add_argument(
        "--macro-prefix",
        default="eval-test-speed",
        help="Base prefix for generated \\DefMacro names (default: eval-test-speed).",
    )
    args = parser.parse_args()

    paths = discover_paths(args.json_paths, args.results_dir.resolve())
    if not paths:
        print("No input JSON files found.", file=sys.stderr)

    platforms: List[Platform] = []
    for path in paths:
        plat = load_platform(path.resolve())
        if plat is not None:
            platforms.append(plat)

    macros_out_dir = args.macros_out_dir.resolve()
    macros_out_dir.mkdir(parents=True, exist_ok=True)
    figures_out_dir = args.figures_out_dir.resolve()
    figures_out_dir.mkdir(parents=True, exist_ok=True)

    macro_lines: List[str] = [
        "%% Auto-generated by report/scripts/eval-speed-chart.py",
        "%% \\UseMacro{eval-test-speed-...} in the pgfplots coordinates",
        "",
    ]
    for plat in platforms:
        emit_macros(macro_lines, args.macro_prefix, plat)
        macro_lines.append("")

    macro_out = macros_out_dir / "eval-speed-macros.tex"
    macro_out.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    chart_out = figures_out_dir / "eval-speed-chart.tex"
    chart_out.write_text(build_chart_tex(args.macro_prefix, platforms), encoding="utf-8")

    ensure_inputs(
        script_dir.parent / "body.tex",
        ["tables/eval-speed-macros", "figures/eval-speed-chart"],
    )

    print(f"Loaded {len(platforms)} platform(s) from {len(paths)} input file(s).")
    print(f"Wrote {macro_out}")
    print(f"Wrote {chart_out}")


if __name__ == "__main__":
    main()
