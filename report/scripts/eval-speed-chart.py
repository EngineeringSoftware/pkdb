#!/usr/bin/env python3
"""
Execution-time (speed) chart + macros, derived entirely from input JSON paths.

Usage:
  python3 eval-speed-chart.py path/to/a.json path/to/b.json

  # Or scan a directory for every matching *.json (default: ../results,
  # resolved relative to the current working directory):
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

# Known paper macros for benchmark display names (paper/sc26/macros.tex);
# anything else falls back to a plain capitalized string.
BENCH_DISPLAY = {
    "examinimd": r"\Exa",
    "boltzmann": r"\Boltz",
    "ewald": r"\Ewald",
    "parki": r"\Parki",
}
BENCH_ORDER = ("examinimd", "boltzmann", "ewald", "parki")

BACKEND_ORDER = {"openmp_cuda": ("openmp", "cuda"), "hip_pair": ("hip",)}
BACKEND_STYLE_NORM = {
    "openmp": "EvalPlotStyleOpenMP",
    "cuda": "EvalPlotStyleCuda",
    "hip": "EvalPlotStyleHIP",
}
BACKEND_STYLE_PKDB = {
    "openmp": "EvalPlotStylePkdbOpenMP",
    "cuda": "EvalPlotStylePkdbCuda",
    "hip": "EvalPlotStylePkdbHIP",
}
BACKEND_LEGEND_NORM = {"openmp": r"\Openmp", "cuda": r"\Cuda", "hip": r"\Hip"}
BACKEND_LEGEND_PKDB = {
    "openmp": r"\ourTool-\Openmp",
    "cuda": r"\ourTool-\Cuda",
    "hip": r"\ourTool-\Hip",
}


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


def platform_label(platform_stem: str) -> str:
    """Human-readable platform text: GPU name only, e.g. 'x86_64_amd_8x(mi300x)' -> '8XMI300X'."""
    s = platform_stem
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
    return m.group("bench").lower(), m.group("platform")


@dataclass
class Platform:
    path: Path
    benchmark: str
    platform_stem: str
    platform_slug: str
    label: str
    layout: str  # "openmp_cuda" or "hip_pair"
    x_key: str
    x_values: List[int]
    series: Dict[str, Tuple[List[float], List[float]]]  # backend -> (normal, pkdb)


def _debug_value(entry: dict) -> Optional[float]:
    v = entry.get("debug_avg")
    if v is None:
        v = entry.get("pkdb_debug_avg")
    return float(v) if v is not None else None


def _x_key_of(sizes: List[dict]) -> Optional[str]:
    if "atoms" in sizes[0]:
        return "atoms"
    if "steps" in sizes[0]:
        return "steps"
    return None


def _try_openmp_cuda(
    data: dict,
) -> Optional[Tuple[str, List[int], Dict[str, Tuple[List[float], List[float]]]]]:
    sizes = data.get("sizes") or []
    if not sizes:
        return None
    x_key = _x_key_of(sizes)
    if x_key is None:
        return None
    xs: List[int] = []
    omp_n: List[float] = []
    omp_p: List[float] = []
    cu_n: List[float] = []
    cu_p: List[float] = []
    for entry in sizes:
        if x_key not in entry:
            break
        by_backend = {
            b.get("backend", "").lower(): b for b in entry.get("backends", [])
        }
        omp = by_backend.get("openmp")
        cu = by_backend.get("cuda")
        if omp is None or cu is None:
            break
        on, op = omp.get("normal_avg"), _debug_value(omp)
        cn, cp = cu.get("normal_avg"), _debug_value(cu)
        if on is None or op is None or cn is None or cp is None:
            break
        xs.append(int(entry[x_key]))
        omp_n.append(float(on))
        omp_p.append(float(op))
        cu_n.append(float(cn))
        cu_p.append(float(cp))
    if not xs:
        return None
    return x_key, xs, {"openmp": (omp_n, omp_p), "cuda": (cu_n, cu_p)}


def _try_hip(
    data: dict,
) -> Optional[Tuple[str, List[int], Dict[str, Tuple[List[float], List[float]]]]]:
    sizes = data.get("sizes") or []
    if not sizes:
        return None
    x_key = _x_key_of(sizes)
    if x_key is None:
        return None
    xs: List[int] = []
    hn: List[float] = []
    hp: List[float] = []
    for entry in sizes:
        if x_key not in entry:
            break
        by_backend = {
            b.get("backend", "").lower(): b for b in entry.get("backends", [])
        }
        hip = by_backend.get("hip")
        if hip is None:
            break
        na = hip.get("normal_avg")
        if na is None:
            break
        pv = _debug_value(hip)
        if pv is None:
            dh = by_backend.get("debughip")
            pv = dh.get("normal_avg") if dh else None
        if pv is None:
            break
        xs.append(int(entry[x_key]))
        hn.append(float(na))
        hp.append(float(pv))
    if not xs:
        return None
    return x_key, xs, {"hip": (hn, hp)}


def load_platform(path: Path) -> Optional[Platform]:
    parsed = parse_filename(path)
    if parsed is None:
        print(
            f"Skipping {path}: filename doesn't match <benchmark>[_results]_<platform>.json",
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

    layout = "openmp_cuda"
    result = _try_openmp_cuda(data)
    if result is None:
        layout = "hip_pair"
        result = _try_hip(data)
    if result is None:
        print(
            f"Skipping {path}: no paired OpenMP+Cuda or HIP(+DebugHIP) sweep found.",
            file=sys.stderr,
        )
        return None

    x_key, xs, series = result
    return Platform(
        path=path,
        benchmark=bench,
        platform_stem=platform_stem,
        platform_slug=_slugify(platform_stem),
        label=platform_label(platform_stem),
        layout=layout,
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
    return sorted(results_dir.glob("*.json"))


def bench_display(bench: str) -> str:
    return BENCH_DISPLAY.get(bench, bench.capitalize())


def _bench_sort_key(bench: str) -> Tuple[int, str]:
    if bench in BENCH_ORDER:
        return (BENCH_ORDER.index(bench), bench)
    return (len(BENCH_ORDER), bench)


def emit_macros(lines: List[str], prefix: str, plat: Platform) -> None:
    p = f"{prefix}-{plat.benchmark}-{plat.platform_slug}"
    for backend, (norm, pkdb) in plat.series.items():
        for tag, ys in (("norm", norm), ("pkdb", pkdb)):
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
    backends = BACKEND_ORDER[plat.layout]
    all_y = [y for b in backends for series in plat.series[b] for y in series]
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
        norm, pkdb = plat.series[backend]
        lines.append(f"      % {backend} (norm)")
        lines.append(
            f"      \\addplot[{BACKEND_STYLE_NORM[backend]}] "
            + _coordinates(f"{p}-{backend}-norm", n)
            + ";"
        )
        lines.append(f"      \\addlegendentry{{{BACKEND_LEGEND_NORM[backend]}}}")
        lines.append(f"      % {backend} pkdb")
        lines.append(
            f"      \\addplot[{BACKEND_STYLE_PKDB[backend]}] "
            + _coordinates(f"{p}-{backend}-pkdb", n)
            + ";"
        )
        lines.append(f"      \\addlegendentry{{{BACKEND_LEGEND_PKDB[backend]}}}")

    lines.append(r"    \end{axis}")
    lines.append(r"  \end{tikzpicture}")
    cap = (
        f"{bench_display(plat.benchmark)} execution time (Non-debug vs "
        f"\\ourTool{{}}) on {plat.label}."
    )
    lines.append(f"  \\caption{{{cap}}}")
    lines.append(f"  \\label{{fig:eval-speed-{plat.benchmark}-{plat.platform_slug}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def build_chart_tex(prefix: str, platforms: List[Platform]) -> str:
    if not platforms:
        return (
            "% Auto-generated by paper/test/scripts/eval-speed-chart.py\n"
            "% No platforms loaded; chart omitted.\n"
        )
    ordered = sorted(
        platforms, key=lambda pl: (_bench_sort_key(pl.benchmark), pl.platform_slug)
    )
    sources = ", ".join(p.path.name for p in platforms)
    parts = [
        "% Auto-generated by paper/test/scripts/eval-speed-chart.py",
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

    parser = argparse.ArgumentParser(
        description="Generate an execution-time chart + macros from benchmark JSON, no config file needed.",
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
        default=Path("../results"),
        help="Directory to scan for *.json when no paths are given (default: ../results, relative to cwd).",
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
        "%% Auto-generated by paper/test/scripts/eval-speed-chart.py",
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
