#!/usr/bin/env python3
"""
Debug-run execution-time chart + macros: old (interpreted) debugger vs the
new pkdb-instrumented debug run, per backend. Derived entirely from
input JSON paths, same idea as eval-sweep-table.py / eval-speed-chart.py.

Reads the `*_old_vs_pkdb_*.json` shape: the `Debug` backend's `pdb_avg` is the
baseline (plain pdb, no accelerator), and each `Debug<Space>` backend's
`pkdb_avg` is the pkdb run. The older `normal_avg`-only shape still works.

Usage:
  # Explicit files:
  python3 eval-debug-chart.py path/to/a.json path/to/b.json

  # Or scan a directory for every matching *.json (default: the repo's
  # benchmarks/results):
  python3 eval-debug-chart.py
  python3 eval-debug-chart.py --results-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from report_body import ensure_inputs

FNAME_RE = re.compile(
    r"^(?P<bench>[A-Za-z][A-Za-z0-9]*)_(?:results_)?(?P<platform>.+)$"
)

CPU_ARCH_PREFIXES = ("x86_64_", "x86-64-", "arm64_", "aarch64_")
VENDOR_TOKENS = {"nvidia", "amd", "intel"}
MARKER_PREFIXES = (
    "old_vs_pkdb_",
    "debuggers_",
    "debug_comparison_",
    "debug-comparison-",
)
# Files with no old-debugger baseline in them; eval-speed-chart.py owns the
# pdb-vs-pkdb sweeps.
SKIP_SUBSTRINGS = ("_debuggers_", "kernel_profile", "hotswap", "steps_result")

# Known paper macros for benchmark display names (paper/sc26/macros.tex);
# anything else falls back to a plain capitalized string.
BENCH_DISPLAY = {
    "examinimd": r"\Exa",
    "boltzmann": r"\Boltz",
    "ewald": r"\Ewald",
    "parki": r"\Parki",
}
BENCH_ORDER = ("examinimd", "boltzmann", "ewald", "parki")

# backend key (lowercase, "debug" prefix stripped) -> (pgfplots style, legend, macro slug)
KNOWN_VARIANTS: Dict[str, Tuple[str, str, str]] = {
    "openmp": ("EvalPlotStylePkdbOpenMP", r"\ourTool-\Openmp", "pkdbopenmp"),
    "cuda": ("EvalPlotStylePkdbCuda", r"\ourTool-\Cuda", "pkdbcuda"),
    "hip": ("EvalPlotStylePkdbHIP", r"\ourTool-\Hip", "pkdbhip"),
}
FALLBACK_VARIANT_STYLES = (
    "EvalPlotStylePkdbOpenMP",
    "EvalPlotStylePkdbCuda",
    "EvalPlotStylePkdbHIP",
)
BASELINE_STYLE = "EvalPlotStyleBaseline"
BASELINE_LEGEND = "Baseline"
BASELINE_SLUG = "baseline"


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
    baseline: List[float]
    variants: Dict[str, List[float]] = field(default_factory=dict)  # key -> y series
    variant_names: Dict[str, str] = field(default_factory=dict)  # key -> original backend name


def _series_value(entry: dict, *, baseline: bool) -> Optional[float]:
    """Baseline takes pdb_avg, a pkdb variant takes pkdb_avg; older JSONs only have normal_avg."""
    v = entry.get("pdb_avg") if baseline else entry.get("pkdb_avg")
    if v is None:
        v = entry.get("normal_avg")
    return float(v) if v is not None else None


def _x_key_of(sizes: List[dict]) -> Optional[str]:
    if "atoms" in sizes[0]:
        return "atoms"
    if "steps" in sizes[0]:
        return "steps"
    return None


def _try_debug_compare(
    data: dict,
) -> Optional[Tuple[str, List[int], Dict[str, List[float]], Dict[str, str]]]:
    sizes = data.get("sizes") or []
    if not sizes:
        return None
    x_key = _x_key_of(sizes)
    if x_key is None:
        return None

    first_map = {b.get("backend", ""): b for b in sizes[0].get("backends", [])}
    names_by_key = {name.lower(): name for name in first_map}
    debug_keys = [k for k in names_by_key if k == "debug" or k.startswith("debug")]
    if "debug" not in debug_keys or len(debug_keys) < 2:
        return None

    xs: List[int] = []
    series: Dict[str, List[float]] = {k: [] for k in debug_keys}
    for entry in sizes:
        if x_key not in entry:
            break
        by_backend = {
            b.get("backend", "").lower(): b for b in entry.get("backends", [])
        }
        row_vals: Dict[str, float] = {}
        row_ok = True
        for k in debug_keys:
            b = by_backend.get(k)
            v = _series_value(b, baseline=(k == "debug")) if b else None
            if v is None:
                row_ok = False
                break
            row_vals[k] = float(v)
        if not row_ok:
            break
        xs.append(int(entry[x_key]))
        for k in debug_keys:
            series[k].append(row_vals[k])

    if not xs:
        return None
    return x_key, xs, series, names_by_key


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

    result = _try_debug_compare(data)
    if result is None:
        print(
            f"Skipping {path}: no 'Debug' baseline + \\ourTool-instrumented 'Debug*' sweep found.",
            file=sys.stderr,
        )
        return None
    x_key, xs, series, names_by_key = result

    baseline = series.pop("debug")
    variant_names = {k: names_by_key[k] for k in series}

    return Platform(
        path=path,
        benchmark=bench,
        platform_stem=platform_stem,
        platform_slug=_slugify(platform_stem),
        label=platform_label(platform_stem),
        x_key=x_key,
        x_values=xs,
        baseline=baseline,
        variants=series,
        variant_names=variant_names,
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


def _variant_meta(key: str, original_name: str, fallback_index: int) -> Tuple[str, str, str]:
    """(pgfplots style, legend text, macro slug) for a 'debug<suffix>' backend key."""
    suffix = key[len("debug") :]
    if suffix in KNOWN_VARIANTS:
        return KNOWN_VARIANTS[suffix]
    style = FALLBACK_VARIANT_STYLES[fallback_index % len(FALLBACK_VARIANT_STYLES)]
    display_suffix = original_name[len("Debug") :] if original_name.lower().startswith("debug") else original_name
    return style, f"\\ourTool-{display_suffix}", f"pkdb{suffix or _slugify(original_name)}"


def emit_macros(lines: List[str], prefix: str, plat: Platform) -> None:
    p = f"{prefix}-{plat.benchmark}-{plat.platform_slug}"
    for i, y in enumerate(plat.baseline):
        _def_macro(lines, f"{p}-{BASELINE_SLUG}-y-{i + 1}", _fmt_num(y))
    _def_macro(lines, f"{p}-{BASELINE_SLUG}-min", _fmt_num(min(plat.baseline)))
    _def_macro(lines, f"{p}-{BASELINE_SLUG}-max", _fmt_num(max(plat.baseline)))

    for idx, (key, ys) in enumerate(sorted(plat.variants.items())):
        _, _, slug = _variant_meta(key, plat.variant_names[key], idx)
        for i, y in enumerate(ys):
            _def_macro(lines, f"{p}-{slug}-y-{i + 1}", _fmt_num(y))
        _def_macro(lines, f"{p}-{slug}-min", _fmt_num(min(ys)))
        _def_macro(lines, f"{p}-{slug}-max", _fmt_num(max(ys)))


def _coordinates(macro_prefix: str, n: int) -> str:
    parts = [f"({i + 1},{{\\UseMacro{{{macro_prefix}-y-{i + 1}}}}})" for i in range(n)]
    return "coordinates { " + " ".join(parts) + " }"


def build_figure(prefix: str, plat: Platform) -> str:
    p = f"{prefix}-{plat.benchmark}-{plat.platform_slug}"
    n = len(plat.x_values)
    all_y = list(plat.baseline) + [y for ys in plat.variants.values() for y in ys]
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

    lines.append(r"      % Debug (old, baseline)")
    lines.append(
        f"      \\addplot[{BASELINE_STYLE}] "
        + _coordinates(f"{p}-{BASELINE_SLUG}", n)
        + ";"
    )
    lines.append(f"      \\addlegendentry{{{BASELINE_LEGEND}}}")

    for idx, key in enumerate(sorted(plat.variants)):
        style, legend, slug = _variant_meta(key, plat.variant_names[key], idx)
        lines.append(f"      % {plat.variant_names[key]} (new, \\ourTool-instrumented)")
        lines.append(
            f"      \\addplot[{style}] " + _coordinates(f"{p}-{slug}", n) + ";"
        )
        lines.append(f"      \\addlegendentry{{{legend}}}")

    lines.append(r"    \end{axis}")
    lines.append(r"  \end{tikzpicture}")
    cap = (
        f"{bench_display(plat.benchmark)} debug-run exec. time "
        f"(old \\PDB{{}} debugger vs \\ourTool{{}}) on {plat.label}."
    )
    lines.append(f"  \\caption{{{cap}}}")
    lines.append(f"  \\label{{fig:eval-debug-{plat.benchmark}-{plat.platform_slug}}}")
    lines.append(r"\end{figure}")
    return "\n".join(lines)


def build_chart_tex(prefix: str, platforms: List[Platform]) -> str:
    if not platforms:
        return (
            "% Auto-generated by report/scripts/eval-debug-chart.py\n"
            "% No platforms loaded; chart omitted.\n"
        )
    ordered = sorted(
        platforms, key=lambda pl: (_bench_sort_key(pl.benchmark), pl.platform_slug)
    )
    sources = ", ".join(p.path.name for p in platforms)
    parts = [
        "% Auto-generated by report/scripts/eval-debug-chart.py",
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
        description="Generate a debug-run (old vs \\ourTool) exec-time chart + macros from benchmark JSON, no config file needed.",
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
        default="eval-test-debug",
        help="Base prefix for generated \\DefMacro names (default: eval-test-debug).",
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
        "%% Auto-generated by report/scripts/eval-debug-chart.py",
        "%% \\UseMacro{eval-test-debug-...} in the pgfplots coordinates",
        "",
    ]
    for plat in platforms:
        emit_macros(macro_lines, args.macro_prefix, plat)
        macro_lines.append("")

    macro_out = macros_out_dir / "eval-debug-macros.tex"
    macro_out.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    chart_out = figures_out_dir / "eval-debug-chart.tex"
    chart_out.write_text(build_chart_tex(args.macro_prefix, platforms), encoding="utf-8")

    ensure_inputs(
        script_dir.parent / "body.tex",
        ["tables/eval-debug-macros", "figures/eval-debug-chart"],
    )

    print(f"Loaded {len(platforms)} platform(s) from {len(paths)} input file(s).")
    print(f"Wrote {macro_out}")
    print(f"Wrote {chart_out}")


if __name__ == "__main__":
    main()
