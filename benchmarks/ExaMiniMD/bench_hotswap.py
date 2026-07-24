from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pexpect

_BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS_ROOT))
from helpers import collect_system_info, system_suffix

_RESULTS_DIR = _BENCHMARKS_ROOT / "results"


def _hotswap_results_json_path(system_info: dict) -> Path:
    return _RESULTS_DIR / f"hotswap_benchmark_{system_suffix(system_info)}.json"


PKDB_PROMPT = r"\(pkdb\)\s*"
# NVE presets probe v[0][0]; alternate workunits nullify velocity (not 0.5x scaling).
_V00_NULL_ABS_TOL = 1e-5
_V00_HOTSWAP_MATCH_TOL = 1e-5
_MIN_LINES_BEFORE_PARALLEL = 2

# Included in --json-out so scenario keys are self-explanatory (JSON has no comments).
_JSON_SCENARIO_LEGEND: dict[str, str] = {
    "scenario_A_edit_rerun": (
        "Scenario A: edit + rerun - change the Python source on disk and run two separate pkdb sessions "
        "(full workunit, then alternate workunit)."
    ),
    "scenario_B_hotswap": (
        "Scenario B: pkdb hotswap - one pkdb session; swap the workunit with the pkdb `hotswap` command "
        "without restarting the process."
    ),
}

# --preset bundles (paths relative to ExaMiniMD/src). "temperature" also doubles as the
# default when no --preset and no override is given (see _resolve_run_config).
_BENCH_PRESETS: dict[str, dict[str, Any]] = {
    "temperature": {
        "prop": "property_temperature.py",
        "class": "Temperature",
        "method": "compute",
        "probe": "T",
        "label": "T",
        "hs_s": "compute_workunit",
        "hs_t": "compute_workunit_half",
        "ef": "compute_workunit,",
        "eh": "compute_workunit_half,",
        "line_pre": 40,
        "line_inspect": 53,
        "hotswap_line": 42,
    },
    "kine": {
        "prop": "property_kine.py",
        "class": "KinE",
        "method": "compute",
        "probe": "KE",
        "label": "KE",
        "hs_s": "work",
        "hs_t": "work_half",
        "ef": "work,",
        "eh": "work_half,",
        "line_pre": 37,
        "line_inspect": 46,
        "hotswap_line": 41,
    },
    "nve-initial": {
        "prop": "integrator_nve.py",
        "class": "IntegratorNVE",
        "method": "initial_integrate",
        "probe": "self.system.v[0][0]",
        "label": "v00",
        "hs_s": "initial_integrate",
        "hs_t": "initial_integrate_half",
        "ef": "initial_integrate,",
        "eh": "initial_integrate_half,",
        "line_pre": 90,
        "line_inspect": 108,
        "hotswap_line": 94,
    },
    "nve-final": {
        "prop": "integrator_nve.py",
        "class": "IntegratorNVE",
        "method": "final_integrate",
        "probe": "self.system.v[0][0]",
        "label": "v00",
        "hs_s": "final_integrate",
        "hs_t": "final_integrate_half",
        "ef": "final_integrate,",
        "eh": "final_integrate_half,",
        "line_pre": 110,
        "line_inspect": 126,
        "hotswap_line": 114,
    },
}
_DEFAULT_PRESET = "temperature"


# --------------------------------------------------------------------------------------
# Parse a numeric value out of pdb's `p <expr>` output.
# --------------------------------------------------------------------------------------

# Float literal inside pdb reprs (bare, scientific, optional fraction).
_FLOAT_LIT = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMPY_FLOAT_WRAP = re.compile(
    rf"(?:numpy|np)\.float(?:32|64)\(\s*({_FLOAT_LIT}|nan|[-+]?(?:inf|infinity))\s*\)",
    re.IGNORECASE,
)
_SPECIAL_FLOAT_WORD = re.compile(r"^(nan|[-+]?(?:inf|infinity))$", re.IGNORECASE)


def _last_value_in_range(text: str, lo: float, hi: float) -> float | None:
    """
    Last float literal in `text` within [lo, hi] (salvage path when no line matched a probe pattern).
    """
    in_range = [
        v
        for v in (float(m.group(0)) for m in re.finditer(_FLOAT_LIT, text))
        if lo <= v <= hi
    ]
    return in_range[-1] if in_range else None


def _parse_float_from_pdb(
    before: str, *, pdb_probe: str, probe_label: str
) -> float | None:
    """
    Parse a numeric from `p <pdb_probe>` output (bare float, `np.float64(...)`, or `expr = ...`).
    """
    text = re.sub(r"\x1b\[[0-9;]*[mK]", "", before.replace("\r\n", "\n"))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    bare = re.compile(rf"^({_FLOAT_LIT})$")
    assign = re.compile(rf"{re.escape(pdb_probe.strip())}\s*=\s*({_FLOAT_LIT})")
    float_call = re.compile(rf"\bfloat\(({_FLOAT_LIT})\)\s*$")
    patterns = (_NUMPY_FLOAT_WRAP, _SPECIAL_FLOAT_WORD, bare, assign, float_call)

    for line in reversed(lines):
        if line.startswith("(pkdb)") or line.startswith("->"):
            continue
        for pattern in patterns:
            m = pattern.search(line)
            if m:
                return float(m.group(1))

    if probe_label == "T":
        # No structured match anywhere; salvage the last plausible temperature (K) printed at all.
        return _last_value_in_range(text, 90.0, 250.0)
    return None


# --------------------------------------------------------------------------------------
# Drive a pkdb session over pexpect.
# --------------------------------------------------------------------------------------
def _spawn_pkdb(*, cwd: Path, exa_args: list[str], env: dict[str, str], logfile):
    child = pexpect.spawn(
        sys.executable,
        ["-m", "pkdb", "main.py", *exa_args],
        cwd=str(cwd),
        env=env,
        encoding="utf-8",
        timeout=36000,
        maxread=200000,
    )
    child.delaybeforesend = 0.05
    if logfile is not None:
        child.logfile_read = logfile
    return child


def _expect_prompt(child, label: str, heartbeat_interval: float | None) -> None:
    stop = threading.Event()
    th = None
    if heartbeat_interval and heartbeat_interval > 0:
        t0 = time.perf_counter()

        def hb() -> None:
            while not stop.wait(heartbeat_interval):
                print(
                    f"[bench] waiting <{label}> {time.perf_counter() - t0:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )

        th = threading.Thread(target=hb, daemon=True)
        th.start()
    try:
        i = child.expect([PKDB_PROMPT, pexpect.EOF, pexpect.TIMEOUT])
        if i == 0:
            return
        if i == 1:
            raise RuntimeError(f"pkdb EOF during <{label}>\n{child.before}")
        raise RuntimeError(f"timeout <{label}>\n{child.before[-4000:]}")
    finally:
        stop.set()
        if th:
            th.join(timeout=2.0)


def _quit_pkdb(child) -> None:
    child.sendline("q")
    try:
        child.expect(pexpect.EOF, timeout=120)
    except pexpect.TIMEOUT:
        child.close(force=True)


def _num(v: object, key: str) -> float:
    if not isinstance(v, (int, float)):
        raise TypeError(f"expected number for {key!r}, got {v!r}")
    return float(v)


def _print_rows(title: str, data: dict[str, Any]) -> None:
    print(title)
    for k in sorted(data):
        v = data[k]
        if k.endswith("_s") and isinstance(v, (int, float)):
            print(f"  {k}: {float(v):.6f}")
        elif isinstance(v, (int, float)):
            print(f"  {k}: {float(v):.6f}")
        elif v is None:
            print(f"  {k}: (none)")
        else:
            print(f"  {k}: {v}")
    print()


# --------------------------------------------------------------------------------------
# Verify the probed values agree with what each scenario is expected to produce.
# --------------------------------------------------------------------------------------


def _verify_edit_rerun(e: dict[str, Any], *, probe_label: str) -> bool:
    """
    Scenario A: run1 (full kernel) vs run2 (alternate kernel); ~2:1 ratio, or run2 ~ 0 for null workunits.
    """
    k1, k2 = f"{probe_label}_run1_full_kernel", f"{probe_label}_run2_half_kernel"
    t1, t2 = e.get(k1), e.get(k2)
    null_v00 = probe_label == "v00"
    print(f"  (1a) full kernel run:  {t1}")
    print(
        f"  (1b) {'null workunit in source' if null_v00 else 'half in source'}:   {t2}"
    )
    if t1 is None or t2 is None:
        print(f"[verify] missing {k1}/{k2}")
        return False
    if null_v00:
        if abs(t2) <= _V00_NULL_ABS_TOL:
            print(f"[verify] {probe_label} run2 ~ 0 (null), run1={t1:g}")
            return True
        print(
            f"[verify] {probe_label} run2 expected ~0 (null), got {t2:g} (tol {_V00_NULL_ABS_TOL:g})"
        )
        return False
    if abs(t2 / t1 - 0.5) <= 0.22:
        print(f"[verify] {probe_label} run1/run2 ~ 2:1 ({t1:g} vs {t2:g})")
        return True
    print(f"[verify] bad ratio {probe_label} run1={t1:g} run2={t2:g}")
    return False


def _verify_hotswap(h: dict[str, Any], *, probe_label: str) -> bool:
    """
    Scenario B: the hotswapped probe value is present, and ~0 for null workunits (v00).
    """
    kh = f"{probe_label}_hotswap"
    th = h.get(kh)
    print(f"  (2)  hotswap path:     {th}")
    if th is None:
        print(f"[verify] missing {kh} (parse pdb output)")
        return False
    if probe_label == "v00":
        if abs(th) <= _V00_NULL_ABS_TOL:
            print(f"[verify] {probe_label} hotswap ~ 0 (null)")
            return True
        print(
            f"[verify] {probe_label} hotswap expected ~0 (null), got {th:g} (tol {_V00_NULL_ABS_TOL:g})"
        )
        return False
    return True


def _verify_hotswap_matches_run2(
    h: dict[str, Any], e: dict[str, Any], *, probe_label: str
) -> bool:
    """
    Cross-check: scenario B's probed value should agree with scenario A's alternate-workunit run.
    """
    kh, k2 = f"{probe_label}_hotswap", f"{probe_label}_run2_half_kernel"
    th, t2 = h.get(kh), e.get(k2)
    if th is None or t2 is None:
        return True  # already flagged as missing by the per-scenario checks above
    if probe_label == "v00":
        delta = abs(th - t2)
        if delta <= _V00_HOTSWAP_MATCH_TOL:
            print(f"[verify] hotswap vs run2 null match (|delta|={delta:.4g})")
            return True
        print(
            f"[verify] hotswap vs run2 |delta|={delta:.4g} (want <= {_V00_HOTSWAP_MATCH_TOL:g})"
        )
        return False
    rel = abs(th - t2) / max(abs(t2), 1e-12)
    if rel >= 0.12:
        print(f"[verify] hotswap vs run2 rel={rel:.4f} (want <0.12)")
        return False
    print(f"[verify] hotswap matches source half (rel {rel:.4f})")
    return True


def _verify_print(
    h: dict[str, Any] | None,
    e: dict[str, Any] | None,
    *,
    verify: bool,
    probe_label: str,
) -> bool:
    if not verify:
        print("[verify] off (--no-verify)")
        return True

    ok = True
    if h or e:
        print(f"--- probe {probe_label} (p at inspect line, after parallel_*) ---")
    if e:
        ok = _verify_edit_rerun(e, probe_label=probe_label) and ok
    if h:
        ok = _verify_hotswap(h, probe_label=probe_label) and ok
    if h is not None and e is not None:
        ok = _verify_hotswap_matches_run2(h, e, probe_label=probe_label) and ok
    print()
    return ok


# --------------------------------------------------------------------------------------
# Scenario B: hotswap. One pkdb session; swap the workunit mid-run via `hotswap`.
# --------------------------------------------------------------------------------------


def run_scenario_hotswap(
    *,
    cwd: Path,
    prop_abs: str,
    line_pre: int,
    line_inspect: int,
    hotswap_line: int,
    hotswap_source: str,
    hotswap_target: str,
    pdb_probe: str,
    probe_label: str,
    exa_args: list[str],
    env: dict[str, str],
    logfile,
    heartbeat_interval: float | None,
    verify: bool,
) -> dict[str, Any]:
    def xp(c, lab: str) -> None:
        _expect_prompt(c, lab, heartbeat_interval)

    print(
        "[bench] pkdb (hotswap): first prompt may be slow (precompile)",
        file=sys.stderr,
        flush=True,
    )
    child = _spawn_pkdb(cwd=cwd, exa_args=exa_args, env=env, logfile=logfile)
    xp(child, "setup")
    child.sendline(f"b {prop_abs}:{line_pre}")
    xp(child, "b pre-reduce")
    child.sendline(f"b {prop_abs}:{line_inspect}")
    xp(child, "b inspect")
    child.sendline("r")
    t0 = time.perf_counter()
    xp(child, "pre-reduce")
    t_pre = time.perf_counter()
    t_hs0 = time.perf_counter()
    hs_cmd = f"hotswap {hotswap_source} {hotswap_target} {prop_abs}:{hotswap_line}"
    child.sendline(hs_cmd)
    xp(child, "hotswap")
    t_hs1 = time.perf_counter()
    child.sendline(f"clear {prop_abs}:{line_pre}")
    xp(child, "clear pre bp")
    child.sendline("c")
    xp(child, "inspect")
    t_fac = time.perf_counter()
    val_hs: float | None = None
    if verify:
        child.sendline(f"p {pdb_probe}")
        xp(child, f"p {pdb_probe}")
        val_hs = _parse_float_from_pdb(
            child.before, pdb_probe=pdb_probe, probe_label=probe_label
        )
    _quit_pkdb(child)
    out: dict[str, float | None | str] = {
        "to_print_s": t_pre - t0,
        "hotswap_cmd_s": t_hs1 - t_hs0,
        "print_to_factor_s": t_fac - t_pre,
        "total_r_to_factor_s": t_fac - t0,
        "hotswap_cli": hs_cmd,
        "pdb_probe": pdb_probe,
        f"{probe_label}_hotswap": val_hs,
    }
    return out


# --------------------------------------------------------------------------------------
# Scenario A: edit + rerun. Two separate pkdb sessions, source edited on disk in between.
# --------------------------------------------------------------------------------------
def _one_edit_rerun_pkdb(
    *,
    cwd: Path,
    prop_abs: str,
    line_inspect: int,
    pdb_probe: str,
    probe_label: str,
    exa_args: list[str],
    env: dict[str, str],
    logfile,
    heartbeat_interval: float | None,
    verify: bool,
    stderr_note: str,
) -> tuple[float, float | None]:
    def xp(c, lab: str) -> None:
        _expect_prompt(c, lab, heartbeat_interval)

    print(stderr_note, file=sys.stderr, flush=True)
    child = _spawn_pkdb(cwd=cwd, exa_args=exa_args, env=env, logfile=logfile)
    xp(child, "initial")
    child.sendline(f"b {prop_abs}:{line_inspect}")
    xp(child, "b inspect")
    child.sendline("r")
    t0 = time.perf_counter()
    xp(child, "inspect")
    t1 = time.perf_counter()
    val: float | None = None
    if verify:
        child.sendline(f"p {pdb_probe}")
        xp(child, f"p {pdb_probe}")
        val = _parse_float_from_pdb(
            child.before, pdb_probe=pdb_probe, probe_label=probe_label
        )
    _quit_pkdb(child)
    return t1 - t0, val


def _ensure_full_kernel_selected(
    text: str, *, edit_full_token: str, edit_half_token: str
) -> str:
    if text.count(edit_full_token) == 0 and text.count(edit_half_token) == 1:
        return text.replace(edit_half_token, edit_full_token, 1)
    return text


def run_scenario_edit_rerun(
    *,
    cwd: Path,
    prop_path: Path,
    prop_abs: str,
    line_inspect: int,
    pdb_probe: str,
    probe_label: str,
    edit_full_token: str,
    edit_half_token: str,
    exa_args: list[str],
    env: dict[str, str],
    logfile,
    heartbeat_interval: float | None,
    verify: bool,
) -> dict[str, float | None]:
    backup = prop_path.read_text(encoding="utf-8")
    try:
        normalized = _ensure_full_kernel_selected(
            backup, edit_full_token=edit_full_token, edit_half_token=edit_half_token
        )
        if normalized != backup:
            prop_path.write_text(normalized, encoding="utf-8")

        run1_s, v1 = _one_edit_rerun_pkdb(
            cwd=cwd,
            prop_abs=prop_abs,
            line_inspect=line_inspect,
            pdb_probe=pdb_probe,
            probe_label=probe_label,
            exa_args=exa_args,
            env=env,
            logfile=logfile,
            heartbeat_interval=heartbeat_interval,
            verify=verify,
            stderr_note="[bench] pkdb edit-rerun 1/2",
        )

        text = prop_path.read_text(encoding="utf-8")
        if text.count(edit_full_token) != 1:
            raise RuntimeError(f"Need exactly one `{edit_full_token}` in {prop_path}")
        prop_path.write_text(
            text.replace(edit_full_token, edit_half_token, 1), encoding="utf-8"
        )

        run2_s, v2 = _one_edit_rerun_pkdb(
            cwd=cwd,
            prop_abs=prop_abs,
            line_inspect=line_inspect,
            pdb_probe=pdb_probe,
            probe_label=probe_label,
            exa_args=exa_args,
            env=env,
            logfile=logfile,
            heartbeat_interval=heartbeat_interval,
            verify=verify,
            stderr_note="[bench] pkdb edit-rerun 2/2",
        )
        return {
            "run1_to_factor_s": run1_s,
            "run2_to_factor_s": run2_s,
            "sum_two_runs_s": run1_s + run2_s,
            f"{probe_label}_run1_full_kernel": v1,
            f"{probe_label}_run2_half_kernel": v2,
        }
    finally:
        prop_path.write_text(backup, encoding="utf-8")


# --------------------------------------------------------------------------------------
# Resolve one run's configuration: target file/class/method + probe/hotswap/edit tokens.
# Precedence is always explicit CLI flag > preset (--preset, or "temperature" as the default).
# --------------------------------------------------------------------------------------
@dataclass
class RunConfig:
    cwd: Path
    prop_path: Path
    prop_abs: str
    class_name: str
    method_name: str
    line_pre: int
    line_inspect: int
    hotswap_line: int
    pdb_probe: str
    probe_label: str
    hotswap_source: str
    hotswap_target: str
    edit_full_token: str
    edit_half_token: str
    exa_args: list[str]
    env: dict[str, str]


def _resolve_breakpoints(
    args: argparse.Namespace, pr: dict[str, Any]
) -> tuple[int, int, int]:
    """
    (line_pre, line_inspect, hotswap_line); each is a fixed line number in the preset's source
    file (see `_BENCH_PRESETS`), overridable for a custom (non-preset) target.
    """
    line_pre = args.line_pre if args.line_pre is not None else pr["line_pre"]
    line_inspect = (
        args.line_inspect if args.line_inspect is not None else pr["line_inspect"]
    )
    hotswap_line = (
        args.hotswap_line if args.hotswap_line is not None else pr["hotswap_line"]
    )
    return line_pre, line_inspect, hotswap_line


def _resolve_probe(args: argparse.Namespace, pr: dict[str, Any]) -> tuple[str, str]:
    pdb_probe = (args.pdb_probe or pr["probe"]).strip()
    probe_label = (args.probe_label or pr["label"]).strip()
    return pdb_probe, probe_label


def _resolve_hotswap_tokens(
    args: argparse.Namespace, pr: dict[str, Any]
) -> tuple[str, str]:
    if args.hotswap_source and args.hotswap_target:
        return args.hotswap_source.strip(), args.hotswap_target.strip()
    return pr["hs_s"], pr["hs_t"]


def _resolve_edit_tokens(
    args: argparse.Namespace, pr: dict[str, Any]
) -> tuple[str, str]:
    if args.edit_full_token and args.edit_half_token:
        return args.edit_full_token, args.edit_half_token
    return pr["ef"], pr["eh"]


def _resolve_run_config(
    args: argparse.Namespace, effective_preset: str | None
) -> RunConfig:
    """
    Resolve one run's target and tokens. `effective_preset` is a key in `_BENCH_PRESETS` or
    `None` for a custom target (unset fields fall back to the "temperature" preset).
    """
    pr = _BENCH_PRESETS[effective_preset or _DEFAULT_PRESET]
    cwd = args.cwd.resolve()
    prop_path = (args.property_file or cwd / pr["prop"]).resolve()
    if not prop_path.is_file():
        raise FileNotFoundError(f"Not found: {prop_path}")

    class_name = (args.class_name or pr["class"]).strip()
    method_name = (args.method_name or pr["method"]).strip()
    line_pre, line_inspect, hotswap_line = _resolve_breakpoints(args, pr)
    pdb_probe, probe_label = _resolve_probe(args, pr)
    hotswap_source, hotswap_target = _resolve_hotswap_tokens(args, pr)
    edit_full_token, edit_half_token = _resolve_edit_tokens(args, pr)

    exa_args = (
        list(args.exa_arg)
        if args.exa_arg
        else ["-il", "../input/in.lj", "--fill", "--atoms", str(args.atoms)]
    )
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if "PK_EXEC_SPACE" not in env and "PK_EXA_SPACE" not in env:
        env["PK_EXEC_SPACE"] = env["PK_EXA_SPACE"] = "DebugCuda"
        print("[bench] PK_EXEC_SPACE=PK_EXA_SPACE=DebugCuda", file=sys.stderr)

    return RunConfig(
        cwd=cwd,
        prop_path=prop_path,
        prop_abs=str(prop_path),
        class_name=class_name,
        method_name=method_name,
        line_pre=line_pre,
        line_inspect=line_inspect,
        hotswap_line=hotswap_line,
        pdb_probe=pdb_probe,
        probe_label=probe_label,
        hotswap_source=hotswap_source,
        hotswap_target=hotswap_target,
        edit_full_token=edit_full_token,
        edit_half_token=edit_half_token,
        exa_args=exa_args,
        env=env,
    )


# --------------------------------------------------------------------------------------
# Run one benchmark configuration (one preset, or one custom target).
# --------------------------------------------------------------------------------------


def _select_scenarios(args: argparse.Namespace) -> tuple[bool, bool]:
    """
    want_hotswap, want_edit_rerun) from --only / --hotswap-only / --edit-rerun-only (default: both).
    """
    if args.hotswap_only and args.edit_rerun_only:
        raise SystemExit("Use only one of --hotswap-only / --edit-rerun-only")
    if args.only_scenarios and (args.hotswap_only or args.edit_rerun_only):
        raise SystemExit(
            "Do not combine --only with --hotswap-only / --edit-rerun-only"
        )
    if args.only_scenarios:
        return "hotswap" in args.only_scenarios, "edit-rerun" in args.only_scenarios
    if args.hotswap_only:
        return True, False
    if args.edit_rerun_only:
        return False, True
    return True, True


def _run_scenarios(
    cfg: RunConfig,
    *,
    line_pre: int,
    line_inspect: int,
    hotswap_line: int,
    want_h: bool,
    want_e: bool,
    logfile,
    heartbeat_interval: float | None,
    verify: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    h_out = e_out = None
    if want_h:
        h_out = run_scenario_hotswap(
            cwd=cfg.cwd,
            prop_abs=cfg.prop_abs,
            line_pre=line_pre,
            line_inspect=line_inspect,
            hotswap_line=hotswap_line,
            hotswap_source=cfg.hotswap_source,
            hotswap_target=cfg.hotswap_target,
            pdb_probe=cfg.pdb_probe,
            probe_label=cfg.probe_label,
            exa_args=cfg.exa_args,
            env=cfg.env,
            logfile=logfile,
            heartbeat_interval=heartbeat_interval,
            verify=verify,
        )
        _print_rows("=== Scenario B: hotswap ===", h_out)
    if want_e:
        e_out = run_scenario_edit_rerun(
            cwd=cfg.cwd,
            prop_path=cfg.prop_path,
            prop_abs=cfg.prop_abs,
            line_inspect=line_inspect,
            pdb_probe=cfg.pdb_probe,
            probe_label=cfg.probe_label,
            edit_full_token=cfg.edit_full_token,
            edit_half_token=cfg.edit_half_token,
            exa_args=cfg.exa_args,
            env=cfg.env,
            logfile=logfile,
            heartbeat_interval=heartbeat_interval,
            verify=verify,
        )
        _print_rows("=== Scenario A: edit + rerun ===", e_out)
    return h_out, e_out


def _print_headline(
    h_out: dict[str, Any] | None, e_out: dict[str, Any] | None
) -> dict[str, float]:
    """
    Print + return the headline timing comparison for whichever scenarios ran.
    """
    headline: dict[str, float] = {}
    if h_out and e_out:
        s2 = _num(e_out["sum_two_runs_s"], "sum_two_runs_s")
        th = _num(h_out["total_r_to_factor_s"], "total_r_to_factor_s")
        hc = _num(h_out["hotswap_cmd_s"], "hotswap_cmd_s")
        print("--- Headline (after `r`, to inspect / p line) ---")
        print(f"  (1) no hotswap: sum_two_runs_s       = {s2:.6f} s")
        print(f"  (2) hotswap:    total_r_to_factor_s  = {th:.6f} s")
        print(f"      hotswap_cmd_s = {hc:.6f} s\n")
        headline["without_hotswap_sum_two_runs_s"] = s2
        headline["with_hotswap_total_r_to_factor_s"] = th
        headline["hotswap_command_s"] = hc
    elif h_out:
        th = _num(h_out["total_r_to_factor_s"], "total_r_to_factor_s")
        print(f"--- total_r_to_factor_s = {th:.6f} s\n")
        headline["with_hotswap_total_r_to_factor_s"] = th
    elif e_out:
        s2 = _num(e_out["sum_two_runs_s"], "sum_two_runs_s")
        print(f"--- sum_two_runs_s = {s2:.6f} s\n")
        headline["without_hotswap_sum_two_runs_s"] = s2
    return headline


def _run_benchmark_once(
    args: argparse.Namespace,
    *,
    effective_preset: str | None,
) -> tuple[bool, dict[str, Any]]:
    """
    Run one benchmark configuration. `effective_preset` is a key in `_BENCH_PRESETS` or
    `None` for a custom target.
    """
    cfg = _resolve_run_config(args, effective_preset)
    line_pre, line_inspect, hotswap_line = (
        cfg.line_pre,
        cfg.line_inspect,
        cfg.hotswap_line,
    )
    print(
        f"[bench] argv {cfg.exa_args} | {cfg.class_name}.{cfg.method_name} | "
        f"b@{line_pre}, b@{line_inspect} (inspect) | "
        f"hotswap@{hotswap_line} ({cfg.hotswap_source}->{cfg.hotswap_target}) | "
        f"p {cfg.pdb_probe!r}",
        file=sys.stderr,
        flush=True,
    )

    want_h, want_e = _select_scenarios(args)
    heartbeat_interval = (
        None
        if args.no_heartbeat or args.heartbeat_interval <= 0
        else float(args.heartbeat_interval)
    )
    logfile = sys.stderr if args.verbose else None
    verify = not args.no_verify

    h_out, e_out = _run_scenarios(
        cfg,
        line_pre=line_pre,
        line_inspect=line_inspect,
        hotswap_line=hotswap_line,
        want_h=want_h,
        want_e=want_e,
        logfile=logfile,
        heartbeat_interval=heartbeat_interval,
        verify=verify,
    )

    headline = _print_headline(h_out, e_out)
    vok = _verify_print(h_out, e_out, verify=verify, probe_label=cfg.probe_label)

    payload = {
        "headline": headline,
        "run": {
            "hotswap": want_h,
            "edit_rerun": want_e,
        },
        "breakpoints": {
            "property_file": cfg.prop_abs,
            "class_name": cfg.class_name,
            "method_name": cfg.method_name,
            "line_pre": line_pre,
            "line_inspect": line_inspect,
            "hotswap_line": hotswap_line,
            "pdb_probe": cfg.pdb_probe,
            "probe_label": cfg.probe_label,
            "min_lines_before_parallel": _MIN_LINES_BEFORE_PARALLEL,
        },
        "preset": effective_preset,
        "scenario_B_hotswap": h_out,
        "scenario_A_edit_rerun": e_out,
        "verify_ok": vok,
    }
    return vok, payload


# --------------------------------------------------------------------------------------
# CLI: argument parsing, preset sweep, results JSON.
# --------------------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    default_cwd = Path(__file__).resolve().parent / "src"
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", type=Path, default=default_cwd)
    ap.add_argument("--property-file", type=Path, default=None)
    ap.add_argument("--preset", choices=tuple(_BENCH_PRESETS.keys()), default=None)
    ap.add_argument("--method-name", type=str, default=None, metavar="NAME")
    ap.add_argument("--pdb-probe", type=str, default=None, metavar="EXPR")
    ap.add_argument("--probe-label", type=str, default=None, metavar="NAME")
    ap.add_argument("--class-name", type=str, default=None, metavar="NAME")
    ap.add_argument("--line-pre", type=int, default=None, metavar="LINE")
    ap.add_argument("--line-inspect", type=int, default=None, metavar="LINE")
    ap.add_argument("--hotswap-line", type=int, default=None, metavar="LINE")
    ap.add_argument("--hotswap-source", type=str, default=None)
    ap.add_argument("--hotswap-target", type=str, default=None)
    ap.add_argument("--edit-full-token", type=str, default=None)
    ap.add_argument("--edit-half-token", type=str, default=None)
    ap.add_argument("--exa-arg", action="append", default=[])
    ap.add_argument("--atoms", type=int, default=1000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--heartbeat-interval", type=float, default=15.0, metavar="SEC")
    ap.add_argument("--no-heartbeat", action="store_true")
    ap.add_argument(
        "--only",
        action="append",
        dest="only_scenarios",
        choices=("hotswap", "edit-rerun"),
        metavar="WHICH",
    )
    ap.add_argument("--hotswap-only", action="store_true")
    ap.add_argument("--edit-rerun-only", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--strict-verify", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None, metavar="PATH")
    ap.add_argument("--no-json-out", action="store_true")
    return ap


def _explicit_single_target(args: argparse.Namespace) -> bool:
    """
    True if the user aimed at one custom configuration (skip the default all-presets sweep).
    """
    return (
        args.property_file is not None
        or args.class_name is not None
        or args.method_name is not None
        or args.pdb_probe is not None
        or args.probe_label is not None
        or args.line_pre is not None
        or args.line_inspect is not None
        or args.hotswap_line is not None
        or args.hotswap_source is not None
        or args.hotswap_target is not None
        or args.edit_full_token is not None
        or args.edit_half_token is not None
    )


def _determine_preset_sequence(args: argparse.Namespace) -> list[str | None]:
    if args.preset is not None:
        return [args.preset]
    if _explicit_single_target(args):
        return [None]
    return list(_BENCH_PRESETS.keys())


def _determine_json_path(args: argparse.Namespace, system_info: dict) -> Path | None:
    if args.no_json_out:
        return None
    if args.json_out is not None:
        return args.json_out.expanduser().resolve()
    return _hotswap_results_json_path(system_info)


def _run_preset_sequence(
    args: argparse.Namespace, preset_sequence: list[str | None]
) -> tuple[bool, dict[str, Any], dict[str, Any] | None]:
    """Run each preset in `preset_sequence`. Returns (all_verify_ok, payload_by_label, last_payload)."""
    all_verify_ok = True
    multi_payload: dict[str, Any] = {}
    last_payload: dict[str, Any] | None = None
    for eff in preset_sequence:
        label = eff if eff is not None else "custom"
        if len(preset_sequence) > 1:
            print(
                f"\n{'=' * 72}\n[bench] --- preset: {label} ---\n{'=' * 72}\n",
                file=sys.stderr,
                flush=True,
            )
        try:
            vok, payload = _run_benchmark_once(args, effective_preset=eff)
        except FileNotFoundError as e:
            raise SystemExit(str(e)) from e
        all_verify_ok = all_verify_ok and vok
        last_payload = payload
        if len(preset_sequence) > 1:
            multi_payload[label] = payload
    return all_verify_ok, multi_payload, last_payload


def _build_output_doc(
    *,
    preset_sequence: list[str | None],
    last_payload: dict[str, Any] | None,
    multi_payload: dict[str, Any],
    all_verify_ok: bool,
    system_info: dict,
    json_suffix: str,
) -> dict[str, Any]:
    legend = {
        "_scenario_legend": _JSON_SCENARIO_LEGEND,
        "system": system_info,
        "system_suffix": json_suffix,
    }
    if len(preset_sequence) == 1:
        if last_payload is None:
            raise RuntimeError("bench: no payload for json-out")
        return {**legend, **last_payload}
    order = [x if x is not None else "custom" for x in preset_sequence]
    return {
        **legend,
        "preset_order": order,
        "presets": multi_payload,
        "verify_all_ok": all_verify_ok,
    }


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.no_json_out and args.json_out is not None:
        sys.exit("Use only one of --json-out PATH or --no-json-out")

    system_info = collect_system_info()
    json_path = _determine_json_path(args, system_info)
    preset_sequence = _determine_preset_sequence(args)

    all_verify_ok, multi_payload, last_payload = _run_preset_sequence(
        args, preset_sequence
    )

    if json_path is not None:
        doc = _build_output_doc(
            preset_sequence=preset_sequence,
            last_payload=last_payload,
            multi_payload=multi_payload,
            all_verify_ok=all_verify_ok,
            system_info=system_info,
            json_suffix=system_suffix(system_info),
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"[bench] wrote {json_path}")

    if args.strict_verify and not args.no_verify and not all_verify_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
