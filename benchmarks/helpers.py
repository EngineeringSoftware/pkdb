"""Shared helpers for the benchmark runner scripts (run_examinimd.py, run_boltzmann.py, run_ewald.py)."""

import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# --- system info -----------------------------------------------------------


def _read_first_existing_text(paths: list[Path]) -> str | None:
    for p in paths:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return None


def _linux_distro_pretty_name() -> str | None:
    txt = _read_first_existing_text([Path("/etc/os-release"), Path("/usr/lib/os-release")])
    if not txt:
        return None
    m = re.search(r'^\s*PRETTY_NAME\s*=\s*("?)(.*?)\1\s*$', txt, flags=re.MULTILINE)
    if m:
        return m.group(2).strip() or None
    return None


def _cpu_info() -> dict:
    """CPU summary from /proc/cpuinfo when available: model name (best-effort) and logical core count."""
    info: dict[str, object] = {
        "name": None,
        "logical_cores": os.cpu_count(),
    }

    try:
        txt = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return info

    model = None
    logical = 0

    # /proc/cpuinfo has one "processor" line per logical CPU
    for line in txt.splitlines():
        if ":" not in line:
            continue
        k, v = [s.strip() for s in line.split(":", 1)]
        kl = k.lower()
        if kl == "processor":
            logical += 1
        if model is None and kl in ("model name", "processor", "cpu", "hardware") and v:
            model = v

    if model:
        info["name"] = model
    if logical:
        info["logical_cores"] = logical
    return info


def _ram_bytes() -> int | None:
    try:
        txt = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"^MemTotal:\s+(\d+)\s+kB\s*$", txt, flags=re.MULTILINE)
    if not m:
        return None
    return int(m.group(1)) * 1024


def _nvidia_gpus() -> list[dict]:
    """Query NVIDIA GPUs via nvidia-smi if present. Returns list of {model, memory_total_bytes}."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    if r.returncode != 0:
        return []

    gpus: list[dict] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        name = parts[0] if parts[0] else None
        mem_mib = None
        if len(parts) >= 2:
            try:
                mem_mib = int(float(parts[1]))
            except ValueError:
                mem_mib = None
        gpus.append({"model": name, "memory_total_bytes": mem_mib * 1024 * 1024 if mem_mib is not None else None})
    return gpus


def _amd_model_from_rocm_json(info: dict) -> str:
    """Best-effort display name from rocm-smi --showproductname JSON for one card."""

    def _usable_label(s: str) -> bool:
        s = s.strip()
        if not s or s.upper() == "N/A":
            return False
        # PCI device id is not a marketing name; prefer GFX / SKU below.
        if re.fullmatch(r"0x[0-9a-fA-F]+", s):
            return False
        return True

    for key in ("Device Name", "Card Series", "Card Model"):
        v = info.get(key)
        if isinstance(v, str) and _usable_label(v):
            return v.strip()
    gfx = info.get("GFX Version")
    sku = info.get("Card SKU")
    pci = info.get("Card Model") or info.get("Device ID")
    pci_s = pci.strip() if isinstance(pci, str) else None
    if isinstance(gfx, str) and gfx.strip():
        g = gfx.strip()
        tail_parts: list[str] = []
        if isinstance(sku, str) and sku.strip() and sku.strip().upper() != "N/A":
            tail_parts.append(sku.strip())
        if pci_s and pci_s.upper() != "N/A" and not _usable_label(pci_s):
            tail_parts.append(pci_s)
        if tail_parts:
            return f"AMD {g} ({', '.join(tail_parts)})"
        return f"AMD {g}"
    if pci_s and pci_s.upper() != "N/A":
        return f"AMD GPU ({pci_s})"
    return "AMD GPU"


def _parse_rocm_smi_id_device_names(blob: str) -> dict[int, str]:
    """Parse `rocm-smi -i` lines like GPU[0] : Device Name: AMD Instinct MI300X."""
    names: dict[int, str] = {}
    for line in blob.splitlines():
        m = re.match(r"GPU\[(\d+)\]\s*:\s*Device Name:\s*(.+)$", line.strip())
        if not m:
            continue
        name = m.group(2).strip()
        if name.upper() != "N/A":
            names[int(m.group(1))] = name
    return names


def _amd_gpus() -> list[dict]:
    """Query AMD GPUs via rocm-smi (ROCm) if present. Returns list of {model, memory_total_bytes} in card order."""
    try:
        r_mem = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    blob = (r_mem.stdout or "") + (r_mem.stderr or "")
    lines = blob.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("device,"):
            header_idx = i
            break
    if header_idx is None:
        return []

    mem_by_card: dict[str, int] = {}
    for line in lines[header_idx + 1 :]:
        line = line.strip()
        if not line:
            break
        row = next(csv.reader([line]))
        if len(row) < 2:
            continue
        card = row[0].strip()
        try:
            mem_by_card[card] = int(row[1])
        except ValueError:
            continue

    if not mem_by_card:
        return []

    product: dict[str, object] = {}
    try:
        r_pr = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        pb = (r_pr.stdout or "") + (r_pr.stderr or "")
        for ln in pb.splitlines():
            s = ln.strip()
            if s.startswith("{"):
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    break
                if isinstance(parsed, dict):
                    product = parsed
                break
    except (OSError, subprocess.TimeoutExpired):
        pass

    names_by_index: dict[int, str] = {}
    try:
        r_id = subprocess.run(["rocm-smi", "-i"], capture_output=True, text=True, check=False, timeout=10)
        ib = (r_id.stdout or "") + (r_id.stderr or "")
        names_by_index = _parse_rocm_smi_id_device_names(ib)
    except (OSError, subprocess.TimeoutExpired):
        pass

    def card_sort_key(card: str) -> tuple[int, str]:
        if card.startswith("card") and card[4:].isdigit():
            return (0, f"{int(card[4:]):06d}")
        return (1, card)

    gpus: list[dict] = []
    for card in sorted(mem_by_card.keys(), key=card_sort_key):
        gpu_idx = int(card[4:]) if card.startswith("card") and card[4:].isdigit() else 0
        model = names_by_index.get(gpu_idx)
        if not model:
            info = product.get(card)
            model = _amd_model_from_rocm_json(info) if isinstance(info, dict) else "AMD GPU"
        gpus.append({"model": model, "memory_total_bytes": mem_by_card[card]})
    return gpus


def collect_system_info() -> dict:
    return {
        "os": {
            "platform": platform.system().lower() or None,
            "distro": _linux_distro_pretty_name(),
            "kernel": platform.release() or None,
        },
        "processor": _cpu_info(),
        "ram": {"bytes": _ram_bytes()},
        "gpus": _nvidia_gpus() or _amd_gpus(),
    }


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def system_suffix(system_info: dict) -> str:
    mach = (platform.machine() or "").lower()
    if mach == "aarch64":
        proc_tag = "arm64"
    elif mach in ("x86_64", "amd64"):
        proc_tag = "x86_64"
    else:
        proc_tag = _slug(mach) if mach else "unknowncpu"

    gpu_tag = "nogpu"
    gpus = system_info.get("gpus") if isinstance(system_info, dict) else None
    if isinstance(gpus, list) and gpus:
        model = gpus[0].get("model") if isinstance(gpus[0], dict) else None
        if isinstance(model, str) and model.strip():
            low = model.lower()
            m = re.search(r"\b(a\d{2,3})\b", low)
            if m:
                gpu_tag = m.group(1)
            else:
                m_mi = re.search(r"\b(mi\d+[a-z]*)\b", low)
                gpu_tag = m_mi.group(1) if m_mi else _slug(model.replace("nvidia", ""))

    return f"{proc_tag}_{gpu_tag}"


def system_and_suffix() -> tuple[dict, str]:
    """collect_system_info() plus its filename suffix (callers need both together)."""
    info = collect_system_info()
    return info, system_suffix(info)


# --- result JSON -------------------------------------------------------------


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not read {path}: {e}; treating as new file.", file=sys.stderr)
        return None
    return data if isinstance(data, dict) else None


def _combine_backend_rows(existing_rows: list[Any], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    seen: set[str] = set()

    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        name = row.get("backend")
        if not name:
            continue
        key = str(name)
        combined[key] = dict(row)
        order.append(key)
        seen.add(key)

    for row in new_rows:
        name = row.get("backend")
        if not name:
            continue
        key = str(name)
        combined[key] = dict(row)
        if key not in seen:
            order.append(key)
            seen.add(key)

    return [combined[k] for k in order]


def _combine_sizes_by_key(existing_sizes: Any, new_sizes: list[dict[str, Any]], x_key: str) -> list[dict[str, Any]]:
    by_x: dict[int, dict[str, Any]] = {}

    if isinstance(existing_sizes, list):
        for item in existing_sizes:
            if not isinstance(item, dict):
                continue
            try:
                xi = int(item[x_key])
            except (KeyError, TypeError, ValueError):
                continue
            backends = item.get("backends")
            if not isinstance(backends, list):
                backends = []
            by_x[xi] = {x_key: xi, "backends": [dict(b) for b in backends if isinstance(b, dict)]}

    for item in new_sizes:
        try:
            xi = int(item[x_key])
        except (KeyError, TypeError, ValueError):
            continue
        new_backends = item.get("backends", [])
        if not isinstance(new_backends, list):
            new_backends = []
        if xi not in by_x:
            by_x[xi] = {x_key: xi, "backends": []}
        by_x[xi]["backends"] = _combine_backend_rows(
            by_x[xi]["backends"], [dict(b) for b in new_backends if isinstance(b, dict)]
        )

    return sorted(by_x.values(), key=lambda d: d[x_key])


def write_results(
    path: Path,
    new_sizes: list[dict[str, Any]],
    *,
    x_key: str,
    test: str,
    output_file: str,
    system: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return the full timing/results JSON object to write to ``path``.

    If ``path`` already exists, prior ``sizes`` and ``system`` are kept where applicable; new
    rows update matching ``x_key`` / ``backend`` entries. ``sizes`` are sorted by ``x_key``.
    No top-level ``runs`` (per-backend ``runs`` only).
    """
    existing = _read_json_object(path)
    doc: dict[str, Any] = dict(existing) if existing else {}
    doc.pop("runs", None)
    doc["test"] = test
    doc["output_file"] = output_file
    if extra_fields:
        doc.update(extra_fields)
    if not doc.get("system"):
        doc["system"] = dict(system)
    doc["sizes"] = _combine_sizes_by_key(doc.get("sizes"), new_sizes, x_key)
    return doc


def write_checkpoint(
    path: Path,
    sizes: list[dict[str, Any]],
    *,
    x_key: str,
    test: str,
    system: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """write_results() for ``sizes`` at ``path``, then serialize the result back to ``path``."""
    doc = write_results(
        path, sizes, x_key=x_key, test=test, output_file=str(path), system=system, extra_fields=extra_fields
    )
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


# --- runner CLI / plan helpers -----------------------------------------------


def results_path(path: Path, prefix: str, ext: str) -> Path:
    """Relative paths land in results/; names get the runner prefix and an extension."""
    base = path if path.is_absolute() else RESULTS_DIR / path
    name = base.name if base.name.startswith(prefix) else f"{prefix}_{base.name}"
    if not Path(name).suffix:
        name += ext
    return base.with_name(name)


def resolve_output_paths(args: argparse.Namespace, prefix: str, suffix: str) -> tuple[Path, Path]:
    """Resolve the timing-results and log paths from CLI args; parent dirs are created.

    Uses args.output and args.log_file (either may be None); the defaults are
    <prefix>_<suffix>.json and <prefix>_out.log under results/.
    """
    output = results_path(args.output or Path(f"{prefix}_{suffix}"), prefix, ".json")
    log_path = results_path(args.log_file, prefix, ".log") if args.log_file else output.parent / f"{prefix}_out.log"
    for path in (output, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    return output, log_path


def parse_spaces(arg: str) -> list[str]:
    """Comma-separated space names -> unique names in order."""
    return list(dict.fromkeys(s.strip() for s in arg.split(",") if s.strip()))


def timing_plan(spaces: list[str]) -> list[tuple[str, str | None]]:
    """Group spaces into rows of (space, its Debug partner or None).

    ["Cuda", "DebugCuda", "Serial"] -> [("Cuda", "DebugCuda"), ("Serial", None)]
    """
    rows: list[tuple[str, str | None]] = []
    for space in spaces:
        if space.startswith("Debug") and space.removeprefix("Debug") in spaces:
            continue  # appears as the partner of its normal space
        debug = f"Debug{space}"
        rows.append((space, debug if debug in spaces else None))
    return rows


def override_flag(script_args: list[str], flag: str, value: str) -> list[str]:
    """script_args with any existing "<flag> V" replaced by "<flag> <value>"."""
    kept: list[str] = []
    args_iter = iter(script_args)
    for arg in args_iter:
        if arg == flag:
            next(args_iter, None)  # drop its value as well
        else:
            kept.append(arg)
    return [*kept, flag, value]


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


class Runner:
    """
    Runs a benchmark script as a fresh subprocess per (space, script_args) call.
    """

    def __init__(self, timeout: int | None = None, space_flag: str | None = None):
        self.timeout = timeout
        self.space_flag = space_flag

    def run_once(self, main_py: Path, space: str, script_args: list[str]) -> tuple[int, str, str]:
        env = os.environ.copy()
        env["PK_EXEC_SPACE"] = env["PK_EXA_SPACE"] = space
        env.pop("PK_KERNEL_PROFILE_PATH", None)  # a stray env var would make runs write profile JSONs

        argv = [sys.executable, str(main_py)]
        if self.space_flag:
            argv += [self.space_flag, space]
        argv += script_args

        proc = subprocess.run(
            argv, cwd=main_py.parent, env=env, capture_output=True, text=True, timeout=self.timeout
        )
        return proc.returncode, proc.stdout, proc.stderr

    def warmup(self, main_py: Path, script_args: list[str], spaces: list[str]) -> None:
        print("Warmup (compile selected spaces):")
        for space in spaces:
            print(f"  {main_py.name} @ {space} ...", end=" ", flush=True)
            try:
                code, out, err = self.run_once(main_py, space, script_args)
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                continue
            print("OK" if code == 0 else "FAIL")
            if code != 0:
                print((err or out).strip()[-500:])
