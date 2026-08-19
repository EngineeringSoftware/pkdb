"""
IPC token extraction for on-device function evaluation.

Uses CuPy IPC API: cupy.cuda.runtime.ipcGetMemHandle(arr.data.ptr)
or the generated JSON file (pkdb_ipc.json under the pkdb project's .internal/ipc
directory) written when the debugger intercepts each ``parallel_*`` dispatch.
Returns: handle_hex, num_elements, dtype_name, shape.
"""

import math
import os
from pathlib import Path
from typing import Any, Optional

# IPC token: handle_hex, num_elements, dtype_name, shape
IpcToken = dict

# JSON key for workunit-parameter -> caller-local-name (written at IPC collection time).
PKDB_ROOT = Path(__file__).resolve().parent.parent
INTERNAL_DIR = PKDB_ROOT / ".internal"
IPC_JSON_PATH = INTERNAL_DIR / "pkdb_ipc.json"
PKDB_WORKUNIT_PARAM_MAP_KEY = "_pkdb_workunit_param_to_caller"


def load_ipc_json() -> dict:
    """Load the generated IPC JSON (var_name -> {ipc_handle, devptr, shape, dtype})."""
    if not os.path.exists(IPC_JSON_PATH):
        return {}
    try:
        import json

        with open(IPC_JSON_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _ipc_json_entry_to_token(entry: dict) -> Optional[IpcToken]:
    if not entry or "ipc_handle" not in entry:
        return None
    shape = entry.get("shape", [])
    if isinstance(shape, tuple):
        shape = list(shape)
    num_elements = int(math.prod(shape)) if shape else 0
    return {
        "handle_hex": entry["ipc_handle"],
        "num_elements": num_elements,
        "dtype_name": entry.get("dtype", ""),
        "shape": shape,
    }


def _resolve_ipc_json_entry(data: dict, var_name: str) -> Optional[dict]:
    """
    Locate the CuPy IPC entry for a name used in eval (e.g. Kokkos view param ``y_view``).

    1. Direct key in JSON (caller local name, e.g. ``y``).
    2. Else, if ``var_name`` is a workunit parameter, follow
       ``PKDB_WORKUNIT_PARAM_MAP_KEY`` to the caller binding (AST-derived at the launch line).
    """
    if var_name == PKDB_WORKUNIT_PARAM_MAP_KEY:
        return None
    entry = data.get(var_name)
    if isinstance(entry, dict) and "ipc_handle" in entry:
        return entry
    pmap = data.get(PKDB_WORKUNIT_PARAM_MAP_KEY)
    if isinstance(pmap, dict) and var_name in pmap:
        caller = pmap[var_name]
        if isinstance(caller, str):
            cand = data.get(caller)
            if isinstance(cand, dict) and "ipc_handle" in cand:
                return cand
    return None


def get_ipc_from_json(var_name: str) -> Optional[IpcToken]:
    """
    Extract IPC token for a variable name from the generated JSON file.
    Returns same shape as extract_ipc_from_arg: handle_hex, num_elements, dtype_name, shape.
    """
    data = load_ipc_json()
    entry = _resolve_ipc_json_entry(data, var_name)
    if not entry:
        return None
    return _ipc_json_entry_to_token(entry)


def extract_ipc_from_arg(arg: Any, context) -> Optional[IpcToken]:
    """
    Extract IPC token from an array argument.

    Uses context to resolve variables when needed (e.g. KokkosViewProxy -> CuPy array).
    If no live array is available, falls back to the generated JSON file (by variable name).
    """
    if arg is None or isinstance(arg, (int, float, str, bool)):
        return None

    var_name = getattr(arg, "var_name", None)

    # Resolve via context when arg is a proxy (e.g. KokkosViewProxy with var_name)
    if not _has_data_ptr(arg) and context:
        arg = _resolve_from_context(arg, context)
        if arg is None and var_name:
            # No live array; try generated JSON from the last parallel_* IPC collection
            return get_ipc_from_json(var_name)
        if arg is None:
            return None

    if not _has_data_ptr(arg):
        return None

    import cupy as cp

    ptr = arg.data.ptr
    handle = cp.cuda.runtime.ipcGetMemHandle(ptr)
    return {
        "handle_hex": handle.hex(),
        "num_elements": int(arg.size),
        "dtype_name": str(arg.dtype),
        "shape": list(arg.shape),
    }


def _resolve_from_context(arg: Any, context) -> Any:
    """Resolve KokkosViewProxy to CuPy array using context."""
    # KokkosViewProxy has var_name and context attributes
    var_name = getattr(arg, "var_name", None)
    ctx = getattr(arg, "context", None) or context
    if not var_name:
        return None
    # Use proxy's context if available (same EvaluationContext)
    if ctx.frame:
        return eval(var_name, ctx.frame.f_globals, ctx.frame.f_locals)
    if var_name in ctx.script_globals:
        return ctx.script_globals[var_name]
    return None


def _has_data_ptr(arr: Any) -> bool:
    """True if array has .data.ptr (CuPy array)."""
    try:
        return hasattr(arr, "data") and hasattr(arr.data, "ptr")
    except Exception:
        return False
