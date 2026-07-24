#!/usr/bin/env python3
"""
Compute sum of a CuPy array accessed via IPC handle using a PyKokkos kernel.
Used by pkdb for "eval sum a" when a is a CuPy array. Reads a JSON file with
handle_hex, num_elements, dtype; opens IPC, binds to CuPy array, runs
parallel_reduce (sum_reduce workunit), prints "SUM=<value>" to stdout.
Only 1D int arrays supported. No CLI flags; single argument: path to JSON.
"""

import json
import sys

import pykokkos as pk


@pk.workunit
def sum_reduce(i, acc, a):
    acc += a[i]


_DTYPE_MAP = None


def _get_dtype_map():
    import numpy as np

    global _DTYPE_MAP
    if _DTYPE_MAP is None:
        _DTYPE_MAP = {
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "int64": np.int64,
            "int8": np.int8,
            "int16": np.int16,
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "uint64": np.uint64,
        }
    return _DTYPE_MAP


def sum_array_from_handle(handle_hex: str, num_elements: int, dtype_name: str = "int64") -> float:
    """
    Get sum of a CuPy array accessed via IPC handle, using a PyKokkos kernel.
    1) Open array from IPC handle
    2) Wrap in PyKokkos view and run sum_reduce kernel
    3) Return the sum. Caller is in the same process (no subprocess).
    """
    import numpy as np
    import cupy as cp

    dtype_map = _get_dtype_map()
    dtype = dtype_map.get(dtype_name, np.int64)
    n = int(num_elements)
    if n <= 0:
        raise ValueError("num_elements must be > 0")
    handle_bytes = bytes.fromhex(handle_hex)
    if len(handle_bytes) != 64:
        raise ValueError("IPC handle must be 64 bytes (128 hex chars)")

    cp.cuda.Device(0).use()
    pk.set_default_space(pk.Cuda)
    runtime = cp.cuda.runtime
    ptr = runtime.ipcOpenMemHandle(handle_bytes)
    try:
        itemsize = np.dtype(dtype).itemsize
        total_bytes = n * itemsize
        mem = cp.cuda.UnownedMemory(int(ptr), total_bytes, owner=None)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        arr = cp.ndarray((n,), dtype=dtype, memptr=memptr)
        total = pk.parallel_reduce("reduce", pk.RangePolicy(pk.Cuda, 0, n), sum_reduce, a=arr)
        return float(total)
    finally:
        runtime.ipcCloseMemHandle(ptr)


def run_sum(json_path: str) -> float:
    """
    Read handle info from JSON file, then open IPC, run PyKokkos sum kernel, return sum.
    """
    with open(json_path) as f:
        data = json.load(f)
    handle_hex = data.get("handle_hex")
    num_elements = data.get("num_elements")
    dtype_name = data.get("dtype", "int64")
    if not handle_hex or num_elements is None:
        raise ValueError("JSON must contain handle_hex and num_elements")
    return sum_array_from_handle(handle_hex, int(num_elements), dtype_name)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m pkdb.scripts.cuda_device_sum <path_to.json>", file=sys.stderr)
        sys.exit(1)
    json_path = sys.argv[1]
    try:
        total = run_sum(json_path)
        print(f"SUM={total}")
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
