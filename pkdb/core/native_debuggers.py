"""
Which native debugger binary pkdb drives, per debugger kind. Configured during
runtime. The binaries are pinned once, at install time, into the generated
`core/debugger_paths.py`. For example:

`CUDA_GDB=/usr/local/cuda-13/bin/cuda-gdb pip install -e .` 

Sets `cuda-gdb` to `cuda-13` version
"""

try:
    from .debugger_paths import CUDA_GDB, GDB, HIP_GDB
except ImportError:  # source tree that was never pip installed
    GDB, CUDA_GDB, HIP_GDB = "gdb", "cuda-gdb", "rocgdb"

# Keys are the debugger kinds. "cuda"/"hip" match
# AcceleratorGDBController.accelerator_type, so call sites pass it straight through.
_EXECUTABLES = {"gdb": GDB, "cuda": CUDA_GDB, "hip": HIP_GDB}


def executable_for(kind: str) -> str:
    """The binary to exec for this debugger kind: "gdb", "cuda" or "hip"."""
    return _EXECUTABLES[kind]
