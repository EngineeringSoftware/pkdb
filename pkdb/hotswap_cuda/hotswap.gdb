# hotswap.gdb - loader for Python-based CUDA kernel hot-swap

python
import sys, os, gdb

try:
    base = os.path.join(os.getcwd(), "hotswap")
    if base not in sys.path:
        sys.path.insert(0, base)
except Exception:
    pass

try:
    import hotswap  # registers hotswap, hotswap-escape, update-kernels
    gdb.write("Hot-swap loaded. Commands: hotswap, hotswap-escape, update-kernels\n")
except Exception as e:
    gdb.write("Error loading hotswap.py: %s\n" % e)
end

