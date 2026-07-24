import pykokkos as pk
from test_import1 import kernel_i_plus_100
import cupy as cp


if __name__ == "__main__":
    N = 64
    pk.set_default_space(pk.ExecutionSpace.DebugCuda)
    a = cp.zeros(N, int)
    print(a)
    pk.parallel_for("kernel_i_plus_100", pk.RangePolicy(0, N), kernel_i_plus_100, a=a)
    print(a)
