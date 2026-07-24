import pykokkos as pk
import cupy as cp


@pk.workunit
def yAx(j, out, m, y, x, A):
    acc: float = 0
    for i in range(m):
        acc += A[j][i] * x[i]
    out += y[j] * acc


def main():
    pk.set_default_space(pk.ExecutionSpace.DebugCuda)

    m, n = 100, 100
    y = cp.random.randn(n)
    x = cp.random.randn(m)
    A = cp.random.randn(n, m)
    p = pk.RangePolicy(0, n)
    result = pk.parallel_reduce(p, yAx, m=m, y=y, x=x, A=A)
    print(result)


main()
