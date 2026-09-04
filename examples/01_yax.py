import cupy as cp
import pykokkos as pk

@pk.workunit
def yAx(j, acc, cols, y_view, x_view, A_view):
    temp2: float = 0
    for i in range(cols):
        temp2 += A_view[j * cols + i] * x_view[i]
    acc += y_view[j] * temp2


def run() -> None:
    N: int = 256   # Rows
    M: int = 1024  # Cols
    y = cp.random.rand(N)
    x = cp.random.rand(M)
    A = cp.random.rand(N * M)

    policy = pk.RangePolicy(pk.ExecutionSpace.Cuda, 0, N)
    result = pk.parallel_reduce(policy, yAx, cols=M, y_view=y, x_view=x, A_view=A)
    print(result)


if __name__ == "__main__":
    run()
