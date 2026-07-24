import cupy as cp
import pykokkos as pk


@pk.workunit
def work(wid, a):
    a[wid] += 1


def main():
    N = 64
    pk.set_default_space(pk.ExecutionSpace.Cuda)
    a = cp.zeros(N, int)
    b = cp.zeros(N, int)
    for i in range(0, N):
        a[i] = i * 10

    print(a)
    print(type(a))

    pk.parallel_for("work", pk.RangePolicy(0, N), work2, a=a)
    print(a)


if __name__ == "__main__":
    main()
