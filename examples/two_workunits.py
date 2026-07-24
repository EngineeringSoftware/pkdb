import cupy as np

import pykokkos as pk


@pk.workunit
def work1(wid, a):
    a[wid] += 1
    a[wid] *= 10


@pk.workunit
def work2(wid, a, b, c):
    a[wid] += b[wid]
    a[wid] *= c


def main():
    N = 10
    a = np.random.randint(100, size=(N))
    print(a)

    pk.set_default_space(pk.ExecutionSpace.DebugCuda)
    pk.parallel_for("work", pk.RangePolicy(0, N), work1, a=a)
    print(f"Intermediate result: {a}")

    b = np.random.randint(100, size=(N))
    pk.parallel_for("work", pk.RangePolicy(0, N), work2, a=a, b=b, c=10)
    print(f"Final result: {a}")


main()
