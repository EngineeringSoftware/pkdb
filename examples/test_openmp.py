import numpy as np
import pykokkos as pk


@pk.workunit
def work(wid, a):
    a[wid] = wid
    a[wid] = a[wid] + 100 * wid


@pk.workunit
def work2(wid, a):
    a[wid] = wid


def main():
    N = 10
    pk.set_default_space(pk.ExecutionSpace.DebugOpenMP)
    a = np.zeros(N, int)

    pk.parallel_for("work", pk.RangePolicy(0, N), work, a=a)
    pk.parallel_for("work2", pk.RangePolicy(0, N), work2, a=a)
    print(f"Result: {a}")


main()
