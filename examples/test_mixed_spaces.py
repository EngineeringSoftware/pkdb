import sys

import cupy as cp
import numpy as np
import pykokkos as pk


@pk.workunit
def on_host(wid, a):
    a[wid] = wid
    a[wid] += 100


@pk.workunit
def on_device(wid, b):
    b[wid] = wid
    b[wid] += 7


def main():
    N = 10
    a = np.zeros(N, int)
    b = cp.zeros(N, int)
    host_launch = ("on_host", pk.RangePolicy(pk.ExecutionSpace.OpenMP, 0, N), on_host, {"a": a})
    device_launch = ("on_device", pk.RangePolicy(pk.ExecutionSpace.Cuda, 0, N), on_device, {"b": b})
    order = [device_launch, host_launch]

    # # Default order exercises the openmp -> cuda -> openmp migration path.
    # if "cuda-first" in sys.argv[1:]:
    #     order = [device_launch, host_launch]
    # else:
    #     order = [host_launch, device_launch, host_launch]
    for name, policy, workunit, args in order:
        pk.parallel_for(name, policy, workunit, **args)

    print(f"host result: {a}")
    print(f"device result: {cp.asnumpy(b)}")


main()
