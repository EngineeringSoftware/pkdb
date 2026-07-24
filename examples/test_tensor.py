import pykokkos as pk
import torch


@pk.workunit
def work(wid, a):
    a[wid] += wid
    a[wid] *= 10

def main():
    N = 64
    pk.set_default_space(pk.ExecutionSpace.DebugCuda)
    a = torch.zeros(N, dtype=torch.int64, device="cuda")

    print(a)
    print(type(a))

    pk.parallel_for("work", pk.RangePolicy(0, N), work, a=a)
    print(f"Result: {a}")
    a[0]=123321
    print(a)



main()
