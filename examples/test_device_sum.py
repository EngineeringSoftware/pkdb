import cupy as cp
import pykokkos as pk


@pk.workunit
def initialize_array(i, a):
    a[i] = i


@pk.workunit
def double_array(i, a):
    a[i] = a[i] * 2


def main():
    N = 100
    pk.set_default_space(pk.ExecutionSpace.DebugCuda)
    
    a = cp.zeros(N, dtype=cp.int32)
    pk.parallel_for("init", N, initialize_array, a=a)
    pk.parallel_for("double", N, double_array, a=a)
    
    expected_sum = sum(range(N)) * 2
    print(f"Expected sum: {expected_sum}")
    print(f"Actual sum: {cp.sum(a)}")


if __name__ == "__main__":
    main()
