"""
Example: workunit that uses another function (no workload, only workunits).

Without a workload/functor class, a workunit can still use another "function"
by defining an inner function and passing it to a pattern (nested parallelism).
Here the outer workunit defines inner_reduce and passes it to
pk.parallel_reduce(pk.TeamThreadRange(...), inner_reduce).
"""

import pykokkos as pk


@pk.workunit
def row_sum(team_member, acc, cols, out_view, A_view):
    j: int = team_member.league_rank()

    def inner_reduce(i: int, inner_acc: pk.Acc[float]):
        inner_acc += A_view[j][i]

    row_total: float = pk.parallel_reduce(
        pk.TeamThreadRange(team_member, cols), inner_reduce)

    if team_member.team_rank() == 0:
        out_view[j] = row_total
    acc += row_total


def main():
    N = 4
    M = 5
    pk.set_default_space(pk.ExecutionSpace.DebugCuda)

    out_view: pk.View1D[pk.double] = pk.View([N], pk.double)
    A_view: pk.View2D[pk.double] = pk.View([N, M], pk.double)
    for j in range(N):
        for i in range(M):
            A_view[j][i] = 1.0

    p = pk.TeamPolicy(N, pk.AUTO)
    total: float = pk.parallel_reduce(
        p, row_sum, cols=M, out_view=out_view, A_view=A_view)

    print("Row sums (each row sums to M):")
    for j in range(N):
        print(f"  out_view[{j}] = {out_view[j]}")
    print(f"Total (parallel_reduce result): {total}")


if __name__ == "__main__":
    main()
