import cupy as cp
import pykokkos as pk


@pk.workunit
def init_data(i: int, view):
    view[i] = i + 1


@pk.workload
class Scale:
    def __init__(self, input_view, output_view, team_size, scale, n_ints, N):
        # PyKokkos records workload members only from annotated assignments in __init__.
        # Do not store pk.ExecutionSpace on the functor — codegen emits an invalid C++ type.
        # Use pk.set_default_space(...) + two-arg TeamPolicy, and pk.execute(space, ...) like team_policy.py.
        self.input_view: pk.View1D[pk.int32] = input_view
        self.output_view: pk.View1D[pk.float64] = output_view
        self.team_size: int = team_size
        self.scale: float = scale
        self.n_ints: int = n_ints
        self.N: int = N
        self.num_teams: int = (N + team_size - 1) // team_size

    @pk.main
    def run(self):
        pk.parallel_for(
            "scale_kernel",
            pk.TeamPolicy(self.num_teams, self.team_size),
            self.scale_kernel,
        )

    @pk.callback
    def results(self):
        in_arr = self.input_view.xp_array if hasattr(self.input_view, "xp_array") else self.input_view
        out_arr = self.output_view.xp_array if hasattr(self.output_view, "xp_array") else self.output_view
        print("input :", cp.asnumpy(in_arr))
        print("output:", cp.asnumpy(out_arr))
        expected = cp.arange(1, self.N + 1, dtype=cp.float64) * self.scale
        assert cp.allclose(out_arr, expected), "Result mismatch!"
        print("Assertion passed: output == input * scale")

    # Two scratch regions:
    #   scratch_idx - n_ints int32 elements  (passed as kwarg, variable size)
    #   scratch_val - N      float64 elements (passed as kwarg)
    @pk.workunit(
        scratch=[
            (int, lambda p, s: s.n_ints),
            (float, lambda p, s: s.N),
        ]
    )
    def scale_kernel(self, team_member: pk.TeamMember):
        offset: int = team_member.league_rank() * self.team_size
        rank: int = team_member.team_rank()

        scratch_idx: pk.ScratchView1D[int] = pk.ScratchView1D(
            team_member.team_scratch(0), self.n_ints
        )
        scratch_val: pk.ScratchView1D[float] = pk.ScratchView1D(
            team_member.team_scratch(0), self.N
        )

        scratch_idx[rank] = self.input_view[offset + rank]
        team_member.team_barrier()

        scratch_val[rank] = float(scratch_idx[rank]) * self.scale
        team_member.team_barrier()

        self.output_view[offset + rank] = scratch_val[rank]


def main():
    N = 64
    team_size = 32
    n_ints = 100
    scale = 2.5
    exec_space = pk.ExecutionSpace.DebugCuda

    input_view = cp.zeros(N, dtype=cp.int32)
    output_view = cp.zeros(N, dtype=cp.float64)

    pk.parallel_for(
        pk.RangePolicy(exec_space, 0, N),
        init_data,
        view=input_view,
    )

    print(f"N={N}  team_size={team_size}  n_ints={n_ints}  scale={scale}")

    pk.set_default_space(exec_space)
    w = Scale(pk.array(input_view), pk.array(output_view), team_size, scale, n_ints, N)
    pk.execute(exec_space, w)


if __name__ == "__main__":
    main()
