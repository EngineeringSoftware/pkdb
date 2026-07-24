import pykokkos as pk


@pk.workunit
def scratch_with_double_float(team_member: pk.TeamMember):
    scratch_mem_d: pk.ScratchView1D[pk.double] = pk.ScratchView1D(team_member.team_scratch(0))
    scratch_mem_f: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0))
    pk.printf("Total threads: %d\n", team_member.team_scratch(0))
    scratch_mem_d[0] = -1
    scratch_mem_f[0] = 1.0
    pk.printf("Im here now: %d\n", team_member.team_scratch(0))


def test_gh_180():
    pk.parallel_for("double_float_scratch", pk.TeamPolicy(2, 2), scratch_with_double_float)


pk.set_default_space(pk.ExecutionSpace.DebugOpenMP)
test_gh_180()
print("execution done")
