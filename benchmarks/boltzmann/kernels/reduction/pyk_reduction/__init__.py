import math

import cupy as cp
import numpy as np
import pykokkos as pk


@pk.workload
class ReductionOpenMP:
    def __init__(self, r, M, RM, w, v, x, p, ns):
        self.w: pk.View1D[float] = pk.array(np.ravel(w, order="A"))
        self.v: pk.View1D[float] = pk.array(np.ravel(v, order="A"))
        self.x: pk.View1D[float] = pk.array(x)

        self.r: int = r
        self.RM: int = RM
        self.M: int = M
        self.p: int = p
        self.scratch_size: int = int(ns / 8)

    @pk.main
    def run(self):
        blockx: int = 64
        blocky: int = self.r
        x_threads: int = blockx * (self.p // blockx + 1)
        y_threads: int = blocky * (self.r // blocky + 1)

        bloks: int = math.ceil(self.M / self.RM)

        league_size: int = self.M // bloks + 1
        team_size: int = bloks

        pk.parallel_for(pk.MDRangePolicy([0, 0], [x_threads, y_threads]), self.reda)
        pk.parallel_for(
            pk.TeamPolicy(league_size, team_size),
            self.redcheck,
        )

    @pk.workunit
    def reda(self, i: int, j: int):
        if i < self.p and j < self.r:
            b: int = self.x[i] * self.M
            pk.atomic_add(self.w, [self.r * b + j], self.v[self.r * i + j])

    @pk.workunit(
        scratch=[
            (float, lambda p, s: s.scratch_size),
        ]
    )
    def redcheck(self, team_member: pk.TeamMember):
        i: int = team_member.league_rank() * team_member.team_size() + team_member.team_rank()
        tid: int = team_member.team_rank()

        if i < self.M:
            bb: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), self.scratch_size)
            for j in range(self.r):
                bb[self.r * tid + j] = self.w[self.r * i + j]

            team_member.team_barrier()

            s: pk.uint32 = team_member.team_size() / 2
            while s > 0:
                if tid < s:
                    for j in range(self.r):
                        bb[self.r * tid + j] += bb[self.r * (tid + s) + j]

                team_member.team_barrier()
                s >>= 1

            if tid == 0:
                for j in range(self.r):
                    self.w[self.r * team_member.league_rank() + j] = bb[j]


@pk.workload(
    w=pk.ViewTypeInfo(space=pk.HIPSpace),
    v=pk.ViewTypeInfo(space=pk.HIPSpace),
    x=pk.ViewTypeInfo(space=pk.HIPSpace),
)
class ReductionHIP:
    def __init__(self, r, M, RM, w, v, x, p, ns):
        self.w: pk.View1D[float] = pk.array(cp.ravel(w, order="A"))
        self.v: pk.View1D[float] = pk.array(cp.ravel(v, order="A"))
        self.x: pk.View1D[float] = pk.array(x)

        self.r: int = r
        self.RM: int = RM
        self.M: int = M
        self.p: int = p
        self.scratch_size: int = int(ns / 8)

    @pk.main
    def run(self):
        blockx: int = 64
        blocky: int = self.r
        x_threads: int = blockx * (self.p // blockx + 1)
        y_threads: int = blocky * (self.r // blocky + 1)

        bloks: int = math.ceil(self.M / self.RM)

        league_size: int = self.M // bloks + 1
        team_size: int = bloks

        pk.parallel_for(pk.MDRangePolicy([0, 0], [x_threads, y_threads]), self.reda)
        pk.parallel_for(
            pk.TeamPolicy(league_size, team_size),
            self.redcheck,
        )

    @pk.workunit
    def reda(self, i: int, j: int):
        if i < self.p and j < self.r:
            b: int = self.x[i] * self.M
            pk.atomic_add(self.w, [self.r * b + j], self.v[self.r * i + j])

    @pk.workunit(
        scratch=[
            (float, lambda p, s: s.scratch_size),
        ]
    )
    def redcheck(self, team_member: pk.TeamMember):
        i: int = team_member.league_rank() * team_member.team_size() + team_member.team_rank()
        tid: int = team_member.team_rank()

        if i < self.M:
            bb: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), self.scratch_size)
            for j in range(self.r):
                bb[self.r * tid + j] = self.w[self.r * i + j]

            team_member.team_barrier()

            s: pk.uint32 = team_member.team_size() / 2
            while s > 0:
                if tid < s:
                    for j in range(self.r):
                        bb[self.r * tid + j] += bb[self.r * (tid + s) + j]

                team_member.team_barrier()
                s >>= 1

            if tid == 0:
                for j in range(self.r):
                    self.w[self.r * team_member.league_rank() + j] = bb[j]


@pk.workload(
    w=pk.ViewTypeInfo(space=pk.CudaSpace),
    v=pk.ViewTypeInfo(space=pk.CudaSpace),
    x=pk.ViewTypeInfo(space=pk.CudaSpace),
)
class ReductionCuda:
    def __init__(self, r, M, RM, w, v, x, p, ns):
        self.w: pk.View1D[float] = pk.array(cp.ravel(w, order="A"))
        self.v: pk.View1D[float] = pk.array(cp.ravel(v, order="A"))
        self.x: pk.View1D[float] = pk.array(x)

        self.r: int = r
        self.RM: int = RM
        self.M: int = M
        self.p: int = p
        self.scratch_size: int = int(ns / 8)

    @pk.main
    def run(self):
        blockx: int = 64
        blocky: int = self.r
        x_threads: int = blockx * (self.p // blockx + 1)
        y_threads: int = blocky * (self.r // blocky + 1)

        bloks: int = math.ceil(self.M / self.RM)

        league_size: int = self.M // bloks + 1
        team_size: int = bloks

        pk.parallel_for(pk.MDRangePolicy([0, 0], [x_threads, y_threads]), self.reda)
        pk.parallel_for(
            pk.TeamPolicy(league_size, team_size),
            self.redcheck,
        )

    @pk.workunit
    def reda(self, i: int, j: int):
        if i < self.p and j < self.r:
            b: int = self.x[i] * self.M
            pk.atomic_add(self.w, [self.r * b + j], self.v[self.r * i + j])

    @pk.workunit(
        scratch=[
            (float, lambda p, s: s.scratch_size),
        ]
    )
    def redcheck(self, team_member: pk.TeamMember):
        i: int = team_member.league_rank() * team_member.team_size() + team_member.team_rank()
        tid: int = team_member.team_rank()

        if i < self.M:
            bb: pk.ScratchView1D[float] = pk.ScratchView1D(team_member.team_scratch(0), self.scratch_size)
            for j in range(self.r):
                bb[self.r * tid + j] = self.w[self.r * i + j]

            team_member.team_barrier()

            s: pk.uint32 = team_member.team_size() / 2
            while s > 0:
                if tid < s:
                    for j in range(self.r):
                        bb[self.r * tid + j] += bb[self.r * (tid + s) + j]

                team_member.team_barrier()
                s >>= 1

            if tid == 0:
                for j in range(self.r):
                    self.w[self.r * team_member.league_rank() + j] = bb[j]


def reduction(r: int, M: int, RM: int, dw, dv, dx, ptcl: int):
    bloks: int = -(M // -RM)
    ns: int = bloks * r * 8

    if pk.get_default_space() in {pk.OpenMP, pk.DebugOpenMP}:
        r = ReductionOpenMP(r, M, RM, dw, dv, dx, ptcl, ns)
    elif pk.get_default_space() in {pk.Cuda, pk.DebugCuda}:
        r = ReductionCuda(r, M, RM, dw, dv, dx, ptcl, ns)
    elif pk.get_default_space() in {pk.HIP, pk.DebugHIP}:
        r = ReductionHIP(r, M, RM, dw, dv, dx, ptcl, ns)

    pk.execute(pk.Default, r)


def pyk_reduction_fast(r: int, M: int, RM: int, dw, dv, dx, ptcl: int):
    space = pk.get_default_space()
    bloks: int = -(M // -RM)
    ns: int = bloks * r * 8

    def _view_len(view) -> int:
        try:
            return int(view.shape[0])
        except Exception:
            return int(len(view))

    # `kernels/reduction/init_reduction()` clears `pyk_reduction_fast.workload` to None.
    # Re-create on first use, and re-create if v/x lengths change.
    needs_recreate = (
        getattr(pyk_reduction_fast, "workload", None) is None
        or _view_len(pyk_reduction_fast.workload.v) != dv.size
        or _view_len(pyk_reduction_fast.workload.x) != dx.size
    )

    if needs_recreate:
        if space in {pk.OpenMP, pk.DebugOpenMP}:
            pyk_reduction_fast.workload = ReductionOpenMP(r, M, RM, dw, dv, dx, ptcl, ns)
        elif space in {pk.Cuda, pk.DebugCuda}:
            pyk_reduction_fast.workload = ReductionCuda(r, M, RM, dw, dv, dx, ptcl, ns)
        elif space in {pk.HIP, pk.DebugHIP}:
            pyk_reduction_fast.workload = ReductionHIP(r, M, RM, dw, dv, dx, ptcl, ns)

    # Reuse existing storage to avoid repeated pk.array allocations.
    if space in {pk.OpenMP, pk.DebugOpenMP}:
        pyk_reduction_fast.workload.w[:] = np.ravel(dw, order="A")
        pyk_reduction_fast.workload.v[:] = np.ravel(dv, order="A")
    else:
        pyk_reduction_fast.workload.w[:] = cp.ravel(dw, order="A")
        pyk_reduction_fast.workload.v[:] = cp.ravel(dv, order="A")

    pyk_reduction_fast.workload.x[:] = dx
    pyk_reduction_fast.workload.r = r
    pyk_reduction_fast.workload.RM = RM
    pyk_reduction_fast.workload.M = M
    pyk_reduction_fast.workload.p = ptcl
    pyk_reduction_fast.workload.scratch_size = int(ns / 8)

    pk.execute(pk.Default, pyk_reduction_fast.workload)


if pk.get_default_space() in {pk.OpenMP, pk.DebugOpenMP}:
    pyk_reduction_fast.workload = ReductionOpenMP(0, 0, 0, np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), 0, 0)
elif pk.get_default_space() in {pk.Cuda, pk.DebugCuda}:
    pyk_reduction_fast.workload = ReductionCuda(0, 0, 0, cp.zeros((1, 1)), cp.zeros((1, 1)), cp.zeros(1), 0, 0)
elif pk.get_default_space() in {pk.HIP, pk.DebugHIP}:
    pyk_reduction_fast.workload = ReductionHIP(0, 0, 0, cp.zeros((1, 1)), cp.zeros((1, 1)), cp.zeros(1), 0, 0)
else:
    pyk_reduction_fast.workload = None
