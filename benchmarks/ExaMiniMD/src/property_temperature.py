import pykokkos as pk

from comm import Comm
from kernel_profiler import profiler
from system import System


@pk.workunit
def compute_workunit(
    i: int,
    acc: pk.Acc[float],
    v: pk.View2D[float],
    mass: pk.View1D[float],
    type: pk.View1D[int],
) -> None:
    mass_index: int = type[i]
    acc += (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) * mass[
        mass_index
    ]


@pk.workunit
def compute_workunit_half(
    i: int,
    acc: pk.Acc[float],
    v: pk.View2D[float],
    mass: pk.View1D[float],
    type: pk.View1D[int],
) -> None:
    mass_index: int = type[i]
    acc += (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]) * mass[mass_index]
    acc /= 2 


@pk.workload
class Temperature:
    def __init__(self, comm: Comm):
        self.comm = comm

    def compute(self, system: System) -> float:
        # with profiler.record("Temperature"):
        T = pk.parallel_reduce(
            "Temperature",
            system.N_local,
            compute_workunit,
            v=system.v,
            mass=system.mass,
            type=system.type,
        )

        dof: int = 3 * system.N - 3
        factor: float = system.mvv2e / (1.0 * dof * system.boltz)
        self.comm.reduce_float(T, 1)

        return T * factor
