import math

import cupy as cp
from numba import njit


@njit(cache=True)
def EF_kernel(
    E_ar,
    ne_ar,
    ni_ar,
    curr_t: float,
    V_ar,
    V_rhs,
    Vc_diag,
    Vc_lower_diag,
    V_tempy,
    N: int,
    dx: float,
    epsilon: float,
    V0: float,
    freq: float,
):
    """
    Calculate voltage and corresponding electric field
    """

    ## VOLTAGE
    for i in range(N - 1):
        V_rhs[i] = (1.0 / epsilon) * (ni_ar[i + 1] - ne_ar[i + 1])
    V_rhs[0] = V_rhs[0] + (1.0 / (dx**2)) * V0 * math.cos(2 * cp.pi * freq * curr_t)

    V_tempy[0] = V_rhs[0] / Vc_diag[0]
    for i in range(1, N - 1):
        V_tempy[i] = (1.0 / Vc_diag[i]) * (V_rhs[i] - Vc_lower_diag[i] * V_tempy[i - 1])

    V_ar[N - 2] = V_tempy[N - 2] / Vc_diag[N - 2]
    for i in range(1, N - 1):
        V_ar[N - 2 - i] = (1.0 / Vc_diag[N - 2 - i]) * (
            V_tempy[N - 2 - i] - Vc_lower_diag[N - 2 - i + 1] * V_ar[N - 2 - i + 1]
        )
    ## ELECTRIC FIELD
    for i in range(N - 2):
        E_ar[i + 1] = -(V_ar[i + 1] - V_ar[i]) / dx
    E_ar[0] = (
        -(V_ar[0] - V0 * math.cos(2 * cp.pi * freq * curr_t)) / dx
    )  # (V_1 - V_0)/dx
    E_ar[N - 1] = -(0 - V_ar[N - 2]) / dx  # (V_N+1 - V_N)/dx
    return (V_ar, E_ar)
