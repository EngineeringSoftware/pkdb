from numba import njit, prange


@njit(cache=True)
def heavies_kernel_fluid(
    E_ar,
    ni_ar,
    ni_rhs,
    Ji_ar,
    ns_ar,
    ns_rhs,
    Js_ar,
    dx: float,
    dt: float,
    mu_i: float,
    D_i: float,
    D_s: float,
    N: int,
):
    """
    Move the ions / heavies with advection / diffusion
    """

    # Ion BCs
    ni_ar[0] = ni_ar[1] * (
        (D_i / dx - 0.5 * mu_i * max(E_ar[0], 0))
        / (D_i / dx + 0.5 * mu_i * max(E_ar[0], 0))
    )
    ni_ar[N] = ni_ar[N - 1] * (
        (D_i / dx + 0.5 * mu_i * min(E_ar[N - 1], 0))
        / (D_i / dx - 0.5 * mu_i * min(E_ar[N - 1], 0))
    )
    ns_ar[0] = 0
    ns_ar[N] = 0

    for i in prange(N):
        Ji_ar[i] = (
            mu_i * (0.5 * (ni_ar[i] + ni_ar[i + 1])) * E_ar[i]
            - D_i * (ni_ar[i + 1] - ni_ar[i]) / dx
        )
        Js_ar[i] = -D_s * (ns_ar[i + 1] - ns_ar[i]) / dx

    # n_i, n_s update
    for i in prange(1, N):
        ni_rhs[i] = ni_ar[i] - dt * (Ji_ar[i] - Ji_ar[i - 1]) / dx
        if ni_rhs[i] < 0 and ni_ar[i] > 0:
            print(
                "NI BREAKING IN PDE SOLVE, index = ",
                i,
                ", 3 values are = ",
                ni_ar[i - 1],
                ni_ar[i],
                ni_ar[i + 1],
            )
        ni_ar[i] = ni_rhs[i]

        ns_rhs[i] = ns_ar[i] - dt * (Js_ar[i] - Js_ar[i - 1]) / dx
        ns_ar[i] = ns_rhs[i]

    ni_ar[0] = ni_ar[1] * (
        (D_i / dx - 0.5 * mu_i * max(E_ar[0], 0))
        / (D_i / dx + 0.5 * mu_i * max(E_ar[0], 0))
    )
    ni_ar[N] = ni_ar[N - 1] * (
        (D_i / dx + 0.5 * mu_i * min(E_ar[N - 1], 0))
        / (D_i / dx - 0.5 * mu_i * min(E_ar[N - 1], 0))
    )
    ns_ar[0] = 0
    ns_ar[N] = 0

    return (ni_ar, ns_ar)
