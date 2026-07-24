import cupy as cp
import numpy as np
import os
import time

from . import stack_profiler as sp
from . import reduction

# Global accumulators for detailed timing
_detail_timings = {"asnumpy1": 0.0, "asnumpy2": 0.0, "reduction_kernels": 0.0, "call_count": 0}


def reset_detail_timings():
    global _detail_timings
    _detail_timings = {"asnumpy1": 0.0, "asnumpy2": 0.0, "reduction_kernels": 0.0, "call_count": 0}


def get_detail_timings():
    return _detail_timings.copy()


def back_to_cpu(
    ng,
    data_out,
    data_out_np,
    data_out2,
    data_out_np2,
    ne_ar,
    ni_src,
    ns_src,
    nrg_ar,
    counter_g0_ar,
    counter_g1_ar,
    counter_g2_ar,
    counter_g3_ar,
    RN,
    ww,
):
    verbose = bool(int(os.environ.get("PKDB_BOLTZ_VERBOSE_TIMINGS", "0")))

    # Handling first reduction
    if verbose:
        t0 = time.perf_counter()
    data_out_np[0:RN, :] = cp.asnumpy(data_out[0:RN, :])
    if verbose:
        t1 = time.perf_counter()
        _detail_timings["asnumpy1"] += t1 - t0

    num_total = int(np.sum(data_out_np[0:RN, 0]) / ww)
    data_out_np[0, :] *= 2
    data_out_np[RN - 1, :] *= 2
    ne_ar[0:RN, ng] = data_out_np[0:RN, 0]
    ni_src[0:RN, ng] = ww * data_out_np[0:RN, 1]
    ns_src[0:RN, ng] = ww * data_out_np[0:RN, 2]
    nrg_ar[0:RN, ng] = data_out_np[0:RN, 3]

    # Handling collision counting reduction
    if verbose:
        t2 = time.perf_counter()
    data_out_np2[0:RN, :] = cp.asnumpy(data_out2[0:RN, :])
    if verbose:
        t3 = time.perf_counter()
        _detail_timings["asnumpy2"] += t3 - t2

    data_out_np2[0, :] *= 2
    data_out_np2[RN - 1, :] *= 2
    counter_g0_ar[0:RN] = data_out_np2[0:RN, 0]
    counter_g1_ar[0:RN] = data_out_np2[0:RN, 1]
    counter_g2_ar[0:RN] = data_out_np2[0:RN, 2]
    counter_g3_ar[0:RN] = data_out_np2[0:RN, 3]

    return (
        ne_ar,
        ni_src,
        ns_src,
        nrg_ar,
        counter_g0_ar,
        counter_g1_ar,
        counter_g2_ar,
        counter_g3_ar,
        num_total,
    )


def post_processing(
    ng,
    data_out,
    data_out_np,
    data_out2,
    data_out_np2,
    big_data_ar,
    big_tosum_ar,
    big_collct_ar,
    Na,
    Nc,
    nni,
    ne_ar,
    ni_src,
    ns_src,
    nrg_ar,
    counter_g0_ar,
    counter_g1_ar,
    counter_g2_ar,
    counter_g3_ar,
    big_N,
    RN,
    L,
    dx,
    ww,
    temp_x,
):
    verbose = bool(int(os.environ.get("PKDB_BOLTZ_VERBOSE_TIMINGS", "0")))

    if verbose:
        _detail_timings["call_count"] += 1

    temp_x[0 : Nc + nni] = (big_data_ar[0 : Nc + nni, 0] + 0.5 * dx) / (L + dx)
    data_out.fill(0)
    data_out2.fill(0)

    if verbose:
        t_red_start = time.perf_counter()

    reduction.reduction(
        4,
        big_N,
        RN,
        data_out,
        big_tosum_ar[0 : Nc + nni, :],
        temp_x[0 : Nc + nni],
        Nc + nni,
    )

    reduction.reduction(
        4,
        big_N,
        RN,
        data_out2,
        big_collct_ar[0 : Nc + nni, :],
        temp_x[0 : Nc + nni],
        Nc + nni,
    )

    if verbose:
        # Add explicit sync to measure actual kernel time
        cp.cuda.Stream.null.synchronize()
        t_red_sync = time.perf_counter()
        _detail_timings["reduction_kernels"] = _detail_timings.get("reduction_kernels", 0.0) + (
            t_red_sync - t_red_start
        )

    (
        ne_ar,
        ni_src,
        ns_src,
        nrg_ar,
        counter_g0_ar,
        counter_g1_ar,
        counter_g2_ar,
        counter_g3_ar,
        Na,
    ) = back_to_cpu(
        ng,
        data_out,
        data_out_np,
        data_out2,
        data_out_np2,
        ne_ar,
        ni_src,
        ns_src,
        nrg_ar,
        counter_g0_ar,
        counter_g1_ar,
        counter_g2_ar,
        counter_g3_ar,
        RN,
        ww,
    )

    Nnew = Nc + nni

    return (
        ne_ar,
        ni_src,
        ns_src,
        nrg_ar,
        counter_g0_ar,
        counter_g1_ar,
        counter_g2_ar,
        counter_g3_ar,
        Nnew,
        Na,
    )
