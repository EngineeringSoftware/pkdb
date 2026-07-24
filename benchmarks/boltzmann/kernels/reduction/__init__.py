reduction = None


def init_reduction():
    global reduction
    from . import pyk_reduction as reduction_impl

    reduction = reduction_impl.pyk_reduction_fast
    reduction_impl.pyk_reduction_fast.workload = None
