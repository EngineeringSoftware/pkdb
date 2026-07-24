import ctypes
import numpy as np

from . import cuda_red


def cupy_to_numpy(array):
    ptr = array.data.ptr
    ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    np_array = np.ctypeslib.as_array(ptr, shape=array.shape)
    return np_array


def reduction(r: int, p: int, M: int, RM: int, dw, dv, dx, ptcl: int, stream):
    cuda_red.calls(
        r,
        p,
        M,
        RM,
        cupy_to_numpy(dw),
        cupy_to_numpy(dv),
        cupy_to_numpy(dx),
        ptcl,
        stream,
    )


def cuda_reduction_fast(r: int, p: int, M: int, RM: int, dw, dv, dx, ptcl: int, stream):
    cuda_red.calls(r, p, M, RM, dw, dv, dx, ptcl, stream)
