import ctypes

import cupy as cp
import numpy as np

import cuda_red


ptcl = 10000
M = 8192
RM = 256
r = 5
p = 0

dw = cp.array((M * r,), dtype=np.float64)
dv = cp.array((ptcl * r,), dtype=np.float64)
dx = cp.array((ptcl,), dtype=np.float64)


def cupy_to_numpy(array):
    ptr = array.data.ptr
    ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    np_array = np.ctypeslib.as_array(ptr, shape=array.shape)
    return np_array


# void calls(int r,int p,int M,int RM, double* dw, double* dv, double* dx, int ptcl)
cuda_red.calls(
    r, p, M, RM, cupy_to_numpy(dw), cupy_to_numpy(dv), cupy_to_numpy(dx), ptcl
)
