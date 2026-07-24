import ctypes
from decimal import Decimal
import cupy as cp
import numpy as np

import cuda_red

l = 1
ptcl = 100000
M = 8192
RM = 256
r = 5
p = 0
xx = Decimal(l) / Decimal(ptcl)
# print(xx)
# print(type(xx))
v = np.zeros((ptcl * r,), dtype=float)
x = np.zeros((ptcl,), dtype=float)


for i in range(len(v)):
    # v[i]=i/100.0
    v[i] = 1.0
for i in range(len(x)):
    x[i] = i * xx


print(v)

dv = cp.asarray(v)
dx = cp.asarray(x)
dw = cp.zeros((M * r,), dtype=np.float64)


def cupy_to_numpy(array):
    ptr = array.data.ptr
    ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
    np_array = np.ctypeslib.as_array(ptr, shape=array.shape)
    return np_array


# void calls(int r,int p,int M,int RM, double* dw, double* dv, double* dx, int ptcl)
cuda_red.calls(
    r, p, M, RM, cupy_to_numpy(dw), cupy_to_numpy(dv), cupy_to_numpy(dx), ptcl
)
print(dw[0 : (RM * r)])
