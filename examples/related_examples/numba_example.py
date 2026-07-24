import numpy as np
import warnings
import os

# Enable CUDA simulator mode for debugging - MUST be set before importing numba.cuda
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from pdb import set_trace


@cuda.jit(debug=True)
def work1_kernel(a):
    wid = cuda.grid(1)
    # Only debug the first thread to avoid multiple debugger sessions
    if cuda.threadIdx.x == 0: # if we select all threads - it would be a mess. We would iterate over threads instead of debugging one thread at the time
        set_trace()
    if wid < a.size:
        a[wid] += 1
        a[wid] *= 10


@cuda.jit
def work2_kernel(a, b, c):
    wid = cuda.grid(1)
    if wid < a.size:
        a[wid] += b[wid]
        a[wid] *= c


def main():
    N = 10
    a = np.random.randint(100, size=(N))
    print(a)

    # Transfer data to GPU
    d_a = cuda.to_device(a)

    threads_per_block = 32
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    work1_kernel[blocks_per_grid, threads_per_block](d_a)
    d_a.copy_to_host(a)
    print(f"Intermediate result: {a}")

    # Prepare second kernel
    b = np.random.randint(100, size=(N))
    d_b = cuda.to_device(b)
    d_a = cuda.to_device(a)

    # Launch work2 kernel
    work2_kernel[blocks_per_grid, threads_per_block](d_a, d_b, 10)

    # Copy back to host
    d_a.copy_to_host(a)
    print(f"Final result: {a}")


if __name__ == "__main__":
    main()
