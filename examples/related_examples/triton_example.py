import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def work1_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    a = a + 1
    a = a * 10
    tl.store(a_ptr + offsets, a, mask=mask)


@triton.jit
def work2_kernel(a_ptr, b_ptr, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    a = a + b
    a = a * c
    tl.store(a_ptr + offsets, a, mask=mask)


# def get_compiled_kernel(jit_fn):
#     """Get the compiled kernel from the JIT cache after at least one launch."""
#     device = triton.runtime.driver.active.get_current_device()
#     kernel_cache = jit_fn.device_caches[device][0]
#     return next(iter(kernel_cache.values()))


def main():
    N = 10
    a_np = np.random.randint(100, size=(N), dtype=np.int32)
    print(a_np)

    # Convert to torch CUDA tensor
    a = torch.from_numpy(a_np).cuda()

    # work1 launch (launch does not return the kernel; we get it from the cache)
    BLOCK_SIZE = 32
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    work1_kernel[grid](a, N, BLOCK_SIZE=BLOCK_SIZE)

    # Output TTIR (and other IRs) from the compiled kernel
    # compiled_kernel = get_compiled_kernel(work1_kernel)
    # print("--- TTIR (intermediate representation) ---")
    # print(compiled_kernel.asm["ttir"])
    # Other stages: compiled_kernel.asm['source'], ['ttgir'], ['llir'], ['ptx'], ['cubin']

    print(f"Intermediate result: {a.cpu().numpy()}")

    # work2 launch
    b_np = np.random.randint(100, size=(N), dtype=np.int32)
    b = torch.from_numpy(b_np).cuda()
    work2_kernel[grid](a, b, 10, N, BLOCK_SIZE=BLOCK_SIZE)
    print(f"Final result: {a.cpu().numpy()}")


if __name__ == "__main__":
    main()
