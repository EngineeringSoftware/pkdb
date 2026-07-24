// hotswap_manager.cu
//
// Linked into the target binary but NEVER called from main.cu.
// All entry points are extern "C" — invoked from cuda-gdb via `call`.
//
// Workflow from cuda-gdb (at a host breakpoint):
//
//   call (int)hs_init()
//   call (int)hs_load("./ptx_output/main.ptx", "_Z9initData2Pii")
//   call (int)hs_arg_reset()
//   call (int)hs_arg_ptr((void*)view)
//   call (int)hs_arg_int(64)
//   call (int)hs_exec(1, 256)
//   call (int)hs_clear_err()
//
// If stopped INSIDE a kernel (device breakpoint):
//   1. Set host breakpoint after kernel:  break main.cu:<line after launch>
//   2. Delete device breakpoints:         delete <bp_num>
//   3. Continue:                          continue
//   4. Now at host breakpoint — run the workflow above.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

static CUmodule   hs_mod  = nullptr;
static CUfunction hs_fn   = nullptr;

// Argument accumulator: up to 16 args, each up to 8 bytes (covers
// pointer, int, float, double, long, size_t on 64-bit).
static char  hs_abuf[16][8];
static void* hs_aptr[16];
static int   hs_argc = 0;

extern "C" {

// Ensure CUDA driver context is available.
// Safe to call multiple times; idempotent.
__attribute__((used))
int hs_init() {
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (!ctx) {
        cuInit(0);
        CUdevice dev;
        cuDeviceGet(&dev, 0);
        cuCtxCreate(&ctx, 0, dev);
    }
    printf("[hs] ctx=%p\n", (void*)ctx);
    return 0;
}

// Load a PTX (or cubin) file and extract a kernel by its mangled name.
// Unloads any previously loaded module first.
__attribute__((used))
int hs_load(const char* ptx, const char* name) {
    if (hs_mod) { cuModuleUnload(hs_mod); hs_mod = nullptr; hs_fn = nullptr; }
    CUresult r = cuModuleLoad(&hs_mod, ptx);
    if (r) { printf("[hs] load '%s' failed: %d\n", ptx, r); return (int)r; }
    r = cuModuleGetFunction(&hs_fn, hs_mod, name);
    if (r) { printf("[hs] func '%s' failed: %d\n", name, r); return (int)r; }
    printf("[hs] loaded %s :: %s\n", ptx, name);
    return 0;
}

// --- Argument accumulator ---------------------------------------------------
// Push args one-by-one, then call hs_exec / hs_exec3d.
// cuLaunchKernel expects void** where each element points to the arg value.
// We copy each value into hs_abuf[i] and set hs_aptr[i] = &hs_abuf[i].

__attribute__((used))
int hs_arg_reset() { hs_argc = 0; return 0; }

__attribute__((used))
int hs_arg_ptr(void* v) {
    memcpy(hs_abuf[hs_argc], &v, sizeof(v));
    hs_aptr[hs_argc] = hs_abuf[hs_argc];
    return ++hs_argc;
}

__attribute__((used))
int hs_arg_int(int v) {
    memcpy(hs_abuf[hs_argc], &v, sizeof(v));
    hs_aptr[hs_argc] = hs_abuf[hs_argc];
    return ++hs_argc;
}

__attribute__((used))
int hs_arg_float(float v) {
    memcpy(hs_abuf[hs_argc], &v, sizeof(v));
    hs_aptr[hs_argc] = hs_abuf[hs_argc];
    return ++hs_argc;
}

__attribute__((used))
int hs_arg_double(double v) {
    memcpy(hs_abuf[hs_argc], &v, sizeof(v));
    hs_aptr[hs_argc] = hs_abuf[hs_argc];
    return ++hs_argc;
}

__attribute__((used))
int hs_arg_long(long v) {
    memcpy(hs_abuf[hs_argc], &v, sizeof(v));
    hs_aptr[hs_argc] = hs_abuf[hs_argc];
    return ++hs_argc;
}

// --- Kernel launch ----------------------------------------------------------

// 1D launch (most common case).  Resets arg accumulator on success.
__attribute__((used))
int hs_exec(unsigned gx, unsigned bx) {
    if (!hs_fn) { printf("[hs] no kernel loaded\n"); return -1; }
    CUresult r = cuLaunchKernel(hs_fn,
                                gx, 1, 1,
                                bx, 1, 1,
                                0, nullptr, hs_aptr, nullptr);
    if (r) { printf("[hs] launch failed: %d\n", r); return (int)r; }
    r = cuCtxSynchronize();
    if (r) { printf("[hs] sync failed: %d\n", r); return (int)r; }
    printf("[hs] done (%d args, grid=%u, block=%u)\n", hs_argc, gx, bx);
    hs_argc = 0;
    return 0;
}

// Full 3D launch with shared memory.
__attribute__((used))
int hs_exec3d(unsigned gx, unsigned gy, unsigned gz,
              unsigned bx, unsigned by, unsigned bz,
              unsigned shared) {
    if (!hs_fn) { printf("[hs] no kernel loaded\n"); return -1; }
    CUresult r = cuLaunchKernel(hs_fn,
                                gx, gy, gz,
                                bx, by, bz,
                                shared, nullptr, hs_aptr, nullptr);
    if (r) { printf("[hs] launch failed: %d\n", r); return (int)r; }
    r = cuCtxSynchronize();
    if (r) { printf("[hs] sync failed: %d\n", r); return (int)r; }
    printf("[hs] done (%d args, grid=(%u,%u,%u), block=(%u,%u,%u))\n",
           hs_argc, gx, gy, gz, bx, by, bz);
    hs_argc = 0;
    return 0;
}

// Clear sticky CUDA runtime error (e.g. after a kernel crash).
// Call this before continuing the program if the original kernel faulted.
__attribute__((used))
int hs_clear_err() {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        printf("[hs] cleared error: %s\n", cudaGetErrorString(e));
    return (int)e;
}

} // extern "C"

