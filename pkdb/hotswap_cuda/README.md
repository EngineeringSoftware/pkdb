# CUDA Kernel Hot-Swap Helpers

Minimal helpers to hot-swap CUDA kernels at debug time under `cuda-gdb`, without modifying `main.cu`.

## Files

- `hotswap_manager.cu` – driver-API helper used from `cuda-gdb` (`hs_*` functions).
- `hotswap.py` – Python module that defines the debugger commands.
- `hotswap.gdb` – small loader that imports `hotswap.py`.

## Setup

1. Build the sample (must link `hotswap/hotswap_manager.cu` into the executable).
2. Start `cuda-gdb` on the binary from the parent directory (where `test` binary lives).
3. In `cuda-gdb`:

   ```gdb
   (cuda-gdb) source hotswap/hotswap.gdb
   ```

You should see: `Hot-swap loaded. Commands: hotswap, hotswap-escape, update_kernels`.

## Commands

- **Swap two kernels in the same binary**

  ```gdb
  (cuda-gdb) break main.cu:<line before launch>
  (cuda-gdb) run
  (cuda-gdb) hotswap initData2 initData
  (cuda-gdb) continue
  ```

- **Update a kernel from PTX and redirect launches to the PTX version**

  ```gdb
  (cuda-gdb) break main.cu:<line before launch>
  (cuda-gdb) run
  (cuda-gdb) update-kernels ./ptx_output/main.ptx initData2
  (cuda-gdb) continue
  ```

- **Escape from a device breakpoint back to host**

  ```gdb
  (cuda-gdb) hotswap-escape <host_line_after_kernel>
  ```

