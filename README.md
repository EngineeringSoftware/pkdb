# pkdb

Interactive Debugger for Performance Portable Python HPC Kernels

Artifact for SC'26. Interactive debugger for PyKokkos kernels.

`pkdb` enables standard interactive debugging like breakpoints, stepping,
continuation, and variable inspection while preserving actual on-device
execution without source modification. `pkdb` also introduces three advanced
capabilities that exploit the dynamic nature of Python and PyKokkos. (i) live
code evaluation, (ii) kernel call-site substitution, and (iii) concurrent kernel
comparison. 

## Prerequisites

Installed target-specific debuggers:

| Target | Debugger |
| ------ | -------- |
| OpenMP | [gdb](https://man7.org/linux/man-pages/man1/gdb.1.html)      |
| Cuda   | [cuda-gdb](https://developer.nvidia.com/cuda-gdb) |
| HIP    | [rocgdb](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/)   |


## Installation

> [!NOTE] Modified version of pykokkos is needed temporarily; code will be merged into the main pykokkos repository shortly

First, install modified version of
pykokkos:
```bash
cd pykokkos
conda create -n pkdb python=3.13 -y
conda env update -n pkdb -f base/environment.yml
conda activate pkdb
python install_base.py install --verbose -- -DENABLE_LAYOUTS=ON -DENABLE_MEMORY_TRAITS=OFF -DENABLE_VIEW_RANKS=3 -DENABLE_THREADS=OFF -DENABLE_OPENMP=ON -DENABLE_CUDA=OFF
conda install -c conda-forge pybind11 patchelf -y
pip install -e .
```

Then, install `pkdb` itself:
```bash
cd pkdb
pip install -r requirements.txt
pip install -e .
```

## Usage

All examples are in the [`examples`](./examples/) directory. To run any example you can use `pkdb` as a python module or as a binary:

```bash
cd examples
python -m pkdb test_openmp.py
...
pkdb test_cuda.py
```

### Example debugging session:

```bash
$ python -m pkdb 01_yax.py 
> /pkdb/examples/01_yax.py(1)<module>()
-> import json
(pkdb) break 7
Breakpoint 1 at /pkdb/examples/01_yax.py:7
(pkdb) run
CUDA kernel breakpoint hit at 01_yax.py:7
(pkdb-cuda) print temp2
temp2 = <optimized out>
(pkdb-cuda) break 10
Breakpoint set at line 10
(pkdb-cuda) continue
CUDA kernel breakpoint hit at 01_yax.py:10
(pkdb-cuda) print temp2
temp2 = 259.53494435546145
  (type: float)
(pkdb-cuda) print len(y_view)
len(y_view) = 256
  (type: int)
(pkdb-cuda) print y_view[j:j+2]
y_view[j:j+2] = [0.2708710881309138, 0.8734105408078869]
  (type: list)
(pkdb-cuda) continue
--Return--
> /pkdb/examples/01_yax.py(26)<module>()->None
-> run()
(pkdb) quit
```

## Current commands list:

We support all
[`pdb`](https://docs.python.org/3/library/pdb.html#debugger-commands) commands; in addition we provide following commands for the Python side:

```
set              - Set debugger properties: set <property> <value>
parallel_launch  - Launch multiple kernels in parallel
hotswap          - Swaps kernel on invocation site to another kernel
eval             - Evaluate expression: eval <expression>
```

And the following list for kernel side commands:
```
backtrace, bt    - Show backtrace/stack trace
c, continue      - Continue execution
h, help          - Show help: h [command]
locals           - Show local variables
l, list          - List Python source around the current workunit line: list [line]
p, print         - Print variable value: p <var>
parallel_print   - Evaluate print an expression on each workunit thread
q, quit          - Quit debugger
s, step          - Step one line
thread           - Switch to thread: thread <num>
threads          - List all threads
send             - Send raw GDB MI command: send <cmd>
whereami         - Show current LOC, thread, number of active threads
```

CUDA-specific commands:
```
cuda thread <block> <thread> - switch CUDA thread to <block> <thread>
```
