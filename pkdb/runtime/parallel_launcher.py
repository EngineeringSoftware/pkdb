import multiprocessing as mp
import inspect
import numbers
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple
import pickle
import io
import contextlib

# Exec-space names from "parallel-launch work pk.DebugCuda …"
_GPU_EXEC_SPACE_NAMES = frozenset({"Cuda", "DebugCuda", "HIP", "DebugHIP"})

# mut:param / mod:param — workunit kwarg name is param; value from frame (CUDA IPC when spawn).
_MUTABLE_CUDA_KWARG_PREFIXES = ("mut:", "mod:")


def _policy_targets_device_execution(policy: Any) -> bool:
    try:
        from pykokkos.interface.execution_space import DeviceExecutionSpace
    except ImportError:
        return False
    inst = getattr(policy, "space", None)
    es = getattr(inst, "space", None) if inst is not None else None
    return es is not None and es in DeviceExecutionSpace


def _kernel_specs_need_spawn_for_cuda(kernel_specs: List["KernelSpec"]) -> bool:
    """
    Use multiprocessing \"spawn\" only when a **GPU** execution space is requested or the
    policy targets a device space **and** the user did not override with an explicit host
    ``pk.<ExecSpace>`` on the command line.

    Otherwise ``pk.RangePolicy(0, N)`` keeps the process default (often DebugCuda) while
    ``parallel_launch ... pk.OpenMP ...`` still means host — without this, we would spawn a
    cold interpreter and pay full PyKokkos JIT again (minutes) and lose the parent's cache.
    """
    for spec in kernel_specs:
        if spec.exec_space:
            if spec.exec_space in _GPU_EXEC_SPACE_NAMES:
                return True
            continue
        if spec.args and _policy_targets_device_execution(spec.args[0]):
            return True
    return False


def _ensure_main_spec_for_mp_spawn() -> None:
    main = sys.modules.get("__main__")
    if main is None or getattr(main, "__spec__", None) is not None:
        return
    import importlib.util

    spec = importlib.util.spec_from_loader("__main__", loader=None)
    if spec is not None:
        main.__spec__ = spec


def _parallel_launch_mp_context(*, use_spawn: bool):
    if use_spawn:
        _ensure_main_spec_for_mp_spawn()
        try:
            return mp.get_context("spawn")
        except ValueError:
            return mp.get_context()
    if sys.platform != "win32":
        return mp.get_context("fork")
    return mp.get_context()


@dataclass
class KernelSpec:
    name: str
    exec_space: Optional[str]
    args: List[Any]
    kwargs: Dict[str, Any]
    mutable_cuda_arg_names: frozenset = field(default_factory=frozenset)
    #: ``for`` (default), ``reduce``, or ``scan`` — must match the workunit (e.g. reduce workunits need ``reduce``).
    dispatch: str = "for"


@dataclass
class _IpcArrayToken:
    """Picklable CUDA IPC handle + shape/dtype for CuPy arrays across spawn."""

    handle_hex: str
    shape: List[int]
    dtype_str: str

    def open(self) -> tuple[Any, int]:
        import cupy as cp
        import numpy as np
        from cupy.cuda import MemoryPointer, UnownedMemory

        handle_bytes = bytes.fromhex(self.handle_hex)
        cp.cuda.Device(0).use()
        ptr = int(cp.cuda.runtime.ipcOpenMemHandle(handle_bytes))
        itemsize = int(np.dtype(self.dtype_str).itemsize)
        total_bytes = int(np.prod(self.shape, dtype=np.int64)) * itemsize
        mem = UnownedMemory(ptr, total_bytes, owner=None)
        memptr = MemoryPointer(mem, 0)
        view = cp.ndarray(tuple(self.shape), dtype=self.dtype_str, memptr=memptr)
        return view, ptr

    @staticmethod
    def from_cupy_array(arr: Any) -> "_IpcArrayToken":
        import cupy as cp

        handle = cp.cuda.runtime.ipcGetMemHandle(arr.data.ptr)
        return _IpcArrayToken(handle_hex=handle.hex(), shape=list(arr.shape), dtype_str=str(arr.dtype))


class _ResultQueue(Protocol):
    def put(self, item: Dict[str, Any]) -> None: ...


def _policy_to_recipe(policy: Any) -> Dict[str, Any]:
    from pykokkos.interface.execution_policy import RangePolicy, TeamPolicy, MDRangePolicy

    space_inst = getattr(policy, "space", None)
    es = getattr(space_inst, "space", space_inst)
    space_name = es.value if hasattr(es, "value") else str(es)

    if isinstance(policy, RangePolicy):
        return {"type": "RangePolicy", "space": space_name, "begin": policy.begin, "end": policy.end}
    if isinstance(policy, TeamPolicy):
        return {
            "type": "TeamPolicy",
            "space": space_name,
            "league_size": policy.league_size,
            "team_size": policy.team_size,
            "vector_length": policy.vector_length,
        }
    if isinstance(policy, MDRangePolicy):
        return {
            "type": "MDRangePolicy",
            "space": space_name,
            "begin": list(policy.begin),
            "end": list(policy.end),
            "tiling": list(policy.tiling) if policy.tiling else None,
        }
    raise TypeError(f"Cannot serialize policy of type {type(policy).__name__}")


def _recipe_to_policy(recipe: Dict[str, Any]) -> Any:
    """Reconstruct a live ExecutionPolicy inside a child process from a recipe dict."""
    import pykokkos as pk

    space = getattr(pk.ExecutionSpace, recipe["space"], pk.ExecutionSpace.Default)
    ptype = recipe["type"]

    if ptype == "RangePolicy":
        return pk.RangePolicy(space, recipe["begin"], recipe["end"])
    if ptype == "TeamPolicy":
        return pk.TeamPolicy(space, recipe["league_size"], recipe["team_size"])
    if ptype == "MDRangePolicy":
        kw: Dict[str, Any] = {"space": space}
        if recipe.get("tiling"):
            kw["tiling"] = recipe["tiling"]
        return pk.MDRangePolicy(recipe["begin"], recipe["end"], **kw)
    raise TypeError(f"Unknown policy type in recipe: {ptype}")


def _coerce_policy_execution_space(policy: Any, exec_space_name: Optional[str], pk: Any) -> Any:
    """
    Rebuild ``policy`` so its execution space matches ``exec_space_name`` (e.g. OpenMP).

    ``pk.RangePolicy(0, N)`` uses the process default space (often DebugCuda); combined with
    ``parallel_launch ... pk.OpenMP ...`` the arrays are converted to NumPy but PyKokkos still
    reads the policy's device space unless we rebind the policy.
    """
    if not exec_space_name or policy is None:
        return policy
    if isinstance(policy, numbers.Integral):
        return policy
    try:
        from pykokkos.interface.execution_policy import ExecutionPolicy, RangePolicy, TeamPolicy, MDRangePolicy
    except ImportError:
        return policy
    if not isinstance(policy, ExecutionPolicy):
        return policy
    space_enum = getattr(pk.ExecutionSpace, exec_space_name, None)
    if space_enum is None:
        return policy

    if isinstance(policy, RangePolicy):
        return pk.RangePolicy(space_enum, policy.begin, policy.end)
    if isinstance(policy, TeamPolicy):
        vl = getattr(policy, "vector_length", -1)
        return pk.TeamPolicy(space_enum, policy.league_size, policy.team_size, vl)
    if isinstance(policy, MDRangePolicy):
        tiling = list(policy.tiling) if getattr(policy, "tiling", None) is not None else None
        return pk.MDRangePolicy(
            list(policy.begin),
            list(policy.end),
            tiling=tiling,
            space=space_enum,
            iter_outer=policy.iter_outer,
            iter_inner=policy.iter_inner,
            rank=None,
        )
    return policy


def _convert_arrays_for_space(arrays: Dict[str, Any], target_space: str) -> Dict[str, Any]:
    converted = {}
    is_gpu_space = target_space in _GPU_EXEC_SPACE_NAMES

    for name, arr in arrays.items():
        if name.startswith("_"):
            converted[name] = arr
            continue
        arr_type = type(arr).__module__

        if is_gpu_space:
            if arr_type.startswith("numpy"):
                try:
                    import cupy as cp

                    converted[name] = cp.asarray(arr)
                except ImportError:
                    raise RuntimeError(f"CuPy required for {target_space} but not available")
            else:
                converted[name] = arr
        else:
            if arr_type.startswith("cupy"):
                try:
                    import cupy as cp

                    converted[name] = cp.asnumpy(arr)
                except:
                    import numpy as np

                    converted[name] = np.array(arr)
            else:
                converted[name] = arr

    return converted


def _serialize_arrays_to_cpu(arrays: Dict[str, Any]) -> Dict[str, Any]:
    serialized = {}
    for name, arr in arrays.items():
        arr_type = type(arr).__module__

        if arr_type.startswith("cupy"):
            try:
                import cupy as cp

                serialized[name] = cp.asnumpy(arr)
            except:
                serialized[name] = arr
        elif arr_type.startswith("numpy"):
            import numpy as np

            serialized[name] = np.copy(arr)
        else:
            try:
                serialized[name] = arr.copy()
            except:
                serialized[name] = arr

    return serialized


def _run_kernel_in_process(kernel_spec: KernelSpec, global_ns: Dict, result_queue: _ResultQueue):
    ipc_ptrs: List[int] = []
    try:
        import numpy as np
        import pykokkos as pk
        from pykokkos.interface.execution_policy import ExecutionPolicy

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            # Set execution space
            if kernel_spec.exec_space:
                space_obj = getattr(pk.ExecutionSpace, kernel_spec.exec_space, None)
                if space_obj is not None:
                    pk.set_default_space(space_obj)

            copied_kwargs = dict(kernel_spec.kwargs)

            ipc_key_list = [k for k, v in copied_kwargs.items() if isinstance(v, _IpcArrayToken)]
            host_numpy_keys = [
                k for k, v in copied_kwargs.items() if not k.startswith("_") and isinstance(v, np.ndarray)
            ]
            if ipc_key_list and host_numpy_keys:
                array_transfer = "mixed"
            elif ipc_key_list:
                array_transfer = "cuda_ipc_inplace"
            else:
                array_transfer = "host_copy"

            for key, val in list(copied_kwargs.items()):
                if isinstance(val, _IpcArrayToken):
                    arr, ptr = val.open()
                    ipc_ptrs.append(ptr)
                    copied_kwargs[key] = arr

            if kernel_spec.exec_space:
                copied_kwargs = _convert_arrays_for_space(copied_kwargs, kernel_spec.exec_space)

            if not copied_kwargs.pop("_is_parallel_for", False):
                raise RuntimeError("parallel_launch: missing _is_parallel_for")

            raw_policy = kernel_spec.args[0]
            raw_kernel = kernel_spec.args[1]

            if isinstance(raw_policy, dict) and "type" in raw_policy:
                policy = _recipe_to_policy(raw_policy)
            elif isinstance(raw_policy, (int, np.integer)) or isinstance(raw_policy, ExecutionPolicy):
                policy = raw_policy
            else:
                result_queue.put(
                    {
                        "success": False,
                        "error": (
                            f"parallel-launch: second token must be an ExecutionPolicy or int, "
                            f"got type {type(raw_policy).__name__!r}."
                        ),
                        "name": kernel_spec.name,
                        "stdout": stdout_capture.getvalue(),
                        "stderr": stderr_capture.getvalue(),
                    }
                )
                return

            if kernel_spec.exec_space:
                policy = _coerce_policy_execution_space(policy, kernel_spec.exec_space, pk)

            if isinstance(raw_kernel, str):
                kernel = global_ns.get(raw_kernel)
                if kernel is None:
                    source_key = f"_source_{raw_kernel}"
                    srcfile_key = f"_srcfile_{raw_kernel}"
                    if source_key in global_ns:
                        ns: Dict[str, Any] = {}
                        exec("import pykokkos as pk", ns)
                        exec(
                            compile(
                                global_ns[source_key],
                                global_ns.get(srcfile_key, "<parallel_launch>"),
                                "exec",
                            ),
                            ns,
                        )
                        kernel = ns.get(raw_kernel)
                if kernel is None:
                    raise NameError(f"Kernel '{raw_kernel}' not found in worker namespace")
            else:
                kernel = raw_kernel

            kind = getattr(kernel_spec, "dispatch", None) or "for"
            try:
                if kind == "for":
                    result = pk.parallel_for(kernel_spec.name, policy, kernel, **copied_kwargs)
                elif kind == "reduce":
                    result = pk.parallel_reduce(kernel_spec.name, policy, kernel, **copied_kwargs)
                elif kind == "scan":
                    result = pk.parallel_scan(kernel_spec.name, policy, kernel, **copied_kwargs)
                else:
                    raise RuntimeError(f"parallel_launch: unknown dispatch {kind!r} (use for, reduce, or scan)")
            except Exception as e:
                error_msg = (
                    f"Failed to launch parallel_{kind} with '{kernel_spec.name}': "
                    f"{str(e)}\n{traceback.format_exc()}"
                )
                result_queue.put(
                    {
                        "success": False,
                        "error": error_msg,
                        "name": kernel_spec.name,
                        "stdout": stdout_capture.getvalue(),
                        "stderr": stderr_capture.getvalue(),
                    }
                )
                return

            output_arrays: Dict[str, Any] = {}
            for key, val in copied_kwargs.items():
                if key.startswith("_"):
                    continue
                arr_type = type(val).__module__
                if arr_type.startswith("cupy"):
                    try:
                        import cupy as cp

                        cp.cuda.Device(0).synchronize()
                        output_arrays[key] = cp.asnumpy(val)
                    except Exception:
                        output_arrays[key] = val
                elif arr_type.startswith("numpy"):
                    output_arrays[key] = val

        result_queue.put(
            {
                "success": True,
                "result": result,
                "name": kernel_spec.name,
                "output_arrays": output_arrays,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "array_transfer": array_transfer,
                "cuda_ipc_kwarg_keys": ipc_key_list,
            }
        )

    except SystemExit as e:
        code = e.code
        msg = code if isinstance(code, str) else repr(code)
        result_queue.put(
            {
                "success": False,
                "error": (
                    f"Worker exited in '{kernel_spec.name}' (often PyKokkos translation): {msg}\n"
                    "Hint: reduction workunits need: parallel_launch reduce <name> ..."
                ),
                "name": kernel_spec.name,
                "stdout": "",
                "stderr": "",
            }
        )
    except Exception as e:
        result_queue.put(
            {
                "success": False,
                "error": f"Process error in '{kernel_spec.name}': {str(e)}\n{traceback.format_exc()}",
                "name": kernel_spec.name,
                "stdout": "",
                "stderr": "",
            }
        )
    finally:
        import cupy as cp

        for ptr in ipc_ptrs:
            cp.cuda.runtime.ipcCloseMemHandle(ptr)


def parse_parallel_launch_command(arg: str, frame_globals: Dict) -> List[KernelSpec]:
    kernel_specs = []
    kernel_strings = [s.strip() for s in arg.split(";") if s.strip()]

    for kernel_str in kernel_strings:
        parts = kernel_str.split()
        if not parts:
            continue

        dispatch = "for"
        if parts[0] in ("reduce", "scan"):
            dispatch = parts[0]
            parts = parts[1:]

        if len(parts) < 2:
            raise ValueError(f"Need at least kernel name and policy: '{kernel_str}'")

        kernel_name = parts[0]
        policy_name = parts[1]

        exec_space = None
        if policy_name.startswith("pk."):
            exec_space = policy_name.replace("pk.", "")
            if len(parts) < 3:
                raise ValueError(f"Need policy after exec space: '{kernel_str}'")
            policy_name = parts[2]
            args_start = 3
        else:
            args_start = 2

        # Get policy and kernel from frame
        try:
            policy = eval(policy_name, frame_globals)
            kernel = eval(kernel_name, frame_globals)
        except Exception as e:
            raise ValueError(f"Failed to resolve kernel/policy in '{kernel_str}': {e}")

        # Parse remaining arguments as kwargs
        kwargs: Dict[str, Any] = {}
        mutable_cuda: set[str] = set()
        for i in range(args_start, len(parts)):
            part = parts[i]
            prefix = next((p for p in _MUTABLE_CUDA_KWARG_PREFIXES if part.startswith(p)), None)
            if prefix is not None:
                rest = part[len(prefix) :]
                if not rest.strip():
                    raise ValueError(f"Empty {prefix[:-1]}: in token '{part}'")
                if "=" in rest:
                    mkey, val_str = rest.split("=", 1)
                    mkey = mkey.strip()
                    val_str = val_str.strip()
                    if not mkey:
                        raise ValueError(f"Missing parameter name in '{part}'")
                    if not val_str:
                        raise ValueError(f"Missing value after = in '{part}'")
                    try:
                        val = eval(val_str, frame_globals)
                    except Exception as e:
                        raise ValueError(f"{prefix}{rest!r}: {e}") from e
                    kwargs[mkey] = val
                    mutable_cuda.add(mkey)
                else:
                    name = rest.strip()
                    try:
                        arg_val = eval(name, frame_globals)
                    except Exception as e:
                        raise ValueError(f"{prefix}{name!r}: cannot resolve in current frame ({e})") from e
                    kwargs[name] = arg_val
                    mutable_cuda.add(name)
                continue
            if "=" in part:
                key, val_str = part.split("=", 1)
                key = key.strip()
                val_str = val_str.strip()
                if ":" in key:
                    mod, _, param = key.partition(":")
                    mod = mod.strip()
                    param = param.strip()
                    if mod and param and f"{mod}:" in _MUTABLE_CUDA_KWARG_PREFIXES:
                        if not val_str:
                            raise ValueError(f"Incomplete assignment {part!r}; expected {mod}:{param}=<expr>")
                        try:
                            val = eval(val_str, frame_globals)
                        except Exception as e:
                            raise ValueError(f"Cannot evaluate {mod}:{param}={val_str!r} in current frame: {e}") from e
                        kwargs[param] = val
                        mutable_cuda.add(param)
                        continue
                try:
                    val = eval(val_str, frame_globals)
                    kwargs[key] = val
                except Exception as e:
                    raise ValueError(f"Cannot evaluate {key!r}={val_str!r} in current frame: {e}") from e
            else:
                try:
                    kwargs[part] = eval(part, frame_globals)
                except Exception:
                    kwargs[part] = part

        kernel_specs.append(
            KernelSpec(
                name=kernel_name,
                exec_space=exec_space,
                args=[policy, kernel],
                kwargs={**kwargs, "_is_parallel_for": True},
                mutable_cuda_arg_names=frozenset(mutable_cuda),
                dispatch=dispatch,
            )
        )

    return kernel_specs


def _parallel_launch_join_timeout_s() -> int:
    try:
        return max(30, int(os.environ.get("PKDB_PARALLEL_LAUNCH_TIMEOUT", "180")))
    except ValueError:
        return 180


def _parallel_launch_slot_label(spec: KernelSpec, index: int, total: int) -> str:
    """Unique CLI label when the same workunit name appears in several ``;``-separated launches."""
    label = spec.name
    if spec.exec_space:
        label = f"{label} [{spec.exec_space}]"
    if total > 1:
        label = f"{label} (#{index + 1} of {total})"
    return label


def _cuda_parallel_launch_prefer_inprocess() -> bool:
    """
    If true (default), GPU ``parallel_launch`` runs in the **current** process so PyKokkos reuses
    the same JIT cache and CUDA context as an ordinary ``parallel_*`` call.

    Set ``PKDB_PARALLEL_LAUNCH_CUDA_INPROCESS=0`` to force the legacy **spawn** path (cold compile
    in a child process, often minutes for DebugCuda).
    """
    v = os.environ.get("PKDB_PARALLEL_LAUNCH_CUDA_INPROCESS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _parallel_launch_run_in_process_slots(
    kernel_specs: List[KernelSpec],
    serializable_globals: Dict[str, Any],
    total_slots: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    ordered: List[Tuple[str, Dict[str, Any]]] = []
    for slot, kernel_spec in enumerate(kernel_specs):
        out: List[Any] = []

        class _SyncQueue:
            def put(self, item: Any) -> None:
                out.append(item)

        label = _parallel_launch_slot_label(kernel_spec, slot, total_slots)
        _run_kernel_in_process(kernel_spec, serializable_globals, _SyncQueue())
        if not out:
            ordered.append(
                (
                    label,
                    {
                        "success": False,
                        "error": "parallel_launch: internal error (no result)",
                        "name": kernel_spec.name,
                    },
                )
            )
        else:
            ordered.append((label, out[0]))
    return ordered


# main parallel_launch function
def launch_kernels_parallel(kernel_specs: List[KernelSpec], frame_globals: Dict) -> List[Tuple[str, Dict[str, Any]]]:
    use_spawn = _kernel_specs_need_spawn_for_cuda(kernel_specs)
    cuda_inprocess = use_spawn and _cuda_parallel_launch_prefer_inprocess()

    if cuda_inprocess:
        total_slots = len(kernel_specs)
        serializable_globals_cuda: Dict[str, Any] = {}
        for spec in kernel_specs:
            if spec.name in frame_globals:
                fn = frame_globals[spec.name]
                if callable(fn):
                    serializable_globals_cuda[spec.name] = fn
        return _parallel_launch_run_in_process_slots(kernel_specs, serializable_globals_cuda, total_slots)

    if use_spawn:
        import cupy as cp

    ctx = _parallel_launch_mp_context(use_spawn=use_spawn) if use_spawn else None
    processes = []
    result_queues = []

    serializable_globals: Dict[str, Any] = {}

    for spec in kernel_specs:
        if spec.name in frame_globals:
            func = frame_globals[spec.name]
            if use_spawn and callable(func):
                source = textwrap.dedent(inspect.getsource(func))
                source_file = inspect.getfile(func)
                serializable_globals[f"_source_{spec.name}"] = source
                serializable_globals[f"_srcfile_{spec.name}"] = source_file
            else:
                pickle.dumps(func)
                serializable_globals[spec.name] = func

        if use_spawn:
            spawn_kwargs: Dict[str, Any] = {}
            for key, val in spec.kwargs.items():
                if key.startswith("_"):
                    spawn_kwargs[key] = val
                    continue
                arr_type = type(val).__module__
                if arr_type.startswith("cupy") and key in spec.mutable_cuda_arg_names:
                    spawn_kwargs[key] = _IpcArrayToken.from_cupy_array(val)
                elif arr_type.startswith("cupy"):
                    try:
                        spawn_kwargs[key] = cp.asnumpy(val)
                    except Exception:
                        spawn_kwargs[key] = val
                else:
                    spawn_kwargs[key] = val
            for key, val in spawn_kwargs.items():
                if key.startswith("_") or isinstance(val, _IpcArrayToken):
                    continue
                pickle.dumps(val)
                serializable_globals[key] = val
            spec.kwargs = spawn_kwargs
        else:
            serialized_kwargs = _serialize_arrays_to_cpu(spec.kwargs)
            for key, val in serialized_kwargs.items():
                if key in ("_is_parallel_for", "_is_call_string"):
                    continue
                pickle.dumps(val)
                serializable_globals[key] = val

            spec.kwargs = serialized_kwargs

        if use_spawn and spec.kwargs.get("_is_parallel_for") and len(spec.args) >= 2:
            from pykokkos.interface.execution_policy import ExecutionPolicy

            policy_obj = spec.args[0]
            if isinstance(policy_obj, ExecutionPolicy):
                spec.args[0] = _policy_to_recipe(policy_obj)
            if callable(spec.args[1]):
                spec.args[1] = spec.name

    total_slots = len(kernel_specs)

    if not use_spawn:
        return _parallel_launch_run_in_process_slots(kernel_specs, serializable_globals, total_slots)

    join_timeout = _parallel_launch_join_timeout_s()
    for kernel_spec in kernel_specs:
        assert ctx is not None
        queue = ctx.Queue()
        p = ctx.Process(target=_run_kernel_in_process, args=(kernel_spec, serializable_globals, queue))
        processes.append(p)
        result_queues.append(queue)
        p.start()

    ordered_results: List[Tuple[str, Dict[str, Any]]] = []
    for slot, (p, queue, kernel_spec) in enumerate(zip(processes, result_queues, kernel_specs)):
        label = _parallel_launch_slot_label(kernel_spec, slot, total_slots)
        p.join(timeout=join_timeout)

        if p.is_alive():
            p.terminate()
            p.join()
            ordered_results.append(
                (
                    label,
                    {
                        "success": False,
                        "error": f"Kernel execution timeout ({join_timeout}s)",
                        "name": kernel_spec.name,
                    },
                )
            )
        else:
            try:
                ordered_results.append((label, queue.get_nowait()))
            except:
                ordered_results.append(
                    (
                        label,
                        {
                            "success": False,
                            "error": "Failed to get result from process",
                            "name": kernel_spec.name,
                        },
                    )
                )

    return ordered_results


def _parallel_launch_array_summary_threshold() -> int:
    """Summarize array text when ``size`` exceeds this (NumPy default is ~1000, so 256-el vectors stay huge)."""
    try:
        return max(4, int(os.environ.get("PKDB_PARALLEL_LAUNCH_ARRAY_SUMMARY_THRESHOLD", "16")))
    except ValueError:
        return 16


def _format_parallel_launch_array_repr(val: Any) -> str:
    """Compact ``head … tail`` text for ndarray-like values in CLI output."""
    try:
        import numpy as np
    except ImportError:
        return str(val)
    try:
        arr = np.asarray(val)
    except Exception:
        return str(val)
    if arr.size == 0:
        return str(arr)
    th = _parallel_launch_array_summary_threshold()
    return np.array2string(
        arr,
        max_line_width=120,
        threshold=th,
        edgeitems=3,
        separator=" ",
    )


def format_parallel_results(results: List[Tuple[str, Dict[str, Any]]]) -> str:
    output = ["\n=== Parallel Kernel Launch Results ===\n"]

    for label, result in results:
        output.append(f"\n--- Kernel: {label} ---")

        if result["success"]:
            output.append(f"Status: SUCCESS")
            xfer = result.get("array_transfer")
            ipc_ks = result.get("cuda_ipc_kwarg_keys") or []
            if xfer == "mixed":
                output.append(
                    f"Arrays: CUDA IPC for [{', '.join(ipc_ks)}]; " "other array kwargs via host copy (isolated)"
                )
            elif xfer == "cuda_ipc_inplace" and ipc_ks:
                output.append("Arrays: CUDA IPC in-place (shared parent device memory) for: " + ", ".join(ipc_ks))
            elif xfer == "host_copy":
                output.append("Arrays: host copy (isolated per worker)")
            if result.get("result") is not None:
                output.append(f"Return value: {result['result']}")

            if result.get("stdout"):
                output.append(f"Stdout:\n{result['stdout']}")

            if result.get("output_arrays"):
                for arr_name, arr_val in result["output_arrays"].items():
                    output.append(f"  {arr_name} = {_format_parallel_launch_array_repr(arr_val)}")
        else:
            output.append(f"Status: FAILED")
            output.append(f"Error: {result.get('error', 'Unknown error')}")

            if result.get("stdout"):
                output.append(f"Stdout:\n{result['stdout']}")
            if result.get("stderr"):
                output.append(f"Stderr:\n{result['stderr']}")

    output.append("\n" + "=" * 40 + "\n")
    return "\n".join(output)
