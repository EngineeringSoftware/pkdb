import os
import sys
from typing import Any, Callable, Dict, Optional, Set, Tuple

from ..core import BreakpointManager

_PARALLEL_DISPATCH_MODULE = "pykokkos.interface.parallel_dispatch"
_PARALLEL_DISPATCH_FUNCS = frozenset({"parallel_for", "parallel_reduce", "parallel_scan"})


def _unwrap_parallel_policy_positional(args_tuple: Tuple[Any, ...], *, is_for: bool, view_type: type) -> Any:
    """
    Locate the policy argument in parallel_* positional args (same rules as
    ``handle_args`` in parallel_dispatch, before int -> RangePolicy conversion).
    """
    unpacked: Tuple[Any, ...] = tuple(args_tuple)
    if len(unpacked) == 2:
        return unpacked[0]
    if len(unpacked) == 3:
        if isinstance(unpacked[0], str) or unpacked[0] is None:
            return unpacked[1]
        if is_for and isinstance(unpacked[2], view_type):
            return unpacked[0]
        if isinstance(unpacked[2], (int, float)):
            return unpacked[0]
        raise TypeError(f"wrong arguments {unpacked!r}")
    if len(unpacked) == 4:
        if isinstance(unpacked[0], str) or unpacked[0] is None:
            return unpacked[1]
        raise TypeError(f"wrong arguments {unpacked!r}")
    raise ValueError(f"incorrect number of arguments {len(unpacked)}")


def _effective_execution_space_from_policy_arg(policy_obj: Any, get_default_space: Callable[[], Any]) -> Any:
    """Resolve the backend ExecutionSpace for this launch (Default -> get_default_space())."""
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore[assignment]

    if np is not None and isinstance(policy_obj, (int, np.integer)):
        return get_default_space()

    try:
        from pykokkos.interface.execution_policy import (
            ExecutionPolicy,
            MDRangePolicy,
            TeamThreadRange,
            ThreadVectorRange,
        )
        from pykokkos.interface.execution_space import ExecutionSpace, ExecutionSpaceInstance
    except Exception:
        return get_default_space()

    if isinstance(policy_obj, (TeamThreadRange, ThreadVectorRange)):
        return ExecutionSpace.Debug

    if not isinstance(policy_obj, ExecutionPolicy):
        return get_default_space()

    if isinstance(policy_obj, MDRangePolicy):
        space = policy_obj.space
        if isinstance(space, ExecutionSpace):
            if space is ExecutionSpace.Default:
                return get_default_space()
            return space
        return get_default_space()

    inst = getattr(policy_obj, "space", None)
    if isinstance(inst, ExecutionSpaceInstance):
        s = inst.space
        if s is ExecutionSpace.Default:
            return get_default_space()
        return s

    return get_default_space()


def _mutate_policy_to_debug_space(policy_obj: Any, target_debug: Any) -> None:
    """Align in-memory policy objects with Debug* after set_default_space (explicit Cuda/HIP/OpenMP)."""
    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore[assignment]

    if np is not None and isinstance(policy_obj, (int, np.integer)):
        return

    try:
        from pykokkos.interface.execution_policy import (
            ExecutionPolicy,
            MDRangePolicy,
            TeamThreadRange,
            ThreadVectorRange,
        )
        from pykokkos.interface.execution_space import ExecutionSpace, ExecutionSpaceInstance
    except Exception:
        return

    if isinstance(policy_obj, (TeamThreadRange, ThreadVectorRange)):
        return
    if not isinstance(policy_obj, ExecutionPolicy):
        return

    prod = (ExecutionSpace.Cuda, ExecutionSpace.HIP, ExecutionSpace.OpenMP)

    if isinstance(policy_obj, MDRangePolicy):
        s = policy_obj.space
        if isinstance(s, ExecutionSpace) and s in prod:
            policy_obj.space = target_debug
        return

    inst = getattr(policy_obj, "space", None)
    if isinstance(inst, ExecutionSpaceInstance) and inst.space in prod:
        inst.space = target_debug


def jit_promote_execution_space_for_parallel_invocation(callee_frame: Any) -> None:
    """
    On entry to parallel_for / parallel_reduce / parallel_scan: read ``args`` from the
    callee frame, derive Cuda/HIP/OpenMP from the policy or from get_default_space() for
    integer bounds, then set_default_space(Debug*) and patch explicit policies in place.
    """
    if callee_frame.f_globals.get("__name__") != _PARALLEL_DISPATCH_MODULE:
        return
    if callee_frame.f_code.co_name not in _PARALLEL_DISPATCH_FUNCS:
        return
    pos = callee_frame.f_locals.get("args")
    if not isinstance(pos, tuple):
        return

    try:
        import pykokkos.kokkos_manager as km
        from pykokkos.interface.execution_space import ExecutionSpace
        from pykokkos.interface.views import ViewType
    except Exception:
        return

    try:
        policy_obj = _unwrap_parallel_policy_positional(pos, is_for=True, view_type=ViewType)
    except (TypeError, ValueError):
        return

    get_d = km.get_default_space
    effective = _effective_execution_space_from_policy_arg(policy_obj, get_d)

    promoted = {
        ExecutionSpace.Cuda: ExecutionSpace.DebugCuda,
        ExecutionSpace.HIP: ExecutionSpace.DebugHIP,
        ExecutionSpace.OpenMP: ExecutionSpace.DebugOpenMP,
    }.get(effective)
    if promoted is None:
        return

    km.set_default_space(promoted)
    _mutate_policy_to_debug_space(policy_obj, promoted)


def accelerator_info_from_parallel_invocation(callee_frame: Any) -> Optional[Dict[str, str]]:
    """
    Derive accelerator debugger choice from the live ``parallel_*`` invocation.

    Returns ``{"type": "cuda", "name": "CUDA"}``, ``{"type": "hip", "name": "HIP"}``,
    or ``None`` for non-accelerator policies.
    """
    if callee_frame.f_globals.get("__name__") != _PARALLEL_DISPATCH_MODULE:
        return None
    if callee_frame.f_code.co_name not in _PARALLEL_DISPATCH_FUNCS:
        return None

    pos = callee_frame.f_locals.get("args")
    if not isinstance(pos, tuple):
        return None

    try:
        import pykokkos.kokkos_manager as km
        from pykokkos.interface.execution_space import ExecutionSpace
        from pykokkos.interface.views import ViewType
    except Exception:
        return None

    try:
        policy_obj = _unwrap_parallel_policy_positional(pos, is_for=True, view_type=ViewType)
    except (TypeError, ValueError):
        return None

    effective = _effective_execution_space_from_policy_arg(policy_obj, km.get_default_space)
    if effective in (ExecutionSpace.Cuda, ExecutionSpace.DebugCuda):
        return {"type": "cuda", "name": "CUDA"}
    if effective in (ExecutionSpace.HIP, ExecutionSpace.DebugHIP):
        return {"type": "hip", "name": "HIP"}
    return None


from .debugger import NativeHandoffConfig, PyKokkosDebugger


def show_usage():
    print("Usage: pk-db <script.py> [args...] OR python -m pkdb <script.py> [args...]")
    print("\nOptions:")
    print("  --profile         - Enable cProfile profiling (generates .prof file)")
    print("\nAt the (pkdb) prompt: use pdb commands (b, c, n, …), set verbose on, etc.")


def validate_script(script_path):
    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found")
        sys.exit(1)


def run_script_with_debugger(
    script_path: str,
    pdb_breakpoints: Optional[Dict[str, Set[int]]] = None,
    native_handoff=None,
):
    """Run ``script_path`` under ``PyKokkosDebugger``; stop at the first line of the script."""
    pdb_breakpoints = pdb_breakpoints or {}
    sys.argv = sys.argv[1:]
    script_path = os.path.abspath(script_path)

    debugger = PyKokkosDebugger(
        trace_mode=True,
        native_handoff=native_handoff,
    )
    for file_path, line_nums in pdb_breakpoints.items():
        abs_file = os.path.abspath(file_path)
        for lineno in line_nums:
            debugger.set_break(abs_file, lineno)

    # Ensure the entry script is always a key in self.breaks so bdb.break_anywhere()
    # returns True for frames in that file, keeping tracing alive through main()'s
    # call chain even when the user only set breakpoints in imported modules.
    main_canon = debugger.canonic(script_path)
    debugger.breaks.setdefault(main_canon, [])

    import __main__
    import io

    __main__.__dict__.clear()
    __main__.__dict__.update(
        {
            "__name__": "__main__",
            "__file__": script_path,
            "__builtins__": __builtins__,
        }
    )
    script_dir = os.path.dirname(script_path)
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Compile the script with the real path as co_filename. Do not use debugger.run(string):
    # Bdb.runctx compiles str sources as "<string>", which makes the first stop <string>(1) and
    # breaks ``b <lineno>`` / file-relative commands. A pre-built code object preserves the file.
    script_canon = debugger.canonic(script_path)
    with io.open_code(script_path) as fp:
        code_obj = compile(fp.read(), script_canon, "exec")
    try:
        debugger.runctx(code_obj, __main__.__dict__, __main__.__dict__)
    finally:
        # End every attach-helper subprocess kept alive across this run (see
        # NativeDebuggerSession), however the script/debugger session ended.
        debugger.terminate_native_sessions()


def _build_native_handoff(
    script_path: str,
    file_breakpoints: Optional[Dict[str, Set[int]]] = None,
):
    """Prepare ``BreakpointManager`` state (runtime mapping arrives during parallel dispatch)."""
    file_breakpoints = file_breakpoints or {}

    breakpoint_manager = BreakpointManager(
        script_path=script_path,
        script_args=sys.argv[2:],
    )

    for file_path, line_numbers in file_breakpoints.items():
        for line_num in sorted(line_numbers):
            if breakpoint_manager.get_cpp_locations(line_num, source_file=file_path):
                breakpoint_manager.add_breakpoint(line_num, file_path=file_path)

    return NativeHandoffConfig(
        script_path=script_path,
        breakpoint_manager=breakpoint_manager,
        accelerator_info=None,
    )


def configure_runtime_flags():
    """Parse pkdb runtime flags from sys.argv and apply side effects."""
    enable_profiling = "--profile" in sys.argv
    if enable_profiling:
        sys.argv.remove("--profile")
        from ..core import get_debug_properties

        get_debug_properties().profile = True


def get_script_path_from_argv():
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)

    script_path = os.path.abspath(sys.argv[1])
    sys.argv[1] = script_path
    return script_path


def run_setup_mode(script_path):
    """Run script under ``(pkdb)`` with runtime-driven handoff (no startup static analysis)."""
    native_handoff = _build_native_handoff(script_path, file_breakpoints={})
    run_script_with_debugger(
        script_path,
        pdb_breakpoints={},
        native_handoff=native_handoff,
    )
    sys.exit(0)
