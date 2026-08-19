"""
PyKokkos debugger class that extends pdb for workunit debugging
"""

import inspect
import json
import linecache
import os
import pdb
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from pkdb.core.debug_properties import verbose_out

from ..analyzers.runtime_mapping import OverlapCppMapping, RuntimeBundleMapping
from ..controllers.gdb_controller import (
    launch_attached_debugger,
    push_breakpoint_update,
    push_control_detach,
    push_control_reattach,
)
from ..evaluators.ipc_utils import IPC_JSON_PATH, PKDB_WORKUNIT_PARAM_MAP_KEY
from ..evaluators.workunit_arg_mapping import build_workunit_param_to_caller_map_from_file
from .helper import (
    accelerator_info_from_parallel_invocation,
    jit_promote_execution_space_for_parallel_invocation,
)
from .pdb_commands import PdbCommandsMixin

# pdb imports bdb; pull Bdb and BdbQuit without a redundant "import bdb"
_bdb = sys.modules["bdb"]
BdbQuit = _bdb.BdbQuit
Bdb = _bdb.Bdb


_PARALLEL_DISPATCH_ENTRY_CODES: Optional[FrozenSet[CodeType]] = None


def _pykokkos_parallel_dispatch_entry_codes() -> FrozenSet[CodeType]:
    """Code objects for ``parallel_for`` / ``parallel_reduce`` / ``parallel_scan`` (identity, not text)."""
    global _PARALLEL_DISPATCH_ENTRY_CODES
    if _PARALLEL_DISPATCH_ENTRY_CODES is not None:
        return _PARALLEL_DISPATCH_ENTRY_CODES
    try:
        from pykokkos.interface import parallel_dispatch as _pd

        codes: list[CodeType] = []
        for name in ("parallel_for", "parallel_reduce", "parallel_scan"):
            fn = getattr(_pd, name, None)
            oc = getattr(fn, "__code__", None)
            if isinstance(oc, CodeType):
                codes.append(oc)
        _PARALLEL_DISPATCH_ENTRY_CODES = frozenset(codes)
    except Exception:
        _PARALLEL_DISPATCH_ENTRY_CODES = frozenset()
    return _PARALLEL_DISPATCH_ENTRY_CODES


def _is_pykokkos_source_path(canonic_file: str) -> bool:
    """True if the canonical path lives under a ``pykokkos`` package directory."""
    try:
        return "pykokkos" in Path(canonic_file).parts
    except (OSError, ValueError):
        return "pykokkos" in canonic_file.replace("\\", "/")


@dataclass
class NativeDebuggerSession:
    """
    One attach-helper subprocess for one debugger kind ("cpu"/"cuda"/"hip").

    Launched at most once per program run, then kept alive for the rest of
    the run: switching to a different kind detaches this session (ptrace
    released, helper process and its gdb left running) rather than killing
    it, so switching back later is a cheap reattach, not a fresh subprocess.
    Only terminated at debug-session end (``PyKokkosDebugger.terminate_native_sessions``).
    """

    kind: str
    process: subprocess.Popen
    # Whether this session currently holds ptrace on the target (vs kept
    # alive but detached, waiting for a future reattach). At most one
    # session across the whole run is ever attached at a time.
    attached: bool = True
    update_seq: int = 0
    last_payload: Dict[str, Any] = field(default_factory=dict)

    def alive(self) -> bool:
        return self.process.poll() is None

    def terminate(self) -> None:
        """End the attach-helper session for good (SIGINT -> clean detach+quit)."""
        proc = self.process
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGINT)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
        proc.wait()
        ack_fd = getattr(proc, "pkdb_bp_ack_read_fd", None)
        if ack_fd is not None:
            os.close(ack_fd)
            setattr(proc, "pkdb_bp_ack_read_fd", None)


# Custom debugger that extends pdb.Pdb for PyKokkos-specific debugging
# features for workunits
@dataclass
class NativeHandoffConfig:
    script_path: str
    breakpoint_manager: Optional[object] = None
    accelerator_info: Optional[dict] = None


class PyKokkosDebugger(PdbCommandsMixin, pdb.Pdb):

    def __init__(
        self,
        trace_mode=True,
        native_handoff: Optional[NativeHandoffConfig] = None,
    ):
        super().__init__()
        self.trace_mode = trace_mode
        # Breakpoints that the user explicitly requested (via pkdb setup or pdb commands)
        self.user_breakpoints: Set[Tuple[str, int]] = set()
        # debugger that we call when we enter kernel (gdb / accelerator-gdb)
        self.native_handoff = native_handoff
        self._wait_for_mainpyfile = False
        self.mainpyfile = None
        self._in_native_handoff = False
        # One NativeDebuggerSession per debugger kind ever launched this run,
        # keyed by kind ("cpu"/"cuda"/"hip"); see NativeDebuggerSession docstring.
        # At most one is ever ``.attached`` at a time - that one is found by
        # iterating this dict, not tracked separately.
        self._native_sessions: Dict[str, NativeDebuggerSession] = {}
        # Union of breakpoint LOCs handed off so far; keeps earlier kernels'
        self.prompt = "(pkdb) "
        self._pk_enabled_python_union: Set[Tuple[str, int]] = set()
        self._pk_parallel_launch_trace: Optional[Dict[str, Any]] = None  # trace for frame data
        # (canonical runtime.py, lineno) from ``inspect.getsourcelines(Runtime.run_workunit / precompile_workunit)``.
        self._pk_runtime_bp_metadata: Optional[Tuple[str, int]] = None
        self._pk_runtime_bp_execute: Optional[Tuple[str, int]] = None
        # Internal bdb breakpoints set at anchor lines; intercepted in user_line without showing a prompt.
        self._pk_internal_anchor_locs: Set[Tuple[str, int]] = set()
        self.reset()
        self._init_pykokkos_runtime_anchor_locations()

    def canonic(self, filename: str) -> str:
        """Like pdb.Pdb.canonic, but resolve symlinks so bdb breakpoint keys match co_filename.

        Interactive breakpoints often use Path.resolve(); import-time co_filename may differ.
        If keys diverge, break_here never fires even though tracing runs (pure Python issue,
        unrelated to parallel / workunit breakpoints).
        """
        if filename == "<" + filename[1:-1] + ">":
            return filename
        canonic = self.fncache.get(filename)
        if canonic:
            return canonic
        try:
            canonic = os.path.normcase(os.path.realpath(os.path.abspath(filename)))
        except OSError:
            canonic = os.path.normcase(os.path.abspath(filename))
        self.fncache[filename] = canonic
        return canonic

    def _init_pykokkos_runtime_anchor_locations(self) -> None:
        """Anchor lines in ``runtime.py`` from ``Runtime`` methods (no hard-coded paths)."""
        self._pk_runtime_bp_metadata = None
        self._pk_runtime_bp_execute = None
        self._pk_internal_anchor_locs = set()
        try:
            from pykokkos.core.runtime import Runtime
        except Exception:
            return

        def line_matching(method: Callable[..., Any], predicate: Callable[[str], bool]) -> Optional[int]:
            try:
                lines, lnum = inspect.getsourcelines(method)
            except (OSError, TypeError):
                return None
            for i, line in enumerate(lines):
                if predicate(line):
                    return lnum + i
            return None

        src_path = inspect.getsourcefile(Runtime.run_workunit)
        if not src_path:
            return
        ln_meta = line_matching(Runtime.run_workunit, lambda s: "return self.execute_workunit(" in s)
        ln_exec = line_matching(Runtime.execute_workunit, lambda s: "return self.execute(" in s)
        if ln_meta is None or ln_exec is None:
            return
        cfile = self.canonic(src_path)
        self._pk_runtime_bp_metadata = (cfile, ln_meta)
        self._pk_runtime_bp_execute = (cfile, ln_exec)
        # Register internal bdb breakpoints so bdb's native mechanism fires at these lines
        # instead of us checking every line event in parallel_stop("runtime_line").
        for loc in (self._pk_runtime_bp_metadata, self._pk_runtime_bp_execute):
            self._pk_internal_anchor_locs.add(loc)
            Bdb.set_break(self, loc[0], loc[1])

    def _parallel_runtime_auto_hook_ok(self, loc: Tuple[str, int]) -> bool:
        """Run non-interactive parallel-launch hooks in pykokkos unless the user broke on this line."""
        if loc in self.user_breakpoints:
            return False
        return _is_pykokkos_source_path(loc[0])

    def break_anywhere(self, frame):
        """
        Stock per-file check, broadened only for kernel hotswap's benefit.
        """
        if super().break_anywhere(frame):
            return True
        return bool(self._kernel_swaps or self._kernel_swaps_loc)

    # Override pdb.Pdb.set_break so that any breakpoints requested interactively
    # (or via pkdb setup for workunit lines) are tracked as "user" breakpoints.
    def set_break(self, filename, lineno, temporary=False, cond=None, funcname=None):
        filename = self.canonic(filename)
        self.user_breakpoints.add((filename, lineno))
        return super().set_break(filename, lineno, temporary, cond, funcname)

    def clear_break(self, filename, lineno):
        """Clear bdb breakpoint and, if applicable, workunit / BreakpointManager state."""
        filename = self.canonic(filename)
        if (filename, lineno) in self._pk_internal_anchor_locs:
            return None
        self.user_breakpoints.discard((filename, lineno))
        nh = self.native_handoff
        cpp_here = bool(
            nh
            and nh.breakpoint_manager is not None
            and nh.breakpoint_manager.get_cpp_locations(lineno, source_file=filename)
        )
        if cpp_here and nh is not None:
            bm = nh.breakpoint_manager
            if bm is not None:
                bm.remove_breakpoint(lineno)
        err = Bdb.clear_break(self, filename, lineno)
        if cpp_here and err is not None:
            return None
        return err

    def setup_mainpyfile(self, filename):
        """Like pdb._runscript: skip stops until we reach the main script."""
        self._wait_for_mainpyfile = True
        self.mainpyfile = self.canonic(filename)

    def trace_dispatch(self, frame, event, arg):
        if event == "call":
            self.parallel_stop("jit_call", frame)
        return super().trace_dispatch(frame, event, arg)

    def dispatch_line(self, frame):
        if self._kernel_swaps or self._kernel_swaps_loc:
            if not (self.stop_here(frame) or self.break_here(frame)):
                cfn = self.canonic(frame.f_code.co_filename)
                ln = frame.f_lineno
                hline = linecache.getline(cfn, ln)
                if "parallel_" in hline and self._hotswap_kernels_call(frame, cfn, ln):
                    return self.trace_dispatch
        return super().dispatch_line(frame)

    def parallel_stop(self, stage: str, frame) -> None:
        """Hook for ``parallel_*`` JIT promotion (call-event only)."""
        if stage == "jit_call":
            parallel_codes = _pykokkos_parallel_dispatch_entry_codes()
            if frame.f_code in parallel_codes:
                jit_promote_execution_space_for_parallel_invocation(frame)

    def _pk_find_caller_frame(self, frame):
        """Walk up from an anchor frame to the first non-pykokkos frame (the parallel_* call site)."""
        f = frame.f_back
        while f is not None:
            if not _is_pykokkos_source_path(self.canonic(f.f_code.co_filename)):
                return f
            f = f.f_back
        return None

    def _pk_find_dispatch_frame(self, frame):
        """Walk up from an anchor frame to the parallel_for/reduce/scan frame."""
        parallel_codes = _pykokkos_parallel_dispatch_entry_codes()
        f = frame.f_back
        while f is not None:
            if f.f_code in parallel_codes:
                return f
            f = f.f_back
        return None

    @staticmethod
    def _primary_workunit_callable(workunit: Any) -> Any:
        """``run_workunit`` passes a callable or a list of callables; use the first for source/LOC."""
        if workunit is None:
            return None
        if isinstance(workunit, list):
            return workunit[0] if workunit else None
        return workunit

    def _find_workunit_overlap_LOCs(self, wu_callable: Any) -> List[Tuple[str, int]]:
        """Return user breakpoints that fall in the workunit source range as (py_path, lineno) pairs."""
        try:
            source_lines, starting_line = inspect.getsourcelines(wu_callable)
        except (OSError, TypeError):
            return []
        wu_file = inspect.getsourcefile(wu_callable)
        if not wu_file:
            return []
        end_line = starting_line + len(source_lines) - 1
        wu_canon = self.canonic(wu_file)
        return sorted(
            (fp, ln) for (fp, ln) in self.user_breakpoints if fp == wu_canon and starting_line <= ln <= end_line
        )

    @staticmethod
    def _count_enabled_cpp_targets(rows: List[OverlapCppMapping], enabled: Set[Tuple[str, int]]) -> int:
        return sum(len(cpp_locs) for py_fp, py_ln, cpp_locs in rows if (py_fp, py_ln) in enabled)

    def _pk_parallel_trace_capture_metadata(self, frame, loc: Tuple[str, int]) -> None:
        tr = self._pk_parallel_launch_trace
        mb = self._pk_runtime_bp_metadata
        if not tr or tr.get("completed") or mb is None or loc != mb:
            return
        metadata = frame.f_locals.get("metadata")
        if metadata is None:
            return
        name = getattr(metadata, "name", None)
        path = getattr(metadata, "path", None)
        if name is None or path is None:
            return
        tr["meta_name"] = name
        tr["meta_path"] = path

        wu_callable = self._primary_workunit_callable(frame.f_locals.get("workunit"))
        if wu_callable is not None:
            overlap = self._find_workunit_overlap_LOCs(wu_callable)
            if overlap:
                tr["workunit_breakpoint_overlap"] = overlap

    def _pk_parallel_trace_report_execute(self, frame, loc: Tuple[str, int]) -> None:
        """At the execute anchor (post-JIT, pre-launch): load bundle mapping and hand off."""
        tr = self._pk_parallel_launch_trace
        pb = self._pk_runtime_bp_execute
        module_setup = frame.f_locals.get("module_setup")
        if (
            not tr
            or tr.get("completed")
            or pb is None
            or loc != pb
            or tr.get("meta_name") is None
            or module_setup is None
        ):
            return
        meta_path = tr.get("meta_path")
        real_src = os.path.realpath(meta_path) if meta_path else ""
        output_dir = getattr(module_setup, "output_dir", None)
        overlap = tr.get("workunit_breakpoint_overlap") or []
        overlap_set = set(overlap)
        rows_overlap: List[OverlapCppMapping] = []
        rows_bundle: List[OverlapCppMapping] = []

        if output_dir is not None and real_src and Path(output_dir).is_dir():
            rows_overlap, rows_bundle = RuntimeBundleMapping(real_src).resolve_overlap(
                sorted(self.user_breakpoints), Path(output_dir)
            )
        handoff = self.native_handoff
        # Make complete mapping that includes both Python and C++ LOCs.
        if handoff is not None and handoff.breakpoint_manager is not None and rows_bundle:
            runtime_entries = []
            for py_fp, py_ln, cpp_locs in rows_bundle:
                for cpp_fp, cpp_ln in cpp_locs:
                    runtime_entries.append(
                        {
                            "cpp_file": str(cpp_fp),
                            "cpp_line": int(cpp_ln),
                            "py_file": str(py_fp),
                            "py_line": int(py_ln),
                        }
                    )
            handoff.breakpoint_manager.apply_handoff_cpp_to_python(runtime_entries)
        self._pk_enabled_python_union |= overlap_set
        enabled_python = sorted(self._pk_enabled_python_union & self.user_breakpoints)
        if handoff is not None and handoff.breakpoint_manager is not None:
            payload = handoff.breakpoint_manager.build_complete_handoff_payload(enabled_python)
            self._launch_native_handoff(handoff_payload=payload)
        tr.pop("caller_frame", None)
        tr["completed"] = True

    def user_call(self, frame, argument_list):
        """Like pdb: skip stops until we reach the main script."""
        if self._wait_for_mainpyfile:
            return
        super().user_call(frame, argument_list)

    # Stop or break at the line
    def user_line(self, frame):
        lineno = frame.f_lineno
        frame_file = self.canonic(frame.f_code.co_filename)
        loc = (frame_file, lineno)

        # Internal anchor breakpoints: run metadata/execute hooks and return without a prompt.
        if loc in self._pk_internal_anchor_locs:
            if loc == self._pk_runtime_bp_metadata:
                # Build launch trace from call stack (replaces dispatch_enter hook).
                caller = self._pk_find_caller_frame(frame)
                if self.native_handoff is not None:
                    disp = self._pk_find_dispatch_frame(frame)
                    if disp is not None:
                        self.native_handoff.accelerator_info = accelerator_info_from_parallel_invocation(disp)
                self._pk_parallel_launch_trace = {
                    "caller_file": self.canonic(caller.f_code.co_filename) if caller else "",
                    "caller_lineno": caller.f_lineno if caller else 0,
                    "caller_frame": caller,
                    "completed": False,
                }
                if caller is not None:
                    self._collect_parallel_launch_ipc(caller)
            tr = self._pk_parallel_launch_trace
            if tr is not None and not tr.get("completed"):
                if self._parallel_runtime_auto_hook_ok(loc):
                    self._pk_parallel_trace_capture_metadata(frame, loc)
                    self._pk_parallel_trace_report_execute(frame, loc)
            return

        verbose_out("")

        # If we have kernel swap rules, try to handle a parallel_* line (cheap
        # ``parallel_`` prefilter — see _hotswap_kernels_call / parsed-file cache).
        if (self._kernel_swaps or self._kernel_swaps_loc) and "parallel_" in linecache.getline(frame_file, lineno):
            if self._hotswap_kernels_call(frame, frame_file, lineno):
                return

        # pdb's user_line guards on _wait_for_mainpyfile and skips any file that is not
        # mainpyfile (main.py). Clear the flag whenever we're genuinely about to stop so
        # breakpoints in imported modules (e.g. examinimd.py) show the prompt correctly.
        if self._wait_for_mainpyfile and (self.break_here(frame) or self.stop_here(frame)):
            self._wait_for_mainpyfile = False

        super().user_line(frame)

    def _collect_parallel_launch_ipc(self, frame):
        """
        At `parallel_*` call collect all CuPy IPC tokens, write to internal file
        and read this file from native debugger
        """
        try:
            import cupy
            from cupy.cuda import runtime as cupy_runtime
        except Exception:
            return

        locals_dict = frame.f_locals or {}
        ipc_entries: Dict[str, Any] = {}

        for name, value in locals_dict.items():
            if isinstance(value, getattr(cupy, "ndarray", ())):
                devptr = int(value.data.ptr)
                handle = cupy_runtime.ipcGetMemHandle(devptr)
                if isinstance(handle, (bytes, bytearray)):
                    handle_repr = handle.hex()
                else:
                    handle_repr = str(handle)

                ipc_entries[name] = {
                    "ipc_handle": handle_repr,
                    "devptr": devptr,
                    "shape": tuple(int(s) for s in value.shape),
                    "dtype": str(value.dtype),
                }

        if not ipc_entries:
            return

        os.makedirs(os.path.dirname(IPC_JSON_PATH), exist_ok=True)

        existing: Dict[str, Any] = {}
        if os.path.exists(IPC_JSON_PATH):
            with open(IPC_JSON_PATH, "r") as f:
                existing = json.load(f)

        existing.update(ipc_entries)

        try:
            frame_file = self.canonic(frame.f_code.co_filename)
            abs_src = str(Path(frame_file).resolve())
        except OSError:
            abs_src = os.path.abspath(frame.f_code.co_filename)

        existing[PKDB_WORKUNIT_PARAM_MAP_KEY] = build_workunit_param_to_caller_map_from_file(abs_src, frame.f_lineno)

        with open(IPC_JSON_PATH, "w") as f:
            json.dump(existing, f, indent=2, sort_keys=True)

    def _launch_debugger_kind_and_info(self) -> Tuple[str, dict]:
        info = self.native_handoff.accelerator_info if self.native_handoff is not None else None
        if not info:
            return "cpu", {}
        return info["type"], info

    def _launch_native_handoff(self, handoff_payload: Dict[str, Any]) -> None:
        """
        Route to the debugger kind this launch needs, reusing a previously
        launched helper for that kind if one exists (see NativeDebuggerSession).
        """
        if self.native_handoff is None or self.native_handoff.breakpoint_manager is None:
            return
        if not handoff_payload:
            return

        launch_kind, accelerator_info = self._launch_debugger_kind_and_info()
        sessions = self._native_sessions
        active = next((s for s in sessions.values() if s.attached and s.alive()), None)

        if active is not None and active.kind == launch_kind:
            # Already attached with the debugger this launch needs: just update.
            if handoff_payload == active.last_payload:
                return
            active.update_seq += 1
            push_breakpoint_update(active.process, active.update_seq, handoff_payload)
            active.last_payload = handoff_payload
            return

        if active is not None:
            # Wrong kind for this launch: release it (kept alive, not killed)
            # so the kind this launch needs can attach below.
            active.update_seq += 1
            push_control_detach(active.process, active.update_seq)
            active.attached = False

        existing = sessions.get(launch_kind)
        if existing is not None and existing.alive():
            # Launched this kind before: reattach the same helper - no new
            # subprocess, no gdb restart - instead of spawning again.
            existing.update_seq += 1
            push_control_reattach(existing.process, existing.update_seq, handoff_payload)
            existing.attached = True
            existing.last_payload = handoff_payload
            return

        self._in_native_handoff = True
        try:
            proc = launch_attached_debugger(
                pid=os.getpid(),
                script_path=self.native_handoff.script_path,
                accelerator_info=accelerator_info or None,
                handoff_payload=handoff_payload,
            )
            sessions[launch_kind] = NativeDebuggerSession(kind=launch_kind, process=proc, last_payload=handoff_payload)
        finally:
            self._in_native_handoff = False

    def terminate_native_sessions(self) -> None:
        """End every attach-helper subprocess launched this run. Called once, at debug-session end."""
        for session in self._native_sessions.values():
            session.terminate()
        self._native_sessions.clear()
