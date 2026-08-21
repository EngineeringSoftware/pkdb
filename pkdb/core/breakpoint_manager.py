"""
Breakpoint manager for PyKokkos debugger
Manages breakpoints dynamically without recompilation
"""

import os
import re
from pathlib import Path
from typing import Any, Set, Optional, Callable, Dict, List, Tuple
from ..core.debug_properties import verbose_out

# Compiled once: load_line_mapping runs on every kernel launch.
_MARKER_PATTERN = re.compile(r"//\s*PY_LINE_MARKER_(.+)")
_MARKER_CACHE: Dict[str, Tuple[int, int, List[Tuple[int, str, int]]]] = {}
_RESOLVED_PATHS: Dict[str, str] = {}


def _resolved_str(path) -> str:
    """`str(Path(path).resolve())`, memoized; falls back to the path as given if it cannot resolve."""
    key = str(path)
    cached = _RESOLVED_PATHS.get(key)
    if cached is not None:
        return cached
    try:
        resolved = str(Path(key).resolve())
    except (OSError, ValueError):
        resolved = key
    _RESOLVED_PATHS[key] = resolved
    return resolved


class BreakpointManager:
    """
    Manages breakpoints for PyKokkos workunits.
    Supports both CUDA and OpenMP execution spaces.

    Tracks three sets of breakpoints for step/continue workflow:
    - breakpoints: currently active breakpoints in GDB
    - continue_active: breakpoints to restore when 'continue' is called
    - all_locs: all breakable locations in workunits (for step mode)
    """

    def __init__(
        self,
        gdb_command_sender: Optional[Callable] = None,
        script_path: Optional[str] = None,
        script_args: Optional[List[str]] = None,
    ):
        self.continue_active: Set[int] = set()
        self.breakpoints: Set[int] = set()
        self.all_locs: Set[int] = set()
        self._step_mode: bool = False
        self._gdb_command_sender = gdb_command_sender
        self._script_path = script_path
        self._script_args = list(script_args or [])

        # Core mapping: (source_file, python_line) -> list of (functor_path, cpp_line)
        # This is the primary data structure - keyed by (file, line) to avoid ambiguity
        self._python_to_cpp: Dict[Tuple[str, int], List[Tuple[Path, int]]] = {}

        # Reverse mapping for each functor: functor_path -> {cpp_line: (source_file, python_line)}
        self._functor_cpp_to_python: Dict[Path, Dict[int, Tuple[str, int]]] = {}

        # Kernel name (e.g. "work", "init") -> resolved path to that kernel's functor.hpp
        self._kernel_to_path: Dict[str, Path] = {}

        # Deprecated: static preload of every functor.hpp under pk_cpp at init (see commented
        # implementations of _precompile_script / _load_line_mappings_from_script below).
        # if script_path:
        #     self._load_line_mappings_from_script(script_path)

    def set_gdb_command_sender(self, sender: Callable):
        """Set the GDB command sender function for updating breakpoints in GDB"""
        self._gdb_command_sender = sender

    def add_breakpoint(self, lineno: int, file_path: Optional[str] = None) -> bool:
        """
        Add a breakpoint at the given line number (to gdb and local state).

        :param lineno: Line number in the Python source file
        :param file_path: Optional source file path to disambiguate when multiple files have the same line number
        :returns: True if breakpoint was added, False if it already exists
        """
        verbose_out(f"[BP_MANAGER] add_breakpoint called for line {lineno}, file={file_path}")
        if lineno in self.continue_active:
            verbose_out(f"[BP_MANAGER] Breakpoint at line {lineno} already exists")
            return False

        self.continue_active.add(lineno)
        self._set_cpp_breakpoint(lineno, file_path=file_path)
        if not self._step_mode:
            self.breakpoints.add(lineno)
        return True

    def remove_breakpoint(self, lineno: int) -> bool:
        """
        Remove a breakpoint at the given line number (from gdb and local state).

        :param lineno: Line number in the Python source file
        :returns: True if breakpoint was removed, False if it didn't exist
        """
        if lineno not in self.continue_active:
            return False

        self.continue_active.discard(lineno)
        if not self._step_mode:
            self.breakpoints.discard(lineno)
            if self._gdb_command_sender is not None:
                self._remove_cpp_breakpoint(lineno)
        return True

    def _remove_cpp_breakpoint(self, python_line: int):
        """
        Remove C++ breakpoints from GDB for a Python line number.
        Clears breakpoints in every kernel (functor file) that contains this line.

        :param python_line: Python source line number
        """
        if self._gdb_command_sender is None:
            return

        for functor_path, cpp_line in self.get_cpp_locations(python_line):
            delete_cmd = f'interpreter-exec console "clear {functor_path}:{cpp_line}"'
            self._gdb_command_sender(delete_cmd)

    def clear_breakpoints(self):
        """Clear all breakpoints"""
        self.continue_active.clear()
        if not self._step_mode:
            self.breakpoints.clear()
            self._update_cpp_breakpoints()

    def get_breakpoints(self) -> Set[int]:
        """Get all currently active breakpoints (in step mode returns all_locs, otherwise continue_active)"""
        return self.breakpoints.copy()

    def has_breakpoint(self, lineno: int) -> bool:
        """Check if a breakpoint exists at the given line number (checks continue_active, the source of truth)"""
        return lineno in self.continue_active

    def set_all_locs(self, locs: Set[int]):
        """
        Set all breakable locations in workunits.
        Called once during initialization with workunit line info.

        :param locs: Set of all line numbers that can be stepped through
        """
        self.all_locs = locs.copy()

    def get_all_locs(self) -> Set[int]:
        """Get all breakable locations"""
        return self.all_locs.copy()

    def is_step_mode(self) -> bool:
        """Check if currently in step mode"""
        return self._step_mode

    def enter_step_mode(self):
        """
        Enter step mode:
        1. Save current breakpoints to continue_active
        2. Activate all LOCs as breakpoints
        3. Set step mode flag
        """
        if not self._step_mode:
            self.continue_active = self.breakpoints.copy()
            self._step_mode = True

        self.breakpoints = self.all_locs.copy()
        self._update_cpp_breakpoints()

    def exit_step_mode(self):
        """
        Exit step mode and restore continue_active breakpoints.
        Called when 'continue' command is issued.
        """
        if self._step_mode:
            self._step_mode = False
            self.breakpoints = self.continue_active.copy()
            self._update_cpp_breakpoints()

    def get_continue_active(self) -> Set[int]:
        """Get the breakpoints that will be active after continue"""
        return self.continue_active.copy()

    def set_continue_active(self, breakpoints: Set[int]):
        """Set the breakpoints to restore on continue"""
        self.continue_active = breakpoints.copy()

    def compare_with_expected(self) -> tuple[bool, Set[int], Set[int]]:
        """
        Compare currently active breakpoints with expected set.
        In step mode, compare with all_locs.
        In continue mode, compare with continue_active.

        :returns: (matches, missing, extra) tuple
        """
        expected = self.all_locs if self._step_mode else self.continue_active
        missing = expected - self.breakpoints
        extra = self.breakpoints - expected
        matches = len(missing) == 0 and len(extra) == 0
        return matches, missing, extra

    @staticmethod
    def _parse_py_line_marker_payload(marker_body: str) -> Optional[Tuple[str, int]]:
        """
        Parse payload after PY_LINE_MARKER_: <file_path>_<python_lineno>_<hash>
        """
        marker_body = marker_body.strip()
        if not marker_body:
            return None
        seen = 0
        file_end = -1
        for i in range(len(marker_body) - 1, -1, -1):
            if marker_body[i] == "_":
                seen += 1
                if seen == 2:
                    file_end = i
                    break
        if file_end < 0:
            return None
        file_path = marker_body[:file_end]
        tail = marker_body[file_end + 1 :]
        tail_parts = tail.rsplit("_", 1)
        if len(tail_parts) != 2:
            return None
        lineno_str, _hash = tail_parts
        try:
            python_line = int(lineno_str)
        except ValueError:
            return None
        return (file_path, python_line)

    @classmethod
    def _load_markers(cls, path: Path) -> List[Tuple[int, str, int]]:
        """
        Parsed `(cpp_line, python_file, python_line)` markers for *path*, cached across
        BreakpointManager instances and invalidated when the file's mtime or size changes.
        """
        try:
            st = path.stat()
        except OSError:
            return []

        key = str(path)
        cached = _MARKER_CACHE.get(key)
        if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]

        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except OSError:
            return []

        markers: List[Tuple[int, str, int]] = []
        for cpp_line_num, line in enumerate(lines, start=1):
            match = _MARKER_PATTERN.search(line)
            if not match:
                continue
            parsed = cls._parse_py_line_marker_payload(match.group(1))
            if parsed is None:
                continue
            file_path, python_line = parsed
            markers.append((cpp_line_num, file_path, python_line))

        _MARKER_CACHE[key] = (st.st_mtime_ns, st.st_size, markers)
        return markers

    def load_line_mapping(self, functor_file_path: Path) -> bool:
        """
        Load line mapping from a C++ functor file by parsing PY_LINE_MARKER comments.
        Format: // PY_LINE_MARKER_<file_path>_<python_lineno>_<hash>

        :param functor_file_path: Path to the functor.hpp file
        :returns: True if mapping was loaded successfully, False otherwise
        """
        if not functor_file_path.exists():
            return False

        resolved_path = functor_file_path.resolve()
        kernel_name = self._kernel_name_from_functor_path(functor_file_path)
        self._kernel_to_path[kernel_name] = resolved_path

        functor_cpp_mapping: Dict[int, Tuple[str, int]] = {}

        for cpp_line_num, file_path, python_line in self._load_markers(resolved_path):
            if cpp_line_num > 1:
                # Primary mapping: (file, line) -> [(functor, cpp_line), ...]
                key = (file_path, python_line)
                self._python_to_cpp.setdefault(key, []).append((resolved_path, cpp_line_num))

                # Reverse mapping: functor -> {cpp_line: (file, python_line)}
                functor_cpp_mapping[cpp_line_num] = (file_path, python_line)

        self._functor_cpp_to_python[resolved_path] = functor_cpp_mapping
        return True

    def apply_handoff_cpp_to_python(self, entries: List[Dict[str, Any]]) -> None:
        """
        Install C++ -> Python locations from the pdb -> device handoff JSON (this launch / bundle only).
        No functor file parsing - parent already resolved markers for the current parallel workunit.
        """
        for e in entries:
            if not isinstance(e, dict):
                continue
            try:
                cpp_file = str(e["cpp_file"])
                cpp_line = int(e["cpp_line"])
                py_file = str(e["py_file"])
                py_line = int(e["py_line"])
            except (KeyError, TypeError, ValueError):
                continue
            cpp_path = Path(_resolved_str(cpp_file))
            py_resolved = _resolved_str(py_file)

            if cpp_path not in self._functor_cpp_to_python:
                self._functor_cpp_to_python[cpp_path] = {}
            self._functor_cpp_to_python[cpp_path][cpp_line] = (py_resolved, py_line)

            py_key = (py_resolved, py_line)
            self._python_to_cpp.setdefault(py_key, [])
            pair: Tuple[Path, int] = (cpp_path, cpp_line)
            if pair not in self._python_to_cpp[py_key]:
                self._python_to_cpp[py_key].append(pair)

    def build_complete_handoff_payload(self, enabled_python: List[Tuple[str, int]]) -> Dict[str, Any]:
        """
        Build the shared-memory handoff dict for the attach helper.

        Includes every (Python file, line) -> C++ location from loaded functor markers,
        not only lines that had user breakpoints at JIT time. Without this, breakpoints
        set from cuda-gdb/rocgdb after attach cannot resolve lines that were not part of
        the initial overlap bundle.
        """
        merged_cpp: Dict[Tuple[str, int], List[List[Any]]] = {}
        for (py_fp, py_ln), cpp_locs in self._python_to_cpp.items():
            pkey = (_resolved_str(py_fp), int(py_ln))
            acc = merged_cpp.setdefault(pkey, [])
            for p, ln in cpp_locs:
                pair: List[Any] = [_resolved_str(p), int(ln)]
                if pair not in acc:
                    acc.append(pair)

        mapping: List[Dict[str, Any]] = []
        cpp_to_python: List[Dict[str, Any]] = []
        seen_ctp: Set[Tuple[str, int, str, int]] = set()

        for py_resolved, py_ln in sorted(merged_cpp.keys()):
            cpp_list = merged_cpp[(py_resolved, py_ln)]
            mapping.append({"py_file": py_resolved, "py_line": py_ln, "cpp": cpp_list})
            for pair in cpp_list:
                ps, ln = str(pair[0]), int(pair[1])
                t = (ps, ln, py_resolved, int(py_ln))
                if t not in seen_ctp:
                    seen_ctp.add(t)
                    cpp_to_python.append(
                        {
                            "cpp_file": ps,
                            "cpp_line": ln,
                            "py_file": py_resolved,
                            "py_line": int(py_ln),
                        }
                    )

        return {
            "mapping": mapping,
            "cpp_to_python": cpp_to_python,
            "enabled_python": [[fp, ln] for fp, ln in enabled_python],
        }

    @staticmethod
    def _functor_types_dir(functor_path: Path) -> Path:
        """Parent of functor.hpp (the ``types_<hash>`` directory)."""
        return functor_path.parent

    @staticmethod
    def _types_dir_has_debug_backend(types_dir: Path) -> bool:
        return any((types_dir / name).is_dir() for name in ("DebugCuda", "DebugHIP", "DebugOpenMP"))

    def _filter_cpp_locations_for_pkdb(self, locations: List[Tuple[Path, int]]) -> List[Tuple[Path, int]]:
        """
        Drop stale pk_cpp variants so GDB breakpoints match the kernel pkdb actually runs.

        PyKokkos emits sibling `types_*` trees (e.g. one with `Cuda/`, one with `DebugCuda/`).
        pkdb JIT-promotes Cuda/HIP/OpenMP to Debug* before launch, so only the Debug* shared
        library loads; breakpoints on the non-debug functor never bind.

        There is deliberately no `pk_cpp` subtree check: every location reaching here already
        comes from the bundle `Runtime.execute_workunit` named for this launch, or from a
        handoff payload built the same way. The check this replaces compared the component
        after `pk_cpp` against the script stem, which only holds when the script sits directly
        in the working directory - PyKokkos mirrors the *workunit* source path under `pk_cpp/`
        (`src/input.py` -> `pk_cpp/src/input/...`), so it discarded every location for a script
        kept in a subdirectory.
        """
        if not locations:
            return locations

        filtered: List[Tuple[Path, int]] = [(Path(fp), cpp_line) for fp, cpp_line in locations]

        if any(self._types_dir_has_debug_backend(self._functor_types_dir(p)) for p, _ in filtered):
            kept = [(p, ln) for p, ln in filtered if self._types_dir_has_debug_backend(self._functor_types_dir(p))]
            if kept:
                if len(kept) != len(filtered):
                    verbose_out(
                        f"[BP_MANAGER] Prefer Debug* build dirs: {len(filtered)} -> {len(kept)} functor location(s)"
                    )
                return kept

        return filtered

    def get_cpp_locations(self, python_line: int, source_file: Optional[str] = None) -> List[Tuple[Path, int]]:
        """
        Get all (functor path, C++ line) pairs for a Python line.
        One Python line may map to multiple kernels (e.g. same line in work and init).

        :param python_line: Python source line number
        :param source_file: Optional source file path to disambiguate when multiple files have same line
        :returns: List of (path, cpp_line) for each kernel that contains this line
        """
        if source_file:
            # Use the new (file, line) key for precise lookup
            try:
                resolved = str(Path(source_file).resolve())
            except Exception:
                resolved = source_file
            key = (resolved, python_line)
            verbose_out(f"[BP_MANAGER] Looking up key: {key}")
            verbose_out(f"[BP_MANAGER] Available keys: {list(self._python_to_cpp.keys())}")
            result = list(self._python_to_cpp.get(key, []))
            verbose_out(f"[BP_MANAGER] Found {len(result)} locations for {key}")
            return self._filter_cpp_locations_for_pkdb(result)
        else:
            # Legacy: return all locations for this line across all files (may be ambiguous)
            all_locs = []
            for (_, line), locs in self._python_to_cpp.items():
                if line == python_line:
                    all_locs.extend(locs)
            verbose_out(f"[BP_MANAGER] Found {len(all_locs)} locations for line {python_line} (no file filter)")
            return self._filter_cpp_locations_for_pkdb(all_locs)

    def get_all_mapped_rows(self) -> List[Tuple[str, int, List[Tuple[Path, int]]]]:
        """
        Every (source_file, python_line, cpp locations) from loaded markers,
        with the same stale-variant filtering as get_cpp_locations().
        """
        rows: List[Tuple[str, int, List[Tuple[Path, int]]]] = []
        for (py_fp, py_ln), locs in sorted(self._python_to_cpp.items()):
            rows.append((py_fp, py_ln, self._filter_cpp_locations_for_pkdb(list(locs))))
        return rows

    def get_python_line_for_functor(self, cpp_line: int, functor_file_path) -> Optional[int]:
        """
        Get Python line number for a C++ line in a functor (backward compatible).
        Returns just the line number without file context.
        """
        location = self.get_python_location_for_functor(cpp_line, functor_file_path)
        if location is None:
            return None
        return location[1]

    def get_python_location_for_functor(self, cpp_line: int, functor_file_path) -> Optional[Tuple[str, int]]:
        """
        Get (source_file, python_line) for a C++ line in a functor.

        :param cpp_line: C++ line number in functor
        :param functor_file_path: Path to functor.hpp file
        :returns: (source_file, python_line) tuple or None
        """
        if functor_file_path is None:
            return None

        path = Path(os.path.realpath(os.path.expanduser(str(functor_file_path))))

        for fp, mapping in self._functor_cpp_to_python.items():
            loc = mapping.get(cpp_line)
            if fp == path and loc is not None:
                return loc

        # GDB / CUDA may report a path symlink-equivalent to the loaded functor path.
        if path.exists():
            rp = os.path.realpath(str(path))
            for fp, mapping in self._functor_cpp_to_python.items():
                loc = mapping.get(cpp_line)
                if fp.exists() and os.path.realpath(str(fp)) == rp and loc is not None:
                    return loc

        target = os.path.normcase(os.path.realpath(str(path)))
        for fp, mapping in self._functor_cpp_to_python.items():
            fp_key = os.path.realpath(str(fp)) if fp.exists() else str(fp)
            loc = mapping.get(cpp_line)
            if os.path.normcase(fp_key) == target and loc is not None:
                return loc

        return None

    def _kernel_name_from_functor_path(self, functor_file_path: Path) -> str:
        """
        Derive kernel name from functor path.
        Path structure: .../pk_cpp/.../script_stem_<kernel>/types_<hash>/functor.hpp
        e.g. test_cuda_work -> "work", test_cuda_init -> "init".

        :param functor_file_path: Path to functor.hpp
        :returns: Kernel name (e.g. "work", "init")
        """
        # Parent of functor.hpp is e.g. types_xxx, its parent is e.g. test_cuda_work
        parent_dir = functor_file_path.parent.parent.name
        if self._script_path:
            script_stem = Path(self._script_path).stem
            prefix = script_stem + "_"
            if parent_dir.startswith(prefix):
                return parent_dir[len(prefix) :]
        return parent_dir

    def _resolve_pk_cpp_path(self, script_path: str) -> Optional[Path]:
        """Resolve the pk_cpp directory path from a script path.

        PyKokkos creates pk_cpp inside the script's directory, e.g.:
          script: .../ExaMiniMD/src/main.py
          pk_cpp: .../ExaMiniMD/src/pk_cpp/
        """
        script_path_obj = Path(script_path).resolve()
        script_dir = script_path_obj.parent
        return script_dir / "pk_cpp"

    def _set_cpp_breakpoint(self, python_line: int, file_path: Optional[str] = None):
        """
        Set C++ breakpoints in GDB for a Python line number.
        Sets a breakpoint in every kernel (functor file) that contains this line.

        :param python_line: Python source line number
        :param file_path: Optional source file path to disambiguate when multiple files have same line number
        """
        if self._gdb_command_sender is None:
            return

        # get_cpp_locations handles file filtering internally
        locations = self.get_cpp_locations(python_line, source_file=file_path)
        verbose_out(
            f"[BP_MANAGER] Setting C++ breakpoints for Python line {python_line}, file={file_path}: found {len(locations)} locations"
        )

        if not locations:
            verbose_out(f"[BP_MANAGER] WARNING: No C++ locations found for line {python_line} in file {file_path}")
            return

        for functor_path, cpp_line in locations:
            break_cmd = f'break-insert -f "{functor_path}:{cpp_line}"'
            verbose_out(f"[BP_MANAGER] Setting breakpoint: {functor_path}:{cpp_line}")
            self._gdb_command_sender(break_cmd)

    def sync_breakpoints_to_gdb(self):
        """
        Sync all active Python breakpoints to C++ breakpoints in GDB.
        This should be called after GDB is started and set up.
        """
        if self._gdb_command_sender is None:
            return

        for python_line in self.breakpoints:
            self._set_cpp_breakpoint(python_line)

    def _update_cpp_breakpoints(self):
        """
        Update C++ breakpoints in GDB for all active breakpoints.
        This is called when breakpoints are added/removed to sync with GDB.
        """
        if self._gdb_command_sender is None:
            return

        # Set C++ breakpoints for all active Python breakpoints
        for python_line in self.breakpoints:
            self._set_cpp_breakpoint(python_line)
