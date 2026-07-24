from __future__ import annotations

import ast
import linecache
import os
import re
from pathlib import Path
import pdb
import textwrap
import warnings
from typing import Dict, Optional, Tuple

import cupy

from pkdb.core.debug_properties import run_pkdb_set_command, verbose_out

from .parallel_launcher import parse_parallel_launch_command, launch_kernels_parallel, format_parallel_results

_PARALLEL_NAMES = frozenset({"parallel_for", "parallel_reduce", "parallel_scan"})


def _is_parallel_ast_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in _PARALLEL_NAMES
    if isinstance(func, ast.Name):
        return func.id in _PARALLEL_NAMES
    return False


def _ast_end_lineno(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end
    mx = getattr(node, "lineno", 1)
    for sub in ast.iter_child_nodes(node):
        if hasattr(sub, "lineno"):
            mx = max(mx, _ast_end_lineno(sub))
    return mx


def _line_numbers_in_code(co) -> set[int]:
    """Bytecode line numbers for this code object (Python 3.10+ co_lines)."""
    lines: set[int] = set()
    try:
        for _bstart, _bend, lineno in co.co_lines():
            if lineno is not None:
                lines.add(lineno)
    except AttributeError:
        pass
    return lines


def _set_frame_lineno_after_parallel(frame, end_line: int) -> None:
    """
    After exec()'ing a swapped multi-line parallel_* block, move the debugger
    to the first valid line after the call. Avoids invalid f_lineno assignments
    and spurious RuntimeWarnings when end+1 is not a bytecode line.
    """
    co = frame.f_code
    valid = _line_numbers_in_code(co)
    target: int | None = None
    if valid:
        nxt = end_line + 1
        if nxt in valid:
            target = nxt
        else:
            later = [ln for ln in valid if ln > end_line]
            if later:
                target = min(later)
    else:
        target = end_line + 1
    if target is None:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frame.f_lineno = target
    except (ValueError, AttributeError):
        pass


def _swap_rule_applies_to_block(source: str, block: str) -> bool:
    """True if ``source`` appears as a whole identifier in ``block`` (not as a substring of another name)."""
    return re.search(r"\b" + re.escape(source) + r"\b", block) is not None


def _apply_kernel_swap_to_block(block: str, source: str, target: str) -> str:
    return re.sub(r"\b" + re.escape(source) + r"\b", target, block)


def _parallel_call_span_for_tree(tree: ast.AST, lineno: int) -> Optional[Tuple[int, int]]:
    """
    If ``lineno`` falls inside a ``parallel_for`` / ``parallel_reduce`` / ``parallel_scan``
    call, return (start_line, end_line) inclusive for that call's AST node (innermost).
    """
    best_start = -1
    best: Optional[Tuple[int, int]] = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_parallel_ast_call(node):
            continue
        start = getattr(node, "lineno", None)
        if start is None:
            continue
        end = _ast_end_lineno(node)
        if start <= lineno <= end and start > best_start:
            best_start = start
            best = (start, end)
    return best


class PdbCommandsMixin(pdb.Pdb):
    """
    Mixin that provides custom pdb commands and helpers for PyKokkosDebugger.

    Expects the concrete debugger class to provide:
      - self._kernel_swaps: Dict[str, str]
      - self._kernel_swaps_loc: Dict[Tuple[str, int], Dict[str, str]]
      - self._launch_native_handoff(frame, handoff_payload=...) -> None
      - self.curframe, self.canonic, self.message, self.error
    """

    _kernel_swaps: Dict[str, str] = {}
    _kernel_swaps_loc: Dict[Tuple[str, int], Dict[str, str]] = {}

    def _effective_kernel_swaps_for_range(self, frame_file: str, start: int, end: int) -> Dict[str, str]:
        """Merge global swaps with any location-specific rules on lines in [start, end] (inclusive)."""
        merged = dict(self._kernel_swaps)
        for ln in range(start, end + 1):
            loc_swaps = self._kernel_swaps_loc.get((frame_file, ln))
            if loc_swaps:
                merged.update(loc_swaps)
        return merged

    def _hotswap_parsed_file_cache(self) -> dict:
        # Attribute must not share this method's name — getattr would return the method.
        attr = "_hotswap_parsed_file_cache_dict"
        c = getattr(self, attr, None)
        if c is None:
            c = {}
            setattr(self, attr, c)
        return c

    def _hotswap_load_parsed_file(self, frame_file: str) -> Tuple[Optional[str], Optional[ast.AST]]:
        """Read and parse ``frame_file`` once per (mtime, size); avoids per-line I/O+parse while tracing."""
        try:
            st = os.stat(frame_file)
        except OSError:
            return None, None
        tag = (st.st_mtime, st.st_size)
        cache = self._hotswap_parsed_file_cache()
        hit = cache.get(frame_file)
        if hit is not None and hit[0] == tag:
            return hit[1], hit[2]
        try:
            with open(frame_file, encoding="utf-8", errors="replace") as sf:
                src = sf.read()
        except OSError:
            return None, None
        try:
            tree = ast.parse(src)
        except SyntaxError:
            cache[frame_file] = (tag, src, None)
            return src, None
        cache[frame_file] = (tag, src, tree)
        return src, tree

    @staticmethod
    def _parse_hotswap_location(loc_arg: str) -> Optional[Tuple[str, int]]:
        """
        Parse 'path:lineno' (line number is the segment after the last ':').
        Returns (absolute_path_str, lineno) or None if invalid.
        """
        if ":" not in loc_arg:
            return None
        path_part, line_part = loc_arg.rsplit(":", 1)
        path_part = path_part.strip()
        if not path_part:
            return None
        try:
            lineno = int(line_part.strip())
        except ValueError:
            return None
        try:
            abs_fp = str(Path(path_part).expanduser().resolve())
        except OSError:
            abs_fp = os.path.abspath(os.path.expanduser(path_part))
        return (abs_fp, lineno)

    def _hotswap_kernels_call(self, frame, frame_file: str, lineno: int) -> bool:
        if not self._kernel_swaps and not self._kernel_swaps_loc:
            return False

        line = linecache.getline(frame_file, lineno)
        if not line or "parallel_" not in line:
            return False

        _file_source, tree = self._hotswap_load_parsed_file(frame_file)
        if tree is None:
            if "parallel_for" not in line and "parallel_reduce" not in line and "parallel_scan" not in line:
                return False
            start = end = lineno
            block = line
            swaps = self._effective_kernel_swaps_for_range(frame_file, lineno, lineno)
        else:
            span = _parallel_call_span_for_tree(tree, lineno)
            if span is None:
                if "parallel_for" not in line and "parallel_reduce" not in line and "parallel_scan" not in line:
                    return False
                start = end = lineno
                block = line
                swaps = self._effective_kernel_swaps_for_range(frame_file, lineno, lineno)
            else:
                start, end = span
                if lineno != start:
                    return False
                has_global = bool(self._kernel_swaps)
                has_loc = bool(self._kernel_swaps_loc)
                if has_loc and not has_global:
                    if (frame_file, start) not in self._kernel_swaps_loc:
                        return False
                block = "".join(linecache.getline(frame_file, i) for i in range(start, end + 1))
                swaps = self._effective_kernel_swaps_for_range(frame_file, start, end)

        if not swaps:
            return False

        for source, target in swaps.items():
            if not _swap_rule_applies_to_block(source, block):
                continue

            swapped = _apply_kernel_swap_to_block(block, source, target)
            code = textwrap.dedent(swapped)
            try:
                exec(code, frame.f_globals, frame.f_locals)

                _set_frame_lineno_after_parallel(frame, end)

                verbose_out(
                    f"hotswap: executed '{target}' instead of '{source}' at "
                    f"{os.path.basename(frame_file)}:{start}-{end}"
                )
                return True
            except Exception as exc:
                self.error(f"hotswap failed at {os.path.basename(frame_file)}:{lineno}: {exc}")
                return False

        return False

    # --- simple demo command ---------------------------------------------

    def do_sayHi(self, arg):
        self.message("Hi")

    def help_sayHi(self):
        self.message("sayHi\n\nPrints 'Hi' in the debugger.")

    # --- kernel hotswap commands -----------------------------------------

    def do_hotswap(self, arg):
        """
        hotswap <source> <target> [path:lineno | lineno]

        Whenever a pk.parallel_* call is encountered that involves <source>
        anywhere in that call (including multi-line calls), the debugger will
        instead execute the same call with <target> substituted for <source>,
        and then skip the rest of the call.

        If a location is given, the substitution applies only there: either
        path:lineno, or a bare lineno (current file from the stack frame when
        you run the command). Other lines keep the original kernel name
        (unless a global hotswap for the same source is also set).
        """
        parts = arg.split()
        if len(parts) not in (2, 3):
            self.error("Usage: hotswap <source> <target> [lineno | path:lineno]")
            return

        source, target = parts[0], parts[1]
        frame = self.curframe
        if len(parts) == 2:
            self._kernel_swaps[source] = target
            self.message(f"hotswap: will substitute '{source}' -> '{target}' on all parallel_* calls")
        else:
            loc_arg = parts[2].strip()
            if loc_arg.isdigit():
                if frame is None:
                    self.error(
                        "hotswap: a bare line number uses the current file; "
                        "there is no current frame — use path:lineno instead."
                    )
                    return
                loc_line = int(loc_arg)
                cfp = self.canonic(frame.f_code.co_filename)
            else:
                parsed = self._parse_hotswap_location(loc_arg)
                if parsed is None:
                    self.error(
                        "Invalid location; use a line number (current file) or path:lineno "
                        "(line number after the last ':')."
                    )
                    return
                abs_fp, loc_line = parsed
                cfp = self.canonic(abs_fp)
            loc_key = (cfp, loc_line)
            inner = self._kernel_swaps_loc.setdefault(loc_key, {})
            inner[source] = target
            self.message(
                f"hotswap: at {os.path.basename(cfp)}:{loc_line} only, "
                f"substitute '{source}' -> '{target}' on parallel_* calls"
            )
        # If we are currently stopped on a swappable pk.parallel_* line,
        # apply the swap immediately so the original line is not executed.
        if frame is not None:
            frame_file = self.canonic(frame.f_code.co_filename)
            lineno = frame.f_lineno
            self._hotswap_kernels_call(frame, frame_file, lineno)

    def help_hotswap(self):
        self.message(
            "hotswap <source> <target> [lineno | path:lineno]\n\n"
            "On pk.parallel_* calls that reference <source> (including multi-line\n"
            "calls), execute the same call with <target> substituted for <source>,\n"
            "and skip the original call.\n"
            "With lineno alone, the rule applies only at that line in the current file\n"
            "(the file of the stack frame when you run the command). With path:lineno,\n"
            "it applies only at that file and line (use the first line of the parallel_*\n"
            "call). Elsewhere that source is unchanged unless you also set a global hotswap.\n"
            "Breakpoint tip: set b file:line at least two source lines away from the\n"
            "parallel_* call / workunit reference; closer lines can confuse pkdb.\n"
            "Example: hotswap work lazy\n"
            "Example: hotswap work lazy 42\n"
            "Example: hotswap work lazy /path/to/script.py:42"
        )

    # Backwards-compatible alias: swap_kernels -> hotswap
    def do_swap_kernels(self, arg):
        return self.do_hotswap(arg)

    def help_swap_kernels(self):
        return self.help_hotswap()

    # --- parallel launch commands ----------------------------------------

    def do_parallel_launch(self, arg):
        """parallel-launch K1 <args>; K2 pk.Space <args>; ..."""
        if not arg or not arg.strip():
            self.error("Usage: parallel-launch K1 <args>; K2 <args>; ...")
            return

        frame = self.curframe
        if frame is None:
            self.error("No current frame available")
            return

        try:
            # Get frame globals for variable resolution
            frame_globals = {**frame.f_globals, **frame.f_locals}
            kernel_specs = parse_parallel_launch_command(arg, frame_globals)

            if not kernel_specs:
                self.error("No valid kernel specifications found")
                return

            self.message(f"Launching {len(kernel_specs)} kernel(s) in parallel...")
            results = launch_kernels_parallel(kernel_specs, frame_globals)
            output = format_parallel_results(results)
            self.message(output)

        except Exception as e:
            import traceback

            self.error(f"Parallel launch failed: {e}\n{traceback.format_exc()}")

    def help_parallel_launch(self):
        self.message(
            "parallel-launch <kernel> <policy> [args...]; ...\n\n"
            "Per-argument CUDA IPC (GPU spawn only): mut: or mod: (same meaning) with the workunit\n"
            "parameter name only, e.g. mut:a=a, mod:a=my_a, or mut:a (value from variable a).\n"
            "Unmarked CuPy\n"
            "args are copied through the host (isolated per worker).\n\n"
            "Host launches (e.g. pk.OpenMP) run in-process (reuses PyKokkos JIT cache).\n"
            "GPU launches default to in-process too (PKDB_PARALLEL_LAUNCH_CUDA_INPROCESS=1); set =0 for spawn. "
            "PKDB_PARALLEL_LAUNCH_TIMEOUT (default 180s) applies only to spawn.\n"
            "Form: [reduce|scan] <kernel> <policy> [kw...] or ... pk.<ExecSpace> <policy> [kw...].\n"
            "Use reduce (or scan) when the workunit is a parallel_reduce / parallel_scan kernel.\n"
            "Kwarg tokens are name=value (evaluated in the current frame); the value may be the next\n"
            "token after a trailing = (e.g. cols= M).\n\n"
            "Example: parallel-launch work pk.DebugCuda pk.RangePolicy(0,N) mut:a\n"
            "Example: parallel-launch work pk.DebugCuda pk.RangePolicy(0,N) a mut:b\n"
            "Example: parallel-launch reduce yAx pk.DebugCuda pk.RangePolicy(0,N) cols=M y_view=y x_view=x A_view=A"
        )

    # --- pkdb debug properties (verbose, print_elements, …) ----------------
    def do_set(self, arg):
        """set [name [value]] — show or change pkdb debug properties."""
        run_pkdb_set_command(arg, writeln=self.message)

    def help_set(self):
        self.message(
            "set [name [value]]\n\n"
            "Without args: list all properties. With one arg: show that property.\n"
            "With two args: set property — booleans use on/off; integers must be positive.\n"
            "Examples: set verbose on    set print_elements 256"
        )

    # --- basic overridden commands ----------------------------------------
    def do_d(self, arg):
        """d(own) [count] or d(elete) file:lineno - clear breakpoint when arg looks like file:line"""
        if arg and ":" in arg:
            return self.do_clear(arg)
        return self.do_down(arg)

    def do_delete(self, arg):
        """delete file:lineno or bpnumber - alias for clear"""
        return self.do_clear(arg)
