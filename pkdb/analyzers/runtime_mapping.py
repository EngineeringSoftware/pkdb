"""
Runtime analysis tools for `debugger.py`: Python workunit lines -> C++ via `PY_LINE_MARKER`
under a PyKokkos `Debug*` bundle directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from pkdb.core.breakpoint_manager import BreakpointManager

# (python source path, line), list of (functor/cpp path, cpp line); empty inner list = no marker
OverlapCppMapping = Tuple[str, int, List[Tuple[Path, int]]]


class RuntimeBundleMapping:
    """Load bundle marker files and map overlapping Python lines to C++ locations."""

    def __init__(self, script_path: str) -> None:
        self._script_path = script_path

    # List of files that we  are intersted at.
    _SOURCES_BESIDE_BUNDLE = ("functor.hpp", "functor_cast.hpp")
    _SOURCES_IN_BUNDLE = ("bindings.cpp",)

    @classmethod
    def iter_bundle_cpp_mapping_files(cls, bundle_output_dir: Path) -> List[Path]:
        """The generated sources for one bundle - `functor.hpp`/`functor_cast.hpp` beside the
        `DebugCuda`/`DebugHIP`/`DebugOpenMP` dir, plus `bindings.cpp` inside it."""
        try:
            od = bundle_output_dir.resolve()
        except OSError:
            od = bundle_output_dir
        if not od.is_dir():
            return []
        candidates = [od.parent / name for name in cls._SOURCES_BESIDE_BUNDLE]
        candidates += [od / name for name in cls._SOURCES_IN_BUNDLE]
        return [p for p in candidates if p.is_file()]

    def resolve_overlap(
        self,
        overlap: List[Tuple[str, int]],
        bundle_output_dir: Path,
    ) -> Tuple[List[OverlapCppMapping], List[OverlapCppMapping]]:
        """
        Load marker mappings from *bundle_output_dir* and map each overlapping Python breakpoint
        to zero or more C++ (file, line) pairs (same kernel may appear in multiple functors).

        :returns: ``(rows for the overlap lines, rows for every mapped line in the bundle)``
        """
        bm = BreakpointManager(script_path=self._script_path)
        for fp in self.iter_bundle_cpp_mapping_files(bundle_output_dir):
            bm.load_line_mapping(fp)

        rows: List[OverlapCppMapping] = []
        for py_fp, py_ln in overlap:
            cpp_locs = list(bm.get_cpp_locations(py_ln, source_file=py_fp))
            rows.append((py_fp, py_ln, cpp_locs))

        return rows, bm.get_all_mapped_rows()
