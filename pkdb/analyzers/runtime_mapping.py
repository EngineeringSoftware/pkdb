"""
Runtime analysis tools for ``debugger.py``: Python workunit lines → C++ via ``PY_LINE_MARKER``
under a PyKokkos ``Debug*`` bundle directory.
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

    @staticmethod
    def iter_bundle_cpp_mapping_files(bundle_output_dir: Path) -> List[Path]:
        """``functor.hpp`` beside the ``DebugCuda``/``DebugHIP``/``DebugOpenMP`` dir plus sources under the bundle."""
        out: List[Path] = []
        try:
            od = bundle_output_dir.resolve()
        except OSError:
            od = bundle_output_dir
        if not od.is_dir():
            return out
        try:
            for p in od.parent.glob("*.hpp"):
                out.append(p)
        except OSError:
            pass
        for pat in ("*.hpp", "*.cpp", "*.cu", "*.cuh"):
            try:
                out.extend(od.rglob(pat))
            except OSError:
                pass
        seen: set[Path] = set()
        uniq: List[Path] = []
        for p in out:
            try:
                r = p.resolve()
            except OSError:
                r = p
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq

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
