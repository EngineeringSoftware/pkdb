"""
Map workunit parameter names to caller variable names.
"""

from __future__ import annotations

import ast
from typing import Optional

_PARALLEL_ATTRS = frozenset({"parallel_for", "parallel_reduce", "parallel_scan"})


def _is_parallel_launch_func(func: ast.AST) -> bool:
    if isinstance(func, ast.Name) and func.id in _PARALLEL_ATTRS:
        return True
    if isinstance(func, ast.Attribute) and func.attr in _PARALLEL_ATTRS:
        return True
    return False


def _find_parallel_call_on_line(tree: ast.AST, lineno: int) -> Optional[ast.Call]:
    """Find the parallel_* call that contains the given line number."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_parallel_launch_func(node.func):
            continue
        end = getattr(node, "end_lineno", None) or node.lineno
        if node.lineno <= lineno <= end:
            return node
    return None


def build_keyword_param_to_caller_map(call: ast.Call) -> dict[str, str]:
    """
    For ``parallel_*(..., p=caller_name, ...)``, return {``p``: ``caller_name``}
    for each keyword whose value is a bare ``ast.Name``.
    """
    m: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        if isinstance(kw.value, ast.Name):
            m[kw.arg] = kw.value.id
    return m


def build_workunit_param_to_caller_map(source: str, filepath: str, call_lineno: int) -> dict[str, str]:
    """
    Parse *source*, find the ``parallel_*`` call that contains *call_lineno*, and return
    the keyword map from that call only (e.g. ``cols=M``, ``y_view=y``).
    """
    try:
        tree = ast.parse(source, filepath)
    except SyntaxError:
        return {}

    call = _find_parallel_call_on_line(tree, call_lineno)
    if call is None:
        return {}

    return build_keyword_param_to_caller_map(call)


def build_workunit_param_to_caller_map_from_file(script_path: str, call_lineno: int) -> dict[str, str]:
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError:
        return {}
    return build_workunit_param_to_caller_map(source, script_path, call_lineno)
