"""
Variable resolution for expression evaluation.
"""

from typing import Any
from .context import EvaluationContext, ContextType


def _convert_to_type(value_str: str) -> Any:
    """Convert a string value to its proper Python type (int, float, or str).
    If GDB truncated the display with '...' on a number (e.g. '1991...'), parse the numeric part.
    """
    value_str = value_str.strip()
    if not value_str:
        return value_str
    # GDB may truncate the value string with "..."; strip it and use the numeric part only
    if value_str.endswith("..."):
        value_str = value_str[:-3].strip()

    try:
        return int(value_str)
    except (ValueError, TypeError):
        try:
            return float(value_str)
        except (ValueError, TypeError):
            return value_str


def gdb_string_result_to_python(value: Any) -> Any:
    """
    Turn a GDB MI value string into a Python scalar or list.

    Native C arrays and Kokkos aggregates are printed like ``{3, 1, 4}``. If we leave that as a
    Python str, ``cell_xyz[0]`` in the expression evaluator becomes string indexing (``'{'``)
    instead of the first element, which breaks arithmetic (e.g. str * str TypeError).
    """
    if not isinstance(value, str):
        return value
    t = value.strip()
    if len(t) >= 2 and t[0] == "{" and t[-1] == "}":
        inner = t[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return [_convert_to_type(p) for p in parts]
    return _convert_to_type(t)


def resolve_variable(context: EvaluationContext, expr: str) -> Any:
    if context.type == ContextType.PDB:
        if context.frame is None:
            raise RuntimeError("No frame available in PDB context")
        try:
            result = eval(expr, context.frame.f_globals, context.frame.f_locals)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to resolve variable '{expr}': {e}")

    from ..translators.variables import translate_view_indexing

    translated = translate_view_indexing(context.controller, expr)
    return translated if translated else expr


def resolve_array_info(context: EvaluationContext, expr: str) -> dict:
    if context.type == ContextType.PDB:
        array = resolve_variable(context, expr)
        info = {}
        if hasattr(array, "shape"):
            info["dimensions"] = list(array.shape)
            info["size"] = array.size if hasattr(array, "size") else len(array)
        elif hasattr(array, "__len__"):
            info["size"] = len(array)
            info["dimensions"] = [len(array)]
        else:
            info["size"] = 1
            info["dimensions"] = []
        if hasattr(array, "dtype"):
            info["dtype"] = str(array.dtype)
        return info

    from ..translators.variables import get_view_dimensions

    base_var = expr.split("[")[0] if "[" in expr else expr
    dimensions = get_view_dimensions(context.controller, base_var)

    if dimensions:
        total_size = 1
        for dim in dimensions:
            total_size *= dim
        return {"dimensions": dimensions, "size": total_size, "dtype": "unknown"}
    return {"size": None, "dimensions": None, "dtype": None}


def resolve_and_evaluate_variable(context: EvaluationContext, expr: str) -> Any:
    if context.type == ContextType.PDB:
        return resolve_variable(context, expr)

    from ..commands.gdb_helpers import (
        get_variable_value,
        parse_array_slice,
        _evaluate_with_cast_if_address,
    )
    from ..core.debug_properties import verbose_out

    slice_info = parse_array_slice(expr)

    if slice_info:
        _, start, end = slice_info
        result_str = get_variable_value(context.controller, expr)
        if result_str.startswith("{") and result_str.endswith("}"):
            values_str = result_str[1:-1]
            values = [_convert_to_type(val) for val in values_str.split(",") if val.strip()]
            print_limit = getattr(getattr(context.controller, "debug_properties", None), "print_elements", 128)
            if not isinstance(print_limit, int) or print_limit <= 0:
                print_limit = 128
            if (end - start) > print_limit:
                values.append("...")
            return values
        return _convert_to_type(result_str)

    translated_expr = resolve_variable(context, expr)
    verbose_out(f"[VARIABLE RESOLVER] expr={expr}, translated_expr={translated_expr}")

    responses = context.execute_gdb_command(f'data-evaluate-expression "{translated_expr}"')
    verbose_out(f"[VARIABLE RESOLVER] responses={responses}")

    base_name = expr.split("[")[0].strip()
    for resp in responses:
        if resp.get("type") == "result":
            if resp.get("message") == "done":
                value = resp.get("payload", {}).get("value", "?")
                verbose_out(f"[VARIABLE RESOLVER] Got value={value} for {expr}")
                value = _evaluate_with_cast_if_address(context.controller, translated_expr, value, base_name)
                return gdb_string_result_to_python(value)
            elif resp.get("message") == "error":
                error_msg = resp.get("payload", {}).get("msg", "Unknown error")
                verbose_out(f"[VARIABLE RESOLVER] GDB error: {error_msg}")
                raise RuntimeError(f"GDB error evaluating '{expr}': {error_msg}")

    raise RuntimeError(f"Could not evaluate '{expr}' in GDB context")
