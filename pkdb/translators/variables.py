import re
from typing import Optional, Tuple, List


def calculate_linear_index(indices, dimensions):
    """
    Calculate linear index from multi-dimensional indices and dimensions.

    args:
        indices: List of indices for each dimension
        dimensions: List of dimension sizes

    returns:
        Calculated linear index
    """
    if len(indices) != len(dimensions):
        raise ValueError(f"Indices length {len(indices)} must match dimensions length {len(dimensions)}")

    linear_index = 0
    stride = 1

    for i in range(len(indices) - 1, -1, -1):
        linear_index += indices[i] * stride
        if i > 0:
            stride *= dimensions[i]

    return linear_index


def parse_indexing_expression(expr: str) -> Optional[Tuple[str, List[int]]]:
    """
    Parse a Python-style indexing expression like 'view[5]' or 'view[5][6]'.

    args:
        expr: The expression to parse (e.g., 'view[5]', 'view[5][6]')

    returns:
        A tuple of (variable_name, list_of_indices) if parsing succeeds, None otherwise.
    """
    pattern = r"^(\w+)((?:\[\s*\d+\s*\])+)$"
    match = re.match(pattern, expr.replace(" ", ""))

    if not match:
        return None

    var_name = match.group(1)
    index_part = match.group(2)
    index_pattern = r"\[(\d+)\]"
    indices = [int(idx) for idx in re.findall(index_pattern, index_part)]

    return (var_name, indices)


def get_view_dimensions(gdb_controller, var_name: str) -> Optional[List[int]]:
    """
    Get the dimensions of a view variable from GDB.

    args:
        gdb_controller: The GDB controller instance to use for queries
        var_name: The name of the view variable

    returns:
        A list of dimensions if successful, None otherwise.
    """
    from ..controllers.accelerator_gdb_controller import AcceleratorGDBController

    # regular GDB (OpenMP), we can access m_dim.Ni directly
    if not isinstance(gdb_controller, AcceleratorGDBController):
        dimensions = []
        max_rank = 8
        for i in range(max_rank):
            dim_expr = f"{var_name}.m_map.m_impl_offset.m_dim.N{i}"
            responses = gdb_controller._send_mi_command(f'data-evaluate-expression "{dim_expr}"')

            dim = None
            for resp in responses:
                if resp.get("type") == "result" and resp.get("message") == "done":
                    value_str = resp.get("payload", {}).get("value", "")
                    try:
                        dim = int(value_str)
                        if dim > 0:
                            dimensions.append(dim)
                        break
                    except ValueError:
                        pass

            if dim is None or dim <= 0:
                break

        return dimensions if dimensions else None

    # CUDA-GDB, direct access fails due to name mangling
    # ({m_dim = {<mangled> ...}})
    dim_expr = f"{var_name}.m_map.m_impl_offset"
    responses = gdb_controller._send_mi_command(f'data-evaluate-expression "{dim_expr}"')

    value_str = None
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            value_str = resp.get("payload", {}).get("value", "")
            break

    if not value_str:
        return None

    dimensions = []
    max_rank = 8

    for i in range(max_rank):
        pattern = f"N{i} = "
        idx = value_str.find(pattern)

        if idx == -1:
            break

        start_idx = idx + len(pattern)
        end_idx = start_idx

        while end_idx < len(value_str) and (value_str[end_idx].isdigit() or value_str[end_idx] == "-"):
            end_idx += 1

        try:
            dim_value = int(value_str[start_idx:end_idx])
            if dim_value > 0:
                dimensions.append(dim_value)
            else:
                break
        except (ValueError, IndexError):
            break

    return dimensions if dimensions else None


def translate_view_indexing(gdb_controller, expr: str) -> Optional[str]:
    """
    Translate a Python-style view indexing expression to C++ format.

    args:
        gdb_controller: The GDB controller instance to use for queries
        expr: The Python-style expression (e.g., 'view[5]', 'view[5][6]')

    returns:
        The translated C++ expression, or None if translation fails.
    """
    parsed = parse_indexing_expression(expr)
    if not parsed:
        return None

    var_name, indices = parsed

    if len(indices) == 1:
        return f"{var_name}.m_map.m_impl_handle[{indices[0]}]"

    dimensions = get_view_dimensions(gdb_controller, var_name)
    if not dimensions:
        return None

    if len(indices) != len(dimensions):
        return None

    try:
        linear_index = calculate_linear_index(indices, dimensions)
    except ValueError:
        return None

    return f"{var_name}.m_map.m_impl_handle[{linear_index}]"
