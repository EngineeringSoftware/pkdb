"""
GDB MI3 helper (help to send command + get internal results)
"""

import re
import sys

from ..translators.variables import translate_view_indexing


# Default when GDB max-value-size cannot be queried (64 KB)
_DEFAULT_MAX_VALUE_SIZE = 65536


def _get_print_elements_limit(controller) -> int:
    """Get max array elements to fetch/print from debug properties."""
    debug_props = getattr(controller, "debug_properties", None)
    if debug_props is None:
        return 128
    value = getattr(debug_props, "print_elements", 128)
    if not isinstance(value, int) or value <= 0:
        return 128
    return value


def get_gdb_max_value_size(controller) -> int:
    """
    Query GDB's max-value-size (max bytes for a single value). Used to cap
    array-slice requests so we don't trigger "value requires N bytes, which is
    more than max-value-size".
    """
    try:
        responses = controller._send_mi_command("gdb-show max-value-size")
        for resp in responses:
            if resp.get("type") == "result" and resp.get("message") == "done":
                payload = resp.get("payload", {})
                val = payload.get("value", "")
                if isinstance(val, int):
                    return max(16, val)
                if isinstance(val, str):
                    val = val.strip().lower()
                    if val == "unlimited":
                        return 1024 * 1024  # 1 MB cap when unlimited
                    # e.g. "65536" or "64k"
                    import re as _re

                    m = _re.match(r"^(\d+)\s*(k|m)?$", val.replace(" ", ""))
                    if m:
                        n = int(m.group(1))
                        unit = (m.group(2) or "").lower()
                        if unit == "k":
                            n *= 1024
                        elif unit == "m":
                            n *= 1024 * 1024
                        return max(16, n)
                    try:
                        return max(16, int(val))
                    except ValueError:
                        pass
                break
    except Exception:
        pass
    return _DEFAULT_MAX_VALUE_SIZE


def _element_size_from_base(controller, base_expr: str) -> int:
    """Infer element size in bytes from base variable (e.g. 'a' -> 8 for int64). Default 8."""
    base_name = base_expr.split("[")[0].strip()
    cpp_type = get_variable_cpp_type(controller, base_name) if hasattr(controller, "_send_mi_command") else None
    if not cpp_type:
        return 8
    cpp_type = (cpp_type or "").lower()
    if "double" in cpp_type or "float64" in cpp_type:
        return 8
    if "float" in cpp_type or "float32" in cpp_type:
        return 4
    if "int64" in cpp_type or "long long" in cpp_type or "uint64" in cpp_type:
        return 8
    if "int32" in cpp_type or "int " in cpp_type or "uint32" in cpp_type:
        return 4
    if "short" in cpp_type or "int16" in cpp_type:
        return 2
    if "char" in cpp_type or "int8" in cpp_type:
        return 1
    return 8


def parse_array_slice(expr: str) -> tuple[str, int, int] | None:
    """
    Parse Python-style array slice notation and separate base from slice.

    Examples:
        'a[0:10]' -> ('a', 0, 10)
        'a[0][1:5]' -> ('a[0]', 1, 5)
        'a[2][3][5:10]' -> ('a[2][3]', 5, 10)

    Args:
        expr: The expression to parse (e.g., 'a[0:10]' or 'a[0][1:5]')

    Returns:
        Tuple of (base_expr, start, end) if parsing succeeds, None otherwise.
        base_expr is the part before the slice, start/end are slice bounds
    """

    # Check if expression has slice notation at the end: [...][start:end]
    slice_pattern = r"^(.+)\[(\d+):(\d+)\]$"
    match = re.match(slice_pattern, expr.strip())

    if not match:
        return None

    base_expr = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))

    if end <= start:
        return None

    return (base_expr, start, end)


def get_frame_info(controller):
    responses = controller._send_mi_command("stack-info-frame")
    if not responses:
        return None
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            frame = resp.get("payload", {}).get("frame", {})
            file = frame.get("file", frame.get("fullname", "?"))
            line = frame.get("line", "?")
            return f"{file}:{line}"
    return None


def _is_functor_frame(frame_data):
    """True if frame is in functor.hpp."""
    file = frame_data.get("file", "")
    fullname = frame_data.get("fullname", file)
    return "functor.hpp" in (file or "") or "functor.hpp" in (fullname or "")


def get_current_python_location(controller):
    """
    Get current execution location in Python source (file and line).
    When stopped in a workunit (functor.hpp), maps C++ line to Python line.
    Returns (script_path, python_line) or (None, None) if not in mappable Python code.
    """
    result = (None, None)
    responses = controller._send_mi_command("stack-info-frame")
    frame = None
    if responses:
        for resp in responses:
            if resp.get("type") == "result" and resp.get("message") == "done":
                frame = resp.get("payload", {}).get("frame", {})
                break
    if frame and _is_functor_frame(frame):
        bp_manager = getattr(controller, "breakpoint_manager", None)
        script_path = getattr(controller, "script_path", None) or ""
        line_str = frame.get("line", "")
        file = frame.get("file", "")
        fullname = frame.get("fullname", file)
        if bp_manager and script_path and line_str and line_str.isdigit():
            cpp_line = int(line_str)
            location = bp_manager.get_python_location_for_functor(cpp_line, fullname or file)
            if location is not None:
                result = location  # (source_file, python_line)
    return result


def get_thread_counts(controller):
    """
    Get current thread id, total thread count, and workunit (OpenMP) thread count.
    Returns (current_thread_id, total_count, workunit_count).
    """
    result = (None, 0, 0)
    responses = controller._send_mi_command("thread-info")
    payload = None
    if responses:
        for resp in responses:
            if resp.get("type") == "result" and resp.get("message") == "done":
                payload = resp.get("payload", {})
                break
    if payload:
        current_id = payload.get("current-thread-id")
        threads_list = payload.get("threads", [])
        total_count = len(threads_list)
        workunit_count = sum(1 for t in threads_list if _is_functor_frame(t.get("frame", {}))) if threads_list else 0
        result = (current_id, total_count, workunit_count)
    return result


# def _get_view_dtype_for_cupy(controller, var_name: str):
#     """Infer view element dtype from GDB 'whatis' for use with CuPy. Returns numpy dtype or None."""
#     import numpy as np

#     responses = controller._send_mi_command(f'interpreter-exec console "whatis {var_name}"')
#     type_str = ""
#     for resp in responses:
#         if resp.get("type") == "console":
#             type_str = resp.get("payload", "")
#             break
#     type_str = (type_str or "").lower()
#     if "double" in type_str:
#         return np.float64
#     if "float" in type_str:
#         return np.float32
#     if "int64" in type_str or "long long" in type_str:
#         return np.int64
#     if "int32" in type_str or "int" in type_str:
#         return np.int32
#     return np.float64


def _chunk_result_to_values(chunk_str: str) -> list:
    """Parse a single chunk result (e.g. '{0, 10, 20}' or '0, 10, 20') into list of value strings."""
    if not chunk_str or not isinstance(chunk_str, str):
        return []
    s = chunk_str.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return [v.strip() for v in s.split(",") if v.strip()]


def _is_value_string_truncated(value_str: str) -> bool:
    """True if GDB truncated the value string (ends with ..., so we got incomplete data)."""
    if not value_str or "..." not in value_str:
        return False
    s = value_str.strip()
    # GDB truncates with ", ...}" or "...}" at the end of the value
    return s.endswith("...}") or s.endswith("...'") or ", ...}" in s or ", ...'" in s


def _fetch_one_slice_chunk(controller, base_expr, chunk_start: int, chunk_count: int):
    """
    Fetch one chunk of an array slice (chunk_count elements starting at chunk_start).
    Returns (value_string, None) on success or (error_string, error_msg) on failure.
    """
    expr_with_start = f"{base_expr}[{chunk_start}]"
    translated_expr = translate_view_indexing(controller, expr_with_start)
    if not translated_expr:
        translated_expr = expr_with_start
    expr_to_use = f"{translated_expr}@{chunk_count}"

    responses = controller._send_mi_command(f'data-evaluate-expression "{expr_to_use}"')
    if not responses:
        return "?", None
    for resp in responses:
        if resp.get("type") == "result":
            if resp.get("message") == "done":
                return resp.get("payload", {}).get("value", "?"), None
            if resp.get("message") == "error":
                error_msg = resp.get("payload", {}).get("msg", "Unknown error")
                return f"Error: {error_msg}", error_msg
    return "?", None


def get_array_slice_value(controller, base_expr, start, end):
    """
    Get array slice value by translating base expression and adding slice notation.
    If the requested slice is larger than GDB's max-value-size, retrieves the array
    in portions, concatenates them, and returns the full result.

    For CUDA/HIP, prefers reading via CuPy after device sync so IPC (other-process) updates are visible.

    Args:
        controller: GDB controller
        base_expr: Base expression before the slice (e.g., 'a' or 'a[0]')
        start: Start index of the slice
        end: End index of the slice

    Returns:
        String representation of the full array slice in the form "{v1, v2, ...}".
    """
    requested_count = end - start
    print_limit = _get_print_elements_limit(controller)
    if requested_count > print_limit:
        requested_count = print_limit
        end = start + print_limit
    max_value_size = get_gdb_max_value_size(controller)
    element_size = _element_size_from_base(controller, base_expr)
    # Start with largest chunk allowed by GDB's max-value-size; shrink if GDB truncates the value string
    chunk_size = max(1, max_value_size // element_size)

    if requested_count <= chunk_size:
        result, err_msg = _fetch_one_slice_chunk(controller, base_expr, start, requested_count)
        if err_msg and "max-value-size" in (err_msg or ""):
            while chunk_size > 1:
                chunk_size = max(1, chunk_size // 2)
                result, err_msg = _fetch_one_slice_chunk(controller, base_expr, start, min(requested_count, chunk_size))
                if err_msg is None or "max-value-size" not in (err_msg or ""):
                    break
            if err_msg and "max-value-size" in (err_msg or ""):
                return f"Error: {err_msg}"
        # If single chunk is truncated, fetch in smaller pieces and concatenate
        if requested_count > 1 and _is_value_string_truncated(result):
            chunk_size = max(1, requested_count // 2)
            all_values = []
            o = 0
            while o < requested_count:
                n = min(chunk_size, requested_count - o)
                r, e = _fetch_one_slice_chunk(controller, base_expr, start + o, n)
                if e:
                    return f"Error: {e}" if e else "?"
                while _is_value_string_truncated(r) and n > 1:
                    n = max(1, n // 2)
                    r, e = _fetch_one_slice_chunk(controller, base_expr, start + o, n)
                    if e:
                        return f"Error: {e}" if e else "?"
                all_values.extend(_chunk_result_to_values(r))
                o += n
            return "{" + ",".join(all_values) + "}"
        return result

    all_values = []
    offset = 0
    while offset < requested_count:
        chunk_start = start + offset
        count_this_chunk = min(chunk_size, requested_count - offset)
        result, err_msg = _fetch_one_slice_chunk(controller, base_expr, chunk_start, count_this_chunk)
        if err_msg:
            if "max-value-size" in (err_msg or ""):
                chunk_size = max(1, chunk_size // 2)
                continue
            return f"Error: {err_msg}"
        # If GDB truncated the value string, retry this range with a smaller chunk
        if count_this_chunk > 1 and _is_value_string_truncated(result):
            chunk_size = max(1, count_this_chunk // 2)
            continue
        values = _chunk_result_to_values(result)
        all_values.extend(values)
        offset += count_this_chunk
        if len(values) < count_this_chunk:
            break

    return "{" + ",".join(all_values) + "}"


def _value_needs_cast(value: str, cpp_type: str | None) -> bool:
    """True if type is scalar but value contains non-scalar chars (e.g. 0x, @) so it's an address."""
    if not value or not isinstance(value, str) or not cpp_type:
        return False
    t = re.sub(r"\s*[&*]\s*$", "", cpp_type.strip())
    t = re.sub(r"^const\s+", "", t).strip()
    scalar = (
        "float",
        "double",
        "int",
        "long",
        "short",
        "bool",
        "int32_t",
        "int64_t",
        "uint32_t",
        "uint64_t",
        "unsigned int",
        "unsigned long",
    )
    if not any(s in t or t == s for s in scalar):
        return False
    v = value.strip()
    return "@" in v or "0x" in v.lower()


def get_variable_cpp_type(controller, var_name: str) -> str | None:
    """C++ type of var_name in current frame from GDB, or None"""
    responses = controller._send_mi_command("stack-list-variables --simple-values")
    if not responses:
        return None
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            for var in resp.get("payload", {}).get("variables", []):
                if var.get("name") == var_name:
                    return var.get("type") or None
    return None


def cpp_type_to_cast(cpp_type: str) -> str | None:
    """Map C++ type to C cast string e.g. '(double)', or None"""
    if not cpp_type:
        return None
    t = re.sub(r"\s*[&*]\s*$", "", cpp_type.strip())
    t = re.sub(r"^const\s+", "", t)
    t = t.strip()
    cast_map = {
        "float": "(float)",
        "double": "(double)",
        "int": "(int)",
        "long": "(long)",
        "long long": "(long long)",
        "short": "(short)",
        "int32_t": "(int)",
        "int64_t": "(long long)",
        "uint32_t": "(unsigned int)",
        "uint64_t": "(unsigned long long)",
        "bool": "(bool)",
        "unsigned int": "(unsigned int)",
        "unsigned long": "(unsigned long)",
    }
    for cpp, cast in cast_map.items():
        if cpp in t or t == cpp:
            return cast
    return None


def _evaluate_with_cast_if_address(controller, expr_to_use: str, value: str, base_name: str) -> str:
    """If type is scalar but value contains address chars, re-evaluate with cast and return value."""
    cpp_type = get_variable_cpp_type(controller, base_name)
    if not _value_needs_cast(value, cpp_type):
        return value
    cast = cpp_type_to_cast(cpp_type) if cpp_type else None
    if not cast:
        return value
    cast_expr = f"{cast}{expr_to_use}"
    responses = controller._send_mi_command(f'data-evaluate-expression "{cast_expr}"')
    for resp in responses:
        if resp.get("type") == "result":
            if resp.get("message") == "done":
                return resp.get("payload", {}).get("value", value)
            if resp.get("message") == "error":
                break
    return value


def get_single_value(controller, expr):
    """Get a single variable value (non-slice) as string."""

    translated_expr = translate_view_indexing(controller, expr)
    expr_to_use = translated_expr if translated_expr else expr
    base_name = expr.split("[")[0].strip()

    responses = controller._send_mi_command(f'data-evaluate-expression "{expr_to_use}"')
    if not responses:
        return "?"
    for resp in responses:
        if resp.get("type") == "result":
            if resp.get("message") == "done":
                value = resp.get("payload", {}).get("value", "?")
                return _evaluate_with_cast_if_address(controller, expr_to_use, value, base_name)
            elif resp.get("message") == "error":
                error_msg = resp.get("payload", {}).get("msg", "Unknown error")
                return f"Error: {error_msg}"
    return "?"


def get_variable_value(controller, var_name):
    """
    Get variable value, handling both single values and array slices.

    Args:
        controller: GDB controller
        var_name: Variable name or expression (may include slice notation)

    Returns:
        String representation of the value
    """
    slice_parsed = parse_array_slice(var_name)
    if slice_parsed:
        base_expr, start, end = slice_parsed
        return get_array_slice_value(controller, base_expr, start, end)

    return get_single_value(controller, var_name)


def is_in_workunit(controller):
    responses = controller._send_mi_command("stack-info-frame")
    if not responses:
        return False
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            frame = resp.get("payload", {}).get("frame", {})
            file_path = frame.get("file", frame.get("fullname", ""))
            return file_path.endswith("@functor.hpp") or file_path.endswith("functor.hpp")
    return False


def get_locals(controller):
    from ..core import infer_python_type

    responses = controller._send_mi_command("stack-list-variables --simple-values")
    if not responses:
        return "No local variables"
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            variables = resp.get("payload", {}).get("variables", [])
            if not variables:
                return "No local variables"

            lines = []
            for var in variables:
                name = var.get("name", "")
                if name == "this":
                    continue
                value = var.get("value", "?")
                cpp_type = var.get("type", "")

                python_type = infer_python_type(cpp_type)

                line = f"{name} = {value}"
                if python_type:
                    line += f"  ({python_type})"
                lines.append(line)

            return "\n".join(lines) if lines else "No local variables"
    return "No local variables"


def get_class_members(controller):
    var_name = "pkdb_this"
    create_responses = controller._send_mi_command(f"var-create {var_name} * this")

    actual_var_name = None
    for resp in create_responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            actual_var_name = resp.get("payload", {}).get("name", var_name)
            break

    if not actual_var_name:
        return None

    try:
        controller._send_mi_command(f"var-list-children {actual_var_name}")

        public_responses = controller._send_mi_command(f"var-list-children {actual_var_name}.public")

        children = []
        for resp in public_responses:
            if resp.get("type") == "result" and resp.get("message") == "done":
                children = resp.get("payload", {}).get("children", [])
                break

        if not children:
            return None

        from ..core import infer_python_type

        lines = []
        for child in children:
            child_name = child.get("name", "")
            member_name = child_name.split(".")[-1]
            cpp_type = child.get("type", "unknown")

            python_type = infer_python_type(cpp_type)

            eval_responses = controller._send_mi_command(f"var-evaluate-expression {child_name}")
            value = "?"
            for resp in eval_responses:
                if resp.get("type") == "result" and resp.get("message") == "done":
                    value = resp.get("payload", {}).get("value", "?")
                    break
            lines.append(f"{member_name} = {value}  ({python_type})")

        return "\n".join(lines) if lines else None
    finally:
        controller._send_mi_command(f"var-delete {actual_var_name}")


def _line_to_function(script_path):
    out = {}
    if not script_path:
        return out
    try:
        import ast

        with open(script_path, "r") as f:
            source = f.read()
        tree = ast.parse(source, script_path)

        def visit(node):
            if isinstance(node, ast.FunctionDef):
                s, e = node.lineno, getattr(node, "end_lineno", node.lineno)
                for ln in range(s, e + 1):
                    out[ln] = node.name
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
    except Exception:
        pass
    return out


def _call_site_for_workunit(script_path, workunit_name, line_to_func):
    if not script_path or not workunit_name:
        return None
    try:
        import ast

        with open(script_path, "r") as f:
            source = f.read()
        tree = ast.parse(source, script_path)
        found: tuple[str, int] | None = None

        def is_parallel_dispatch_call(node):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            return name in ("parallel_for", "parallel_reduce", "parallel_scan")

        def visit(node):
            nonlocal found
            if isinstance(node, ast.Call) and is_parallel_dispatch_call(node.func):
                if len(node.args) >= 3 and isinstance(node.args[2], ast.Name) and node.args[2].id == workunit_name:
                    found = (line_to_func.get(node.lineno, "?"), node.lineno)
                    return
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        return found
    except Exception:
        pass
    return None


def get_backtrace(controller):
    responses = controller._send_mi_command("stack-list-frames")
    if not responses:
        return "No stack frames"
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            stack = resp.get("payload", {}).get("stack", [])
            if not stack:
                return "No stack frames"

            bp_manager = getattr(controller, "breakpoint_manager", None)
            python_frames: list[tuple[str, int, str]] = []
            workunit_name: str | None = None
            script_path: str | None = None
            line_to_func: dict = {}

            for frame in stack:
                file = frame.get("file", frame.get("fullname", ""))
                line_str = frame.get("line", "")
                fullname = frame.get("fullname", file)

                if "functor.hpp" not in (file or "") and "functor.hpp" not in (fullname or ""):
                    continue
                if not bp_manager or not line_str:
                    continue
                try:
                    cpp_line = int(line_str)
                    functor_path = fullname or file
                    try:
                        functor_path_resolved = str(__import__("pathlib").Path(functor_path).resolve())
                    except Exception:
                        functor_path_resolved = functor_path

                    location = bp_manager.get_python_location_for_functor(cpp_line, functor_path_resolved)
                    if location is not None:
                        loc_script_path, loc_python_line = location
                        if loc_script_path and loc_python_line is not None:
                            line_to_func = _line_to_function(loc_script_path)
                            py_func = line_to_func.get(loc_python_line, "?")
                            script_path = loc_script_path
                            workunit_name = py_func
                            python_frames.append((loc_script_path, loc_python_line, py_func))
                            break
                except (ValueError, TypeError):
                    pass

            if not python_frames:
                return "No stack frames"

            call_site = (
                _call_site_for_workunit(script_path, workunit_name, line_to_func)
                if workunit_name and script_path and line_to_func
                else None
            )
            lines = []
            if call_site is not None:
                caller_name, caller_line = call_site[0], call_site[1]
                lines.append(f"#0 {script_path}:{caller_line} ({caller_name})")
            for path, ln, name in python_frames:
                lines.append(f"#{len(lines)} {path}:{ln} ({name})")
            return "\n".join(lines)
    return "No stack frames"


def get_thread_info(controller, thread_id):
    controller._send_mi_command(f"thread-select {thread_id}")
    is_workunit_val = is_in_workunit(controller)
    return {"is_workunit": is_workunit_val}


def thread_select(controller, thread_id: str) -> bool:
    """Select GDB inferior thread and update ``controller.current_thread_id`` on success."""
    responses = controller._send_mi_command(f"thread-select {thread_id}")
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            controller.current_thread_id = thread_id
            return True
    return False


def switch_thread(controller, thread_id, *, quiet: bool = False):
    ok = thread_select(controller, thread_id)
    if ok and not quiet:
        print(f"Switched to thread {thread_id}")
    return ok


def get_current_thread_id(controller):
    responses = controller._send_mi_command("thread-info")
    if not responses:
        return None
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            payload = resp.get("payload", {})
            return payload.get("current-thread-id")
    return None


def get_all_thread_pcs(controller, thread_ids=None):
    """
    Get PC addresses for OpenMP threads (or specified threads) in one call.
    Uses thread-info to get all thread information at once, avoiding thread switching.
    If thread_ids is None, automatically filters to only OpenMP threads.

    Args:
        controller: GDB controller
        thread_ids: Optional list of thread IDs to filter. If None, returns only OpenMP threads.

    Returns:
        Dictionary mapping thread_id -> PC address (as int)
    """
    responses = controller._send_mi_command("thread-info")
    if not responses:
        return {}

    if thread_ids is None:
        thread_ids = get_openmp_threads(controller)

    thread_pcs = {}
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            payload = resp.get("payload", {})
            threads_list = payload.get("threads", [])

            for thread in threads_list:
                thread_id = thread.get("id", "")
                if thread_id not in thread_ids:
                    continue
                frame = thread.get("frame", {})
                addr = frame.get("addr", "")
                if addr:
                    try:
                        pc = int(addr, 16)
                        thread_pcs[thread_id] = pc
                    except ValueError:
                        pass
            break

    return thread_pcs


def _step_all_threads_past_breakpoint(controller, thread_ids):
    """
    Step all threads past a breakpoint when they are all at the same PC.
    Waits for all threads to move and then synchronize.

    Args:
        controller: GDB controller
        thread_ids: List of thread IDs to step
    """
    thread_pcs = get_all_thread_pcs(controller, thread_ids)
    if not thread_pcs or len(set(thread_pcs.values())) != 1:
        return

    thread_list = " ".join(thread_ids)
    initial_pc = next(iter(thread_pcs.values()))
    responses = controller._send_mi_command(f'interpreter-exec console "thread apply {thread_list} nexti"')
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            while True:
                current_pcs = get_all_thread_pcs(controller, thread_ids)
                if not current_pcs:
                    sys.exit("Unable to get thread PC values!")
                if all(pc != initial_pc for pc in current_pcs.values()):
                    break
            _wait_for_threads_synchronized(controller, thread_ids)
            break


def _verify_threads_synchronized(controller, thread_ids):
    """
    Verify that all threads are synchronized (all at the same PC).
    Exits with error if threads are not synchronized.

    Args:
        controller: GDB controller
        thread_ids: List of thread IDs to verify
    """
    final_pcs = get_all_thread_pcs(controller, thread_ids)
    if final_pcs:
        max_pc = max(final_pcs.values())
        if not all(pc == max_pc for pc in final_pcs.values()):
            sys.exit("Threads failed to synchronize!")


def _wait_for_threads_synchronized(controller, thread_ids):
    """
    Poll all threads until their PC addresses are synchronized (all at the same PC).
    Waits indefinitely until all threads are synchronized.

    Args:
        controller: GDB controller
        thread_ids: List of thread IDs to check

    Returns:
        True when threads are synchronized
    """
    while True:
        thread_pcs = get_all_thread_pcs(controller, thread_ids)
        if not thread_pcs:
            sys.exit("Unable to get all threads PC values!")
        max_pc = max(thread_pcs.values())
        if all(pc == max_pc for pc in thread_pcs.values()):
            return True


def synchronize_threads_to_highest_pc(controller, thread_ids):
    """
    Synchronize threads by stepping only threads that are behind the highest PC.
    Continues until all threads are synchronized.

    Args:
        controller: GDB controller
        thread_ids: List of thread IDs to synchronize

    Returns:
        True when threads are synchronized
    """
    if not thread_ids:
        return True

    while True:
        thread_pcs = get_all_thread_pcs(controller, thread_ids)
        if not thread_pcs:
            return True
        max_pc = max(thread_pcs.values())
        if all(pc == max_pc for pc in thread_pcs.values()):
            return True
        behind_threads = [tid for tid, pc in thread_pcs.items() if pc < max_pc]
        behind_list = " ".join(behind_threads)
        responses = controller._send_mi_command(f'interpreter-exec console "thread apply {behind_list} nexti"')
        for resp in responses:
            if resp.get("type") == "result" and resp.get("message") == "done":
                _wait_for_threads_synchronized(controller, thread_ids)
                break


def parse_thread_range(args, default_threads):
    """
    Parse thread range arguments or return default thread list.

    Args:
        args: Command arguments string (e.g., "10-12" or empty string)
        default_threads: List of default thread IDs to use if args is empty

    Returns:
        List of thread IDs as strings, or None if parsing failed
    """
    if not args:
        return default_threads

    # Parse thread range like "continue 10-12", so we will continue only part of
    # threads
    parts = args.strip().split("-")
    if len(parts) == 2:
        try:
            thread_begin = int(parts[0])
            thread_end = int(parts[1])
            if thread_begin > thread_end:
                print(f"Error: Invalid range {thread_begin}-{thread_end}")
                return None
            return [str(i) for i in range(thread_begin, thread_end + 1)]
        except ValueError:
            print(f"Error: Invalid thread range format. Use: continue <begin>-<end>")
            return None
    else:
        print(f"Error: Invalid format. Use: continue <begin>-<end>")
        return None


def get_openmp_threads(controller):
    """Get thread IDs that are executing OpenMP code (stopped at functor.hpp)"""
    responses = controller._send_mi_command("thread-info")
    if not responses:
        return []

    openmp_threads = []
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            payload = resp.get("payload", {})
            threads_list = payload.get("threads", [])

            for thread in threads_list:
                thread_id = thread.get("id", "")
                frame = thread.get("frame", {})
                fullname = frame.get("fullname", "")
                file = frame.get("file", "")

                # Include threads that are stopped at functor.hpp
                if "functor.hpp" in fullname or "functor.hpp" in file:
                    openmp_threads.append(thread_id)

            break

    return sorted(openmp_threads, key=lambda x: int(x))


def get_threads(controller):
    responses = controller._send_mi_command("thread-info")
    if not responses:
        return "No threads"

    current_id = None
    threads_list = []

    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            payload = resp.get("payload", {})
            current_id = payload.get("current-thread-id")
            threads_list = payload.get("threads", [])
            break

    if not threads_list:
        return "No threads"

    workunit_threads = []
    system_threads = []

    for thread in threads_list:
        thread_id = thread.get("id", "?")
        target_id = thread.get("target-id", "unknown")
        state = thread.get("state", "unknown")

        frame = thread.get("frame", {})
        file_path = frame.get("file", frame.get("fullname", ""))
        is_workunit_val = file_path.endswith("@functor.hpp") or file_path.endswith("functor.hpp")

        marker = "*" if thread_id == current_id else " "
        thread_info = (thread_id, target_id, state, marker)

        if is_workunit_val:
            workunit_threads.append(thread_info)
        else:
            system_threads.append(thread_info)

    workunit_threads.sort(key=lambda x: int(x[0]))
    system_threads.sort(key=lambda x: int(x[0]))

    lines = []
    if workunit_threads:
        lines.append("Workunit Threads:")
        lines.extend(f"  {marker} Thread {tid} ({name}) - {state}" for tid, name, state, marker in workunit_threads)
    if system_threads:
        if workunit_threads:
            lines.append("")
        lines.append("System Threads:")
        lines.extend(f"  {marker} Thread {tid} ({name}) - {state}" for tid, name, state, marker in system_threads)

    return "\n".join(lines) if lines else "No threads"
