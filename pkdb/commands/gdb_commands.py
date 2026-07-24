"""
GDB commands handler. Used by `pkdb/controllers/gdb_controller.py`
Used as main command interface provider, same to `gdb` implementation
"""

from .gdb_helpers import (
    get_current_python_location,
    get_thread_counts,
    is_in_workunit,
    get_locals,
    get_class_members,
    get_backtrace,
    get_threads,
    get_thread_info,
    get_openmp_threads,
    get_current_thread_id,
    thread_select,
    switch_thread,
    synchronize_threads_to_highest_pc,
    parse_thread_range,
    _step_all_threads_past_breakpoint,
    _verify_threads_synchronized,
    get_frame_info,
)
from pathlib import Path
from typing import Optional

from ..core.debug_properties import verbose_out


def _parse_lineno(args):
    """Parse and validate line number from args. Returns lineno or None."""
    try:
        lineno = int(args.strip().split()[0])
        return lineno if lineno > 0 else None
    except (ValueError, IndexError, AttributeError):
        return None


def extract_script_functions(script_path):
    """
    Extract function definitions from a script file.

    Returns a dictionary mapping function names to callable function objects.
    This is done by parsing the AST and compiling only function definitions,
    avoiding execution of top-level code that might have side effects.
    """
    script_globals = {}

    if not script_path:
        return script_globals

    import ast

    # Read and parse the script
    with open(script_path, "r") as f:
        source = f.read()
    tree = ast.parse(source, script_path)
    namespace = {}

    # Collect imports to provide dependencies for functions
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                try:
                    module = __import__(alias.name)
                    namespace[alias.asname or alias.name] = module
                except:
                    pass
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                try:
                    module = __import__(node.module, fromlist=[n.name for n in node.names])
                    for alias in node.names:
                        namespace[alias.asname or alias.name] = getattr(module, alias.name)
                except:
                    pass

    # Compile and execute only function definitions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_module = ast.Module(body=[node], type_ignores=[])
            code = compile(func_module, script_path, "exec")
            exec(code, namespace)
            script_globals[node.name] = namespace[node.name]

    return script_globals


def extract_script_variables(script_path):
    if not script_path:
        return {}
    out = {}
    try:
        import ast

        with open(script_path, "r") as f:
            source = f.read()
        tree = ast.parse(source, script_path)
        names = {}

        def resolve(node):
            if node is None:
                return None
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return names.get(node.id)
            if isinstance(node, ast.List):
                return [resolve(e) for e in node.elts]
            if isinstance(node, ast.Tuple):
                return tuple(resolve(e) for e in node.elts)
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name):
                    val = resolve(node.value)
                    if val is not None and not callable(val):
                        names[t.id] = val

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if not isinstance(t, ast.Name):
                    continue
                val = node.value
                dims = None
                if isinstance(val, ast.Call):
                    if isinstance(val.func, ast.Name) and val.func.id == "View" and val.args:
                        dims = resolve(val.args[0])
                    elif isinstance(val.func, ast.Attribute):
                        if val.func.attr == "View" and val.args:
                            dims = resolve(val.args[0])
                        elif val.func.attr == "zeros" and val.args:
                            d = resolve(val.args[0])
                            dims = [d] if d is not None else None
                if dims is not None:
                    if isinstance(dims, (list, tuple)):
                        resolved = []
                        for x in dims:
                            if isinstance(x, (int, type(None))):
                                if x is None:
                                    break
                                resolved.append(x)
                            else:
                                break
                        else:
                            if resolved:
                                out[t.id] = resolved
    except Exception:
        pass
    return out


def _collect_script_globals(controller):
    """
    Collect function definitions from the script being debugged.

    Needed for evaluators. For example, if we do "eval <python_function>"

    Returns the pre-extracted functions stored in the controller.
    """
    return getattr(controller, "script_globals", {})


# specific behavior for last-line breakpoint inside of workunit, so we need to know that
def _at_last_active_breakpoint(controller) -> bool:
    if not getattr(controller, "_attached_mode", False):
        return False
    current_line = getattr(controller, "current_breakpoint_line", None)
    if current_line is None:
        return False
    active_bps = controller.breakpoint_manager.get_breakpoints()
    return not active_bps or current_line >= max(active_bps)


def _synchronize_and_continue(controller, thread_ids):
    """
    Synchronize threads and continue execution.
    Shared logic for both continue and step commands.

    Args:
        controller: GDB controller
        thread_ids: List of thread IDs to synchronize and continue
    """
    controller._send_mi_command("gdb-set scheduler-locking on")
    _step_all_threads_past_breakpoint(controller, thread_ids)
    synchronize_threads_to_highest_pc(controller, thread_ids)
    _verify_threads_synchronized(controller, thread_ids)
    controller._send_mi_command("gdb-set scheduler-locking off")

    if _at_last_active_breakpoint(controller):
        verbose_out("last workunit breakpoint reached; detaching back to pdb")
        controller.stop()
        raise StopIteration

    controller._send_mi_command("exec-continue")
    raise StopIteration


def register_gdb_commands(controller):
    cmd = controller.command

    @cmd("b", "break")
    def break_cmd(controller, args):
        """Set a breakpoint at the specified line number: break <line_number>"""
        lineno = _parse_lineno(args)
        if lineno is None:
            print(f"Usage: break <line_number>")
            return

        script = getattr(controller, "script_path", "") or ""
        file_path: Optional[str] = None
        if script:
            try:
                file_path = str(Path(script).resolve())
            except OSError:
                file_path = script
        is_new = controller.breakpoint_manager.add_breakpoint(lineno, file_path=file_path)
        msg = f"Breakpoint set at line {lineno}" if is_new else f"Breakpoint at line {lineno} already exists"
        print(msg)

    @cmd("d", "delete")
    def delete_cmd(controller, args):
        """Delete a breakpoint: delete <line_number> or delete all"""
        if not args:
            print("Usage: delete <line_number> or delete all")
            return

        # 'delete all'
        if args.strip().lower() == "all":
            count = len(controller.breakpoint_manager.get_breakpoints())
            controller.breakpoint_manager.clear_breakpoints()
            print(f"Deleted all {count} breakpoint(s)" if count > 0 else "No breakpoints to delete")
            return

        # 'delete <line_number>'
        lineno = _parse_lineno(args)
        if lineno is None:
            print("Usage: delete <line_number> or delete all")
            return

        is_deleted = controller.breakpoint_manager.remove_breakpoint(lineno)
        msg = f"Breakpoint at line {lineno} deleted" if is_deleted else f"No breakpoint at line {lineno}"
        print(msg)

    @cmd("breakpoints")
    def breakpoints_cmd(controller, args):
        """List all current breakpoints"""
        breakpoints = controller.breakpoint_manager.get_breakpoints()
        if not breakpoints:
            print("No breakpoints set")
            return

        print(f"Current breakpoints ({len(breakpoints)}):")
        for lineno in sorted(breakpoints):
            print(f"  Line {lineno}")

    @cmd("c", "continue")
    def continue_cmd(controller, args):
        """Continue execution: continue or continue <thread_begin>-<thread_end>"""
        # change mode (stepping -> continue)
        if controller.breakpoint_manager.is_step_mode():
            controller.breakpoint_manager.exit_step_mode()

        thread_ids = parse_thread_range(args, controller.openmp_threads)
        if thread_ids is None or not thread_ids:
            print("No threads to continue")
            return

        _synchronize_and_continue(controller, thread_ids)

    @cmd("s", "step")
    def step_cmd(controller, args):
        """Step to next line in workunit: activates all LOCs and continues"""
        all_locs = controller.breakpoint_manager.get_all_locs()
        if not all_locs:
            print("No step locations available")
            return

        controller.breakpoint_manager.set_continue_active(all_locs)
        controller.breakpoint_manager.enter_step_mode()

        thread_ids = parse_thread_range(args, controller.openmp_threads)
        if thread_ids is None or not thread_ids:
            print("No threads to step")
            return

        _synchronize_and_continue(controller, thread_ids)

    @cmd("q", "quit")
    def quit_cmd(controller, args):
        controller.running = False
        controller.stop()
        raise StopIteration

    @cmd("p", "print")
    def print_cmd(controller, args):
        if not args:
            print("Usage: print <expression>")
            return

        from ..evaluators.expression_evaluator import ExpressionEvaluator
        from ..evaluators.context import EvaluationContext, ContextType
        from ..controllers.accelerator_gdb_controller import AcceleratorGDBController

        if isinstance(controller, AcceleratorGDBController):
            if controller.accelerator_type == "hip":
                context_type = ContextType.HIP
            else:
                context_type = ContextType.CUDA
        else:
            context_type = ContextType.GDB

        script_globals = _collect_script_globals(controller)
        context = EvaluationContext(context_type, controller, script_globals=script_globals)
        evaluator = ExpressionEvaluator(context)
        result = evaluator.evaluate(args)

        if result.success:
            print(f"{args} = {result.value}")
            if result.type_info:
                print(f"  (type: {result.type_info})")
        else:
            print(f"Error: {result.error}")

    @cmd("eval")
    def eval_cmd(controller, args):
        """Evaluate expression: eval <expression>"""
        if not args:
            print("Usage: eval <expression>")
            return

        try:
            from ..evaluators.expression_evaluator import ExpressionEvaluator
            from ..evaluators.context import EvaluationContext, ContextType
            from ..controllers.accelerator_gdb_controller import AcceleratorGDBController

            if isinstance(controller, AcceleratorGDBController):
                if controller.accelerator_type == "hip":
                    context_type = ContextType.HIP
                else:
                    context_type = ContextType.CUDA
            else:
                context_type = ContextType.GDB

            # Collect script globals (functions from the script being debugged)
            script_globals = _collect_script_globals(controller)
            context = EvaluationContext(context_type, controller, script_globals=script_globals)
            evaluator = ExpressionEvaluator(context)
            result = evaluator.evaluate(args)

            if result.success:
                print(f"{args} = {result.value}")
                if result.type_info:
                    print(f"  (type: {result.type_info})")
            else:
                print(f"Error: {result.error}")

        except Exception as e:
            print(f"Error evaluating expression: {e}")
            import traceback

            verbose_out(f"Exception details: { traceback.print_exc()}")

    @cmd("parallel_print")
    def parallel_print_cmd(controller, args):
        """OpenMP only: evaluate an expression on each workunit thread; print tid | value."""
        expr = args.strip() if args else ""
        if not expr:
            print("Usage: parallel_print <expression>")
            print("  (OpenMP / DebugOpenMP only; same expressions as 'print')")
            return

        from ..controllers.accelerator_gdb_controller import AcceleratorGDBController
        from ..evaluators.expression_evaluator import ExpressionEvaluator
        from ..evaluators.context import EvaluationContext, ContextType

        if isinstance(controller, AcceleratorGDBController):
            print("parallel_print is only supported for OpenMP (CPU) debugging, not CUDA/HIP.")
            return

        omp_threads = get_openmp_threads(controller)
        if not omp_threads:
            print("No OpenMP workunit threads found (expected threads stopped in functor.hpp).")
            print("Use 'threads' to list threads, or ensure you are stopped in a workunit.")
            return

        orig = get_current_thread_id(controller)
        rows: list[tuple[int, str]] = []
        try:
            script_globals = _collect_script_globals(controller)
            context = EvaluationContext(ContextType.GDB, controller, script_globals=script_globals)
            evaluator = ExpressionEvaluator(context)

            for tid in range(len(omp_threads)):
                gdb_tid = omp_threads[tid]
                if not thread_select(controller, gdb_tid):
                    rows.append((tid, f"<failed to select GDB thread {gdb_tid}>"))
                    continue
                result = evaluator.evaluate(expr)
                if result.success:
                    rows.append((tid, str(result.value)))
                else:
                    rows.append((tid, f"<error: {result.error}>"))
        finally:
            if orig:
                thread_select(controller, orig)

        label = expr.replace("\n", " ").strip()
        header = f"{'tid':<4}| {label}"
        print(header)
        print("-" * len(header))
        for tid, text in rows:
            print(f"{tid:<4}| {text}")

    @cmd("locals")
    def locals_cmd(controller, args):
        is_workunit_val = is_in_workunit(controller)
        output = get_locals(controller)
        print(output)

        # show class members (e.g., arrays converted to class members, not to
        # local variables)
        members = get_class_members(controller)
        if members and members != "No members found":
            print(members)

        if not is_workunit_val:
            print("\n  Note: This thread is not in a workunit context.")
            print("  Use 'threads' to see which threads are workunit threads.")

    @cmd("bt", "backtrace", "where")
    def backtrace_cmd(controller, args):
        output = get_backtrace(controller)
        print(output)

    @cmd("whereami")
    def whereami_cmd(controller, args):
        """Show current location (Python file:line), current thread, and number of active threads"""
        file_path, python_line = get_current_python_location(controller)
        current_id, total_count, workunit_count = get_thread_counts(controller)

        has_python_loc = file_path is not None and python_line is not None
        frame_info = None if has_python_loc else get_frame_info(controller)
        location = f"{file_path}:{python_line}" if has_python_loc else (frame_info or "unknown")
        if not has_python_loc and frame_info:
            location = f"{location} (C++; not in Python workunit)"

        thread_str = str(current_id) if current_id is not None else "unknown"
        active_str = (
            "0"
            if total_count == 0
            else f"{workunit_count} workunit, {total_count} total" if workunit_count > 0 else str(total_count)
        )

        print(f"Location: {location}")
        print(f"Current thread: {thread_str}")
        print(f"Active threads: {active_str}")

    @cmd("l", "list")
    def list_cmd(controller, args):
        """List Python source around the current workunit line: list [line]"""
        # Accelerator controllers resolve location via CLI with functor mapping.
        loc_fn = getattr(controller, "_get_current_location_via_cli", None)
        if loc_fn is not None:
            file_path, python_line, _frame = loc_fn()
        else:
            file_path, python_line = get_current_python_location(controller)

        if not file_path:
            file_path = getattr(controller, "script_path", "") or ""

        center = _parse_lineno(args) if args.strip() else python_line
        if args.strip() and center is None:
            print("Usage: list [line_number]")
            return
        if not file_path or center is None:
            print("No Python source location available")
            return

        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
        except OSError as e:
            print(f"Cannot read {file_path}: {e}")
            return

        start = max(1, center - 5)
        end = min(len(lines), center + 5)
        for n in range(start, end + 1):
            marker = "->" if n == python_line else "  "
            print(f"{n:4d} {marker} {lines[n - 1].rstrip()}")

    @cmd("threads")
    def threads_cmd(controller, args):
        output = get_threads(controller)
        print(output)

    @cmd("thread")
    def thread_cmd(controller, args):
        if not args:
            print("Usage: thread <num>")
            return
        thread_id = args.split()[0]
        thread_info = get_thread_info(controller, thread_id)
        switch_thread(controller, thread_id)
        if thread_info and not thread_info.get("is_workunit", False):
            print(f"  Note: Thread {thread_id} is not a workunit thread")
        elif thread_info and thread_info.get("is_workunit", False):
            print("  Workunit thread")

    @cmd("send")
    def send_cmd(controller, args):
        if not args:
            print("Usage: send <MI-command>")
            return
        responses = controller._send_mi_command(args)
        for resp in responses:
            print(resp)

    @cmd("info")
    def info_cmd(controller, args):
        if not args:
            print("Usage: info <command>")
            return
        if args == "set":
            # Show all debug properties
            props = controller.debug_properties.list_properties()
            print("Debug properties:")
            for name, value in sorted(props.items()):
                if isinstance(value, bool):
                    shown = "on" if value else "off"
                else:
                    shown = value
                print(f"  {name}: {shown}")
            return
        full_cmd = f"info {args}"
        responses = controller._send_mi_command(f'interpreter-exec console "{full_cmd}"')
        for resp in responses:
            if resp.get("type") == "console":
                output = resp.get("payload", "")
                if output:
                    print(output.strip())

    @cmd("h", "help")
    def help_cmd(controller, args):
        if args:
            if args == "info":
                # show available info commands
                print("Available info commands:")
                print("  info threads        - List all threads")
                print("  info breakpoints    - List all breakpoints")
                print("  info locals         - List local variables")
                print("  info args           - List function arguments")
                print("  info registers      - Show register values")
                print("  info frame          - Show current stack frame")
                print("  info set            - Show all debug properties")
                if hasattr(controller, "accelerator_type"):
                    accel = controller.accelerator_type
                    print(f"  info {accel} threads - Show all {accel.upper()} threads")
                print("\nUse 'info <command>' to execute")
            else:
                handler = controller.commands.get(args)
                if handler and handler.__doc__:
                    print(f"{args}: {handler.__doc__}")
                else:
                    print(f"No help available for: {args}")
        else:
            # dynamically gather commands from the controller
            print("\nAvailable Commands:")
            print("-" * 70)
            cmd_groups = {}
            for cmd_name, handler in controller.commands.items():
                if handler not in cmd_groups:
                    cmd_groups[handler] = []
                cmd_groups[handler].append(cmd_name)
            for handler, cmd_names in sorted(cmd_groups.items(), key=lambda x: min(x[1])):
                aliases = ", ".join(sorted(cmd_names))
                desc = handler.__doc__.strip() if handler.__doc__ else "No description"
                print(f"  {aliases:30} - {desc}")

    @cmd("set")
    def set_cmd(controller, args):
        """Set debugger properties: set <property> <on|off>"""
        if not args:
            # List all properties and their values
            props = controller.debug_properties.list_properties()
            print("Debug properties:")
            for name, value in sorted(props.items()):
                if isinstance(value, bool):
                    shown = "on" if value else "off"
                else:
                    shown = value
                print(f"  {name}: {shown}")
            return

        parts = args.strip().split(maxsplit=1)
        if len(parts) == 1:
            # Show specific property value
            prop_name = parts[0]
            value = controller.debug_properties.get_property(prop_name)
            if isinstance(value, bool):
                shown = "on" if value else "off"
            else:
                shown = value
            print(f"{prop_name}: {shown}")
            return

        prop_name = parts[0]
        prop_value_raw = parts[1].strip()

        current_props = controller.debug_properties.list_properties()
        if prop_name not in current_props:
            print(f"Unknown property: {prop_name}")
            print("\nAvailable properties:")
            for name in sorted(current_props.keys()):
                print(f"  {name}")
            return

        current_value = current_props[prop_name]
        if isinstance(current_value, bool):
            prop_value = prop_value_raw.lower()
            if prop_value in ["on", "1", "true"]:
                value = True
            elif prop_value in ["off", "0", "false"]:
                value = False
            else:
                print(f"Invalid value: {prop_value_raw}")
                print("Usage: set <property> <on|off>")
                return
        elif isinstance(current_value, int):
            try:
                value = int(prop_value_raw)
                if value <= 0:
                    raise ValueError
            except ValueError:
                print(f"Invalid value: {prop_value_raw}")
                print("Usage: set <property> <positive-integer>")
                return
        else:
            value = prop_value_raw

        # Set property
        if controller.debug_properties.set_property(prop_name, value):
            if isinstance(value, bool):
                shown = "on" if value else "off"
            else:
                shown = value
            print(f"{prop_name}: {shown}")
        else:
            print(f"Unknown property: {prop_name}")
            print("\nAvailable properties:")
            props = controller.debug_properties.list_properties()
            for name in sorted(props.keys()):
                print(f"  {name}")

    @cmd("len")
    def len_cmd(controller, args):
        if not args:
            print("Usage: len <variable>")
            return
        var_name = args.strip().split()[0]
        script_vars = getattr(controller, "script_variables", {})
        if var_name not in script_vars:
            print(f"Unknown variable: {var_name}")
            return
        dims = script_vars[var_name]
        print(dims[0])
