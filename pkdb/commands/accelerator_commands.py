from typing import Callable, Optional, Tuple, cast

from .gdb_commands import register_gdb_commands
from .accelerator_helpers import (
    collect_console_stream_output,
    format_info_accelerator_threads_summary,
)
from ..analyzers.performance_analyzer import PerformanceAnalyzer

_GetPythonLocFn = Callable[[], Tuple[Optional[str], Optional[int], Optional[str]]]


def register_accelerator_commands(controller, accelerator_prefix):
    register_gdb_commands(controller)

    cmd = controller.command

    @cmd(accelerator_prefix)
    def accelerator_cmd(controller, args):
        if args.startswith("thread "):
            parts = args[7:].split()
            if len(parts) != 2:
                print(f"Usage: {accelerator_prefix} thread <block> <thread>")
                return
            try:
                block = int(parts[0])
                thread = int(parts[1])
                responses = controller._send_mi_command(
                    f'interpreter-exec console "{accelerator_prefix} thread ({block},{thread})"'
                )
                for resp in responses:
                    if resp.get("type") == "console":
                        output = resp.get("payload", "")
                        if output:
                            print(output.strip())
            except ValueError:
                print("Error: block and thread must be integers")
        else:
            print(f"Unknown {accelerator_prefix} command: {args}")

    # Registered by register_gdb_commands above; used as fallback for other info args.
    generic_info_cmd = controller.commands["info"]

    @cmd("info")
    def info_accelerator_cmd(controller, args):
        """Run an info command; 'info <cuda|hip> threads' shows a thread summary"""
        if args.split() != [accelerator_prefix, "threads"]:
            generic_info_cmd(controller, args)
            return
        responses = controller._send_mi_command(f'interpreter-exec console "info {accelerator_prefix} threads"')
        raw = collect_console_stream_output(responses)
        script_path = None
        python_line = None
        loc_fn = cast(
            Optional[_GetPythonLocFn],
            getattr(controller, "_get_current_location_via_cli", None),
        )
        if loc_fn is not None:
            script_path, python_line, _frame_str = loc_fn()
        summary = format_info_accelerator_threads_summary(
            raw,
            script_path=script_path,
            python_line=python_line,
        )
        if summary is not None:
            print(summary)
        elif raw.strip():
            print(raw.strip())

    @cmd("c", "continue")
    def continue_cmd(controller, args):
        """Continue execution, synchronizing all thread blocks at breakpoints"""
        # change mode (stepping -> continue)
        if controller.breakpoint_manager.is_step_mode():
            controller.breakpoint_manager.exit_step_mode()

        controller._send_mi_command("exec-continue", wait_for_prompt=False)
        raise StopIteration

    @cmd("s", "step")
    def step_cmd(controller, args):
        """Step to next line in workunit: activates all LOCs and continues"""
        # change mode (continue -> stepping)
        if controller.breakpoint_manager.get_all_locs():
            controller.breakpoint_manager.enter_step_mode()

        controller._send_mi_command("exec-continue", wait_for_prompt=False)
        raise StopIteration

    perf_analyzer = PerformanceAnalyzer(controller)

    @cmd("divergence", "div")
    def divergence_cmd(controller, args):
        """Analyze thread divergence in current warp state"""
        analysis = perf_analyzer.divergence.analyze_current_state()
        print(perf_analyzer.divergence.format_report(analysis))

    @cmd("locals")
    def locals_cmd(controller, args):
        """Show local variables at current location"""
        responses = controller._send_mi_command('interpreter-exec console "info locals"')
        for resp in responses:
            if resp.get("type") == "console":
                output = resp.get("payload", "")
                if output:
                    print(output.strip())

    @cmd("args")
    def args_cmd(controller, args):
        """Show function arguments at current location"""
        responses = controller._send_mi_command('interpreter-exec console "info args"')
        for resp in responses:
            if resp.get("type") == "console":
                output = resp.get("payload", "")
                if output:
                    print(output.strip())
