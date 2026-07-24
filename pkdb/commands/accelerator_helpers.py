"""
Accelerator-specific (CUDA/HIP) helper functions for GDB debugging
"""

import os
import re
from typing import Any, Optional

# cuda-gdb / rocgdb "info {cuda|hip} threads" data rows (PC + file + line omitted when formatting).
_ACCEL_THREAD_ROW_RE = re.compile(
    r"^(?P<current>\*)?\s*"
    r"\((?P<b0>\d+),(?P<b1>\d+),(?P<b2>\d+)\)\s+"
    r"\((?P<f0>\d+),(?P<f1>\d+),(?P<f2>\d+)\)\s+"
    r"\((?P<tb0>\d+),(?P<tb1>\d+),(?P<tb2>\d+)\)\s+"
    r"\((?P<tt0>\d+),(?P<tt1>\d+),(?P<tt2>\d+)\)\s+"
    r"(?P<count>\d+)\s+"
    r"0x[0-9a-fA-F]+"
)

_KERNEL_HEADER_RE = re.compile(r"^Kernel\s+(\d+)\s*$", re.IGNORECASE)


def collect_console_stream_output(responses: list[Any]) -> str:
    """Join MI `console` stream payloads from `_send_mi_command` results."""
    parts: list[str] = []
    for resp in responses:
        if resp.get("type") == "console":
            out = resp.get("payload", "")
            if out:
                parts.append(out)
    return "".join(parts)


def format_info_accelerator_threads_summary(
    raw_output: str,
    *,
    script_path: Optional[str] = None,
    python_line: Optional[int] = None,
) -> Optional[str]:
    """
    Parse `info cuda threads` / `info hip threads` text and print block/thread ranges
    and counts without PC or device source path; optionally append mapped Python file:line.

    Returns formatted text, or None if the output does not look like a threads table (caller
    should print raw GDB text).
    """
    text = raw_output.strip()
    if not text:
        return None

    lines_out: list[str] = []
    saw_kernel_or_row = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "BlockIdx" in stripped and "ThreadIdx" in stripped and "Count" in stripped:
            continue

        km = _KERNEL_HEADER_RE.match(stripped)
        if km:
            saw_kernel_or_row = True
            lines_out.append(f"Kernel {km.group(1)}")
            continue

        m = _ACCEL_THREAD_ROW_RE.match(stripped)
        if m:
            saw_kernel_or_row = True
            cur = m.group("current") == "*"
            block = f"({m.group('b0')},{m.group('b1')},{m.group('b2')})"
            t_from = f"({m.group('f0')},{m.group('f1')},{m.group('f2')})"
            to_block = f"({m.group('tb0')},{m.group('tb1')},{m.group('tb2')})"
            t_to = f"({m.group('tt0')},{m.group('tt1')},{m.group('tt2')})"
            count = m.group("count")
            prefix = "* " if cur else "  "
            lines_out.append(
                f"{prefix}"
                f"block_idx={block}  thread_idx={t_from}..{t_to}  "
                f"to_block_idx={to_block}  count={count}"
            )
            continue

    if not saw_kernel_or_row:
        return None

    if script_path and python_line is not None:
        lines_out.append(f"Python: {os.path.basename(script_path)}:{python_line}")

    return "\n".join(lines_out)


def synchronize_accelerator_blocks_at_breakpoint(controller, accelerator_prefix):
    """
    Synchronize all CUDA/HIP thread blocks at a breakpoint.
    Continues execution until all blocks that will hit this breakpoint have reached it.

    This prevents multiple stops when different thread blocks hit the same breakpoint
    at different times during kernel execution.

    Args:
        controller: Accelerator GDB controller
        accelerator_prefix: 'cuda' or 'hip'

    Returns:
        True if synchronized, False otherwise
    """
    # Get current kernel and breakpoint location
    responses = controller._send_mi_command(
        'interpreter-exec console "info cuda kernels"'
        if accelerator_prefix == "cuda"
        else 'interpreter-exec console "info hip kernels"'
    )

    # Check if we're in a kernel context
    in_kernel = False
    for resp in responses:
        if resp.get("type") == "console":
            output = resp.get("payload", "")
            if "Kernel" in output:
                in_kernel = True
                break

    if not in_kernel:
        return False

    initial_thread_info = _get_accelerator_thread_info(controller, accelerator_prefix)
    if not initial_thread_info:
        return False

    max_attempts = 10  # Prevent infinite loops
    for _ in range(max_attempts):
        controller._send_mi_command("exec-continue")

        import queue

        try:
            parsed = controller.output_queue.get(timeout=5.0)
            msg_type = parsed.get("type")
            message = parsed.get("message")
            payload = parsed.get("payload", {})

            if msg_type == "notify" and message == "stopped":
                reason = payload.get("reason")

                if reason == "breakpoint-hit":
                    current_thread_info = _get_accelerator_thread_info(controller, accelerator_prefix)
                    if current_thread_info and _is_same_breakpoint_location(initial_thread_info, current_thread_info):
                        continue
                    else:
                        return True
                else:
                    return True
        except queue.Empty:
            return False

    return True


def _get_accelerator_thread_info(controller, accelerator_prefix):
    responses = controller._send_mi_command("stack-list-frames 0 0")
    for resp in responses:
        if resp.get("type") == "result" and resp.get("message") == "done":
            payload = resp.get("payload", {})
            stack = payload.get("stack", [])
            if stack:
                frame = stack[0]
                return {
                    "file": frame.get("fullname", frame.get("file", "")),
                    "line": frame.get("line", ""),
                    "func": frame.get("func", ""),
                }
    return None


def _is_same_breakpoint_location(info1, info2):
    if not info1 or not info2:
        return False

    same_file = info1.get("file") == info2.get("file")
    same_line = info1.get("line") == info2.get("line")

    return same_file and same_line
