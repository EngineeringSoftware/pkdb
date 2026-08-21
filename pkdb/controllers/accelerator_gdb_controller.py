import os
import re
import select
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pygdbmi import gdbmiparser

from .gdb_controller import GDBController
from ..commands import register_accelerator_commands
from ..core import native_debuggers
from ..core.debug_properties import verbose_out

# GDB annotate=3: annotation lines start with two Ctrl-Z (0x1a) then name
ANNOTATION_PREFIX = b"\x1a\x1a"
_ANNOTATION_ESCAPE_RE = re.compile("\x1a\x1a[^\n]*\n?")

# Flow-dependent regexes (execution is interrupted or done)
_FAULT_SIGNAL_RE = re.compile(r"\bSIG(SEGV|ABRT|BUS|FPE|ILL)\b")
_EXEC_CONTINUE_ACCEPTED_RE = re.compile(r"\d+\^(?:running|done)\b")

_ACCELERATOR_GDB_ALERT_SUBSTRINGS = ("error", "exception", "signal")


def _mi_payload_text(text: str) -> Optional[str]:
    """Unescaped payload of an MI record (e.g. break-insert's own log/console output)."""
    try:
        parsed = gdbmiparser.parse_response(text)
    except Exception:
        return None
    payload = parsed.get("payload") if parsed else None
    return payload if isinstance(payload, str) else None


def _accelerator_gdb_text_suggests_problem(text: str) -> bool:
    if not text or not text.strip():
        return False
    checked = _mi_payload_text(text)
    if checked is None:
        checked = text
    stripped = _ANNOTATION_ESCAPE_RE.sub("", checked).strip()
    if not stripped:
        return False
    tl = stripped.lower()
    return any(s in tl for s in _ACCELERATOR_GDB_ALERT_SUBSTRINGS)


def _escape_for_gdb_cli_double_quotes(s: str) -> str:
    """Escape s for embedding in a GDB CLI double-quoted argument (interpreter-exec mi3 \"...\")."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


# Shared GDB controller for accelerator debuggers (CUDA and HIP)
# Supports both cuda-gdb and rocgdb with shared command interface
class AcceleratorGDBController(GDBController):
    def __init__(
        self,
        args,
        accelerator_type="cuda",
        breakpoints=None,
        all_locs=None,
        script_path="",
        script_args=None,
        script_globals=None,
        script_variables=None,
        breakpoint_manager=None,
        cwd=None,
        handoff_breakpoint_spec: Optional[Dict[str, Any]] = None,
        bp_ack_write_fd: Optional[int] = None,
    ):
        super().__init__(
            args,
            breakpoints=breakpoints,
            all_locs=all_locs,
            script_path=script_path,
            script_args=script_args,
            script_globals=script_globals,
            script_variables=script_variables,
            breakpoint_manager=breakpoint_manager,
            cwd=cwd,
            handoff_breakpoint_spec=handoff_breakpoint_spec,
            bp_ack_write_fd=bp_ack_write_fd,
        )
        self.accelerator_type = accelerator_type.lower()
        self.command_prompt = f"(pkdb-{self.accelerator_type}) "

        if self.accelerator_type == "hip":
            self.accelerator_prefix = "hip"
            # TODO: fix to MI3 later, since rocgdb supports mi3 mode, and it's more stable
            self.command_execution_mode = "annotate"
            register_accelerator_commands(self, self.accelerator_prefix)
        else:  # default to cuda
            self.accelerator_type = "cuda"
            self.accelerator_prefix = "cuda"
            self.command_execution_mode = "annotate"
            register_accelerator_commands(self, self.accelerator_prefix)

        self.stepping_out_of_brkpt = False
        self.breakpoints_synced = False  # Track if we've synced breakpoints
        self._pending_resume_after_attach = False
        self._pending_resume_fault_seen = False
        # Settle-stop tracking; see _track_attach_state().
        self._attach_signal_seen = False
        self._attach_stop_seen = False
        # True once the target exited; stop() then skips the detach.
        self._inferior_exited = False

    def _emit_accelerator_gdb_alert_if_needed(self, text: Optional[str]) -> None:
        """
        If GDB reports an error, exception, or signal, print it even when verbose is off.
        When verbose is on, traffic is already logged via verbose_out.
        """
        if not text or not text.strip():
            return
        if not self.running:
            return
        if not _accelerator_gdb_text_suggests_problem(text):
            return
        if self.debug_properties.verbose:
            print(
                f"[pkdb-{self.accelerator_type}] accelerator debugger: {text.rstrip()}",
                flush=True,
            )

    def _readline_gdb(self) -> Tuple[Optional[str], bool, Optional[str]]:  # type: ignore[override]
        """
        Read one line from GDB (annotate=3 mode).
        Returns (text to display, saw_prompt, annotation_name or None).
        Only send "continue" when we've seen the "stopped" annotation before prompt (avoids double-continue skipping breakpoint).
        """
        if not hasattr(self, "_gdb_master_stream") or self._gdb_master_stream is None:
            return (None, False, None)
        try:
            self._gdb_master_stream.flush()
            line = self._gdb_master_stream.readline()
        except (OSError, ValueError):
            return (None, False, None)
        if not line:
            return (None, False, None)
        if line.startswith(ANNOTATION_PREFIX):
            ann = line[len(ANNOTATION_PREFIX) :].decode("ascii", errors="replace").strip()
            return ("", ann == "prompt", ann)
        decoded = line.decode("utf-8", errors="replace")

        # Skip lines that are only Ctrl-Z/annotation bytes (PTY can split ^Z^Zprompt into separate lines)
        if decoded.strip() and not all(ord(c) == 0x1A for c in decoded.strip()):
            return (decoded, False, None)
        return ("", False, None)

    def _flush_pending_output(self) -> None:
        """
        Best-effort flush of any pending cuda-gdb output so that subsequent
        calls to _send_gdb_command() see only output for the newest command.
        """
        if not self.process or self.process.poll() is not None:
            return
        if not hasattr(self, "_gdb_master_fd") or self._gdb_master_fd is None:
            return
        while True:
            ready, _, _ = select.select([self._gdb_master_fd], [], [], 0)
            if not ready:
                break
            text, saw_prompt, _ann = self._readline_gdb()
            if text:
                self._emit_accelerator_gdb_alert_if_needed(text)
            self._track_attach_state(_ann)
            if text is None and not saw_prompt:
                break

    def _update_seen_stopped(self, seen_stopped: bool, text: Optional[str], ann: Optional[str]) -> bool:
        if text and "stopped" in text:
            seen_stopped = True
        if ann == "stopped":
            seen_stopped = True
        return seen_stopped

    def _track_attach_state(self, ann: Optional[str]) -> None:
        # Settle stop = "signal" then "stopped"; attach's own stop has no signal.
        if ann == "signal":
            self._attach_signal_seen = True
        if ann == "stopped" and self._attach_signal_seen:
            self._attach_stop_seen = True

    def _send_gdb_command(self, command: str, wait_until_attached: bool = False, wait_for_prompt: bool = True) -> list:
        if not self.process:
            return []
        verbose_out(f"[SEND CLI] {command}", False)

        # Flush any stale output before sending the new command.
        self._flush_pending_output()

        if not hasattr(self, "_gdb_stdin") or self._gdb_stdin is None:
            print("broken pipe")
            return []
        try:
            self._gdb_stdin.write(f"{command}\n".encode("utf-8"))
            self._gdb_stdin.flush()
        except OSError:
            pass

        if not wait_for_prompt:
            return []

        responses: list[str] = []
        # If we're not in "wait until attached" mode, treat GDB as ready immediately.
        seen_stopped = not wait_until_attached
        while True:
            text, saw_prompt, ann = self._readline_gdb()
            if text is None and not saw_prompt:
                break
            if text:
                self._emit_accelerator_gdb_alert_if_needed(text)
                verbose_out(f"[RECV CLI] {text.rstrip()}", True)
                responses.append(text)
            if ann:
                verbose_out(f"[ANNOTATION] {ann}", True)
            if wait_until_attached:
                seen_stopped = self._update_seen_stopped(seen_stopped, text, ann)
            if saw_prompt and seen_stopped:
                break
        self._gdb_stdin.flush()
        return responses

    def _send_mi_command(self, command, **kwargs) -> list:  # type: ignore[override]
        """Send MI3 command through annotate3 interface using interpreter-exec."""
        if not self.process or self.process.poll() is not None:
            return []

        wait_for_prompt = kwargs.get("wait_for_prompt", True)
        wait_for_result_only = kwargs.get("wait_for_result_only", False)

        # In annotate mode, we wrap MI3 commands in interpreter-exec
        if self.command_execution_mode == "annotate":
            return self._send_mi_via_annotate(
                command, wait_for_prompt=wait_for_prompt, wait_for_result_only=wait_for_result_only
            )
        else:
            # Should not happen for accelerator controllers, but fallback to
            # parent (in case if we are doing HIP, for example)
            return super()._send_mi_command(command, **kwargs)

    def _send_mi_via_annotate(self, command: str, wait_for_prompt: bool = True, wait_for_result_only: bool = False) -> list:
        """Send MI3 command wrapped in interpreter-exec through annotate3 interface."""

        verbose_out(f"[SEND MI3 via ANNOTATE] {command} (wait_for_prompt={wait_for_prompt})", False)
        self._flush_pending_output()
        token = self.token_counter
        self.token_counter += 1

        # Wrap MI3 command in interpreter-exec (escape " and \ so mi_cmd stays one CLI argument)
        mi_cmd = f"{token}-{command}"
        wrapped_cmd = f'interpreter-exec mi3 "{_escape_for_gdb_cli_double_quotes(mi_cmd)}"'
        verbose_out(f"[WRAPPED CMD] {wrapped_cmd}", False)

        if not hasattr(self, "_gdb_stdin") or self._gdb_stdin is None:
            print("broken pipe")
            return []

        try:
            self._gdb_stdin.write(f"{wrapped_cmd}\n".encode("utf-8"))
            self._gdb_stdin.flush()
        except OSError:
            return []

        # If not waiting for prompt (async commands like continue), return immediately
        if not wait_for_prompt and not wait_for_result_only:
            verbose_out(f"[SEND MI3 via ANNOTATE] Not waiting for prompt (async command)", False)
            return []

        mi_responses = []
        got_result = False
        seen_prompt = False

        while True:
            text, saw_annotation, ann = self._readline_gdb()
            if text is None and not saw_annotation:
                break
            if saw_annotation:
                seen_prompt = True
            self._track_attach_state(ann)

            if text and text.strip():
                self._emit_accelerator_gdb_alert_if_needed(text)
                verbose_out(f"[RECV ANNOTATE] {text.rstrip()}", True)
                try:
                    parsed = gdbmiparser.parse_response(text)
                    if parsed:
                        verbose_out(f"[PARSED MI3] {parsed}", True)
                        mi_responses.append(parsed)
                        # A prompt annotation may arrive wrapped inside an MI
                        # console record instead of a raw annotation line.
                        payload = parsed.get("payload")
                        if isinstance(payload, str) and "\x1a\x1aprompt" in payload:
                            seen_prompt = True
                        pt = parsed.get("token")
                        if parsed.get("type") == "result" and pt is not None:
                            try:
                                if int(pt) == token:
                                    got_result = True
                            except (TypeError, ValueError):
                                pass
                        if parsed.get("type") == "notify" and parsed.get("message") in ("exited", "exited-normally"):
                            break
                except Exception as e:
                    verbose_out(f"[PARSE FAILED] Could not parse as MI3: {e}", True)

            if got_result and (seen_prompt or wait_for_result_only):
                break

        verbose_out(f"[SEND MI3 via ANNOTATE] Returning {len(mi_responses)} MI responses", False)
        return mi_responses

    def _get_current_location_via_cli(self) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Get current execution location via GDB CLI (for --annotate=3).
        Returns (script_path, python_line, frame_str)
        """
        script_path = None
        python_line = None
        frame_str = None

        # Get stack frames via MI command
        responses = self._send_mi_command("stack-list-frames")
        stack = []
        for resp in responses:
            if resp.get("type") == "result" and resp.get("message") == "done":
                stack = resp.get("payload", {}).get("stack", [])
                break

        if not stack:
            return (None, None, None)

        # Use only the #0 frame from the CLI backtrace
        frame = stack[0]
        file = frame.get("file", "")
        fullname = frame.get("fullname", file)
        line_str = frame.get("line", "")

        if fullname and line_str:
            frame_str = f"{fullname}:{line_str}"

        bp_manager = getattr(self, "breakpoint_manager", None)
        if bp_manager and line_str and line_str.isdigit():
            cpp_line = int(line_str)
            # Only attempt functor mapping when we are in a functor.hpp frame
            if "functor.hpp" in (file or "") or "functor.hpp" in (fullname or ""):
                # Use resolved path where possible to match breakpoint manager expectations
                path_for_mapping = fullname or file
                try:
                    resolved = str(Path(path_for_mapping).resolve())
                except Exception:
                    resolved = path_for_mapping
                location = bp_manager.get_python_location_for_functor(cpp_line, resolved)
                if location is not None:
                    script_path, python_line = location
                    verbose_out(f"[LOCATION MAPPING] Mapped to {script_path}:{python_line}")
        return (script_path, python_line, frame_str)

    def _setup_gdb(self, debug_pid: int):
        if debug_pid <= 0:
            print("error: debugger PID is incorrect. The debugger process probably can not start")
            sys.exit(1)
        self._target_pid = debug_pid

        # Use MI3 commands through annotate interface (via interpreter-exec)
        self._send_mi_command("gdb-set pagination off")
        self._send_mi_command("gdb-set print elements 0")
        self._send_mi_command("gdb-set confirm off")
        self._send_mi_command("gdb-set print thread-events off")
        self._send_mi_command("gdb-set filename-display absolute")
        self._send_mi_command("gdb-set breakpoint pending on")
        self._send_mi_command(f"gdb-set inferior-tty {self.pty_handler.get_slave_name()}")

        # Attach to the process
        self._attach_signal_seen = False
        self._attach_stop_seen = False
        self._send_mi_command(f"target-attach {debug_pid}")
        self._install_attached_breakpoints()

    def _get_attach_gdb_cmd(self):
        self.gdb_executable = native_debuggers.executable_for(self.accelerator_type)
        if self.command_execution_mode == "mi3":
            base = [self.gdb_executable, "--interpreter=mi3", "-ex", "set pagination off"]
        else:
            # cuda-gdb does not support mi3 mode
            base = [self.gdb_executable, "--annotate=3", "-ex", "set pagination off"]
            # When stdout is a PIPE, cuda-gdb uses full buffering; use stdbuf -oL so each line is flushed
            if shutil.which("stdbuf"):
                return ["stdbuf", "-oL", "-eL"] + base
        return base

    def _sync_breakpoints_to_gdb(self):
        """Synchronize breakpoints to pk_managed_breakpoints via GDB"""
        if self.breakpoint_manager.breakpoints:
            self.breakpoint_manager._update_cpp_breakpoints()

    def _wait_for_prompt_after_interrupt(self) -> None:
        """Consume interrupt-stop output until gdb shows a prompt again."""
        while True:
            text, saw_prompt, _ann = self._readline_gdb()
            if saw_prompt:
                return
            if text is None:
                return
            verbose_out(f"[INTERRUPT] {text.rstrip()}", True)

    def _apply_breakpoint_update_spec(self, spec: Dict[str, Any]) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        os.kill(self.process.pid, signal.SIGINT)
        self._wait_for_prompt_after_interrupt()
        self._install_breakpoint_spec(spec)
        self._send_mi_command("exec-continue", wait_for_prompt=False)

    def _continue_after_attach(self) -> None:
        self._pending_resume_after_attach = True
        self._pending_resume_fault_seen = False
        if self._attach_stop_seen:
            self._send_mi_command("exec-continue", wait_for_prompt=False)

    def _handle_post_attach_resume(self, text: Optional[str], ann: Optional[str]) -> None:
        """Drive the post-attach resume state machine; see _continue_after_attach."""
        if not self._pending_resume_after_attach:
            return
        if text and _EXEC_CONTINUE_ACCEPTED_RE.search(text):
            # gdb accepted the resume: attach settled, target running.
            self._disarm_post_attach_resume()
            return
        if text and _FAULT_SIGNAL_RE.search(text):
            self._pending_resume_fault_seen = True
        if ann == "stopped" and self._attach_stop_seen:
            if self._pending_resume_fault_seen:
                self._disarm_post_attach_resume()  # real fault: leave it stopped
            else:
                self._send_mi_command("exec-continue", wait_for_prompt=False)

    def _disarm_post_attach_resume(self) -> None:
        self._pending_resume_after_attach = False
        self._pending_resume_fault_seen = False

    def _interrupt_and_detach(self) -> None:
        """
        Interrupt the running target and detach. Waits for detach's ^done only;
        cuda-gdb emits no reliable prompt afterwards, so waiting one hangs.
        """
        os.kill(self.process.pid, signal.SIGINT)
        self._gdb_stdin.write(b"echo\n")  # force a prompt if the target was idle
        self._gdb_stdin.flush()
        self._wait_for_prompt_after_interrupt()
        self._send_mi_command("target-detach", wait_for_prompt=False, wait_for_result_only=True)
        self._close_gdb_master_streams()

    def _apply_detach(self) -> None:
        """Release ptrace on the target; cuda-gdb/rocgdb stays alive for a future reattach."""
        if self.process is None or self.process.poll() is not None:
            return
        self._pending_resume_after_attach = False
        self._interrupt_and_detach()
        self._is_gdb_attached = False

    def _apply_reattach(self, spec: Dict[str, Any]) -> None:
        """Re-attach to the target pid this helper was launched with, and reinstall its breakpoint spec."""
        self._attach_signal_seen = False
        self._attach_stop_seen = False
        self._send_mi_command(f"target-attach {self._target_pid}")
        self._is_gdb_attached = True
        self._install_breakpoint_spec(spec)
        self._continue_after_attach()

    # main debugger loop
    def _transparent_loop(self):
        if not hasattr(self, "_gdb_master_fd") or self._gdb_master_fd is None:
            return
        stdin_fd = sys.stdin.fileno()
        stdin_is_tty = sys.stdin.isatty()
        while self.running and self.process and self.process.poll() is None:
            try:
                self._drain_pending_breakpoint_updates()
                fds = [self._gdb_master_fd]
                if stdin_is_tty:
                    fds.append(stdin_fd)
                ready, _, _ = select.select(fds, [], [])
                if self._gdb_master_fd in ready:
                    text, _, _ann = self._readline_gdb()
                    self._emit_accelerator_gdb_alert_if_needed(text)
                    verbose_out(f"{text}; {_ann}", True)
                    if text is None:
                        break

                    # Inferior exited: stop the loop so we detach/quit and return control to pkdb.
                    # "Program terminated with signal" covers a killed inferior.
                    if text and ("exited normally" in text or "Program exited" in text or "Program terminated" in text):
                        self._inferior_exited = True
                        self.running = False
                        break
                    if text and "hit Breakpoint" in text and "pending" not in text:
                        self._disarm_post_attach_resume()
                        self._enter_user_code_mode()

                    self._track_attach_state(_ann)
                    self._handle_post_attach_resume(text, _ann)
            except KeyboardInterrupt:
                print("\nInterrupted", flush=True)
                break
            except Exception:
                continue
        self.stop()

    def _close_gdb_master_streams(self) -> None:
        if hasattr(self, "_gdb_stdin") and self._gdb_stdin is not None:
            try:
                self._gdb_stdin.close()
            except OSError:
                pass
            self._gdb_stdin = None
        if hasattr(self, "_gdb_master_stream") and self._gdb_master_stream is not None:
            try:
                self._gdb_master_stream.close()
            except OSError:
                pass
            self._gdb_master_stream = None
        self._gdb_master_fd = None

    def stop(self):
        """Override: detach the target, then kill gdb and release streams."""
        self.running = False
        try:
            if self.process:
                if self.process.poll() is None:
                    still_attached = self._is_gdb_attached and not self._inferior_exited
                    if still_attached and getattr(self, "_gdb_stdin", None) is not None:
                        self._interrupt_and_detach()
                    self.process.kill()
                self.process.wait()
        finally:
            self.process = None
            self._close_gdb_master_streams()
            self.pty_handler.stop()

    def start_attached(self, pid: int, ready_fd: Optional[int] = None, go_pipe_fd: Optional[int] = None):
        gdb_cmd = self._get_attach_gdb_cmd()
        self._is_gdb_attached = True

        # Use PIPE (like the working cuda-gdb wrapper): PTY can cause cuda-gdb to block after
        # "continue" and never deliver further output when the inferior runs; PIPE + bufsize=0 works.
        self.process = subprocess.Popen(
            gdb_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            close_fds=True,
            cwd=self.cwd,
        )
        self._gdb_stdin = self.process.stdin
        stdout_stream = self.process.stdout
        if stdout_stream is None:
            raise RuntimeError("cuda-gdb stdout is None")
        self._gdb_master_stream = stdout_stream
        self._gdb_master_fd = stdout_stream.fileno()
        self.running = True
        self._install_breakpoint_update_signal_handler()

        # Wait for initial GDB prompt before sending any commands (like reference wrapper)
        while self.process.poll() is None:
            _, saw_prompt, _ann = self._readline_gdb()
            if saw_prompt:
                break

        self.pty_handler.start()
        self._setup_gdb(pid)
        self._signal_ready(ready_fd)
        # Send "continue" without waiting for prompt so we don't block; _transparent_loop will
        # read output and handle the next prompt (breakpoint hit or inferior exit).
        self._continue_after_attach()
        # Tell the parent (inferior) it can run the parallel_for; it was blocking on the go pipe.
        self._transparent_loop()

    def _enter_user_code_mode(self):
        """Enter interactive command mode at user code breakpoint. Uses CLI only (no MI)."""
        accelerator_name = "CUDA" if self.accelerator_type == "cuda" else "HIP"
        script_path, python_line, frame_str = self._get_current_location_via_cli()
        verbose_out(f"[ENTER_USER_CODE] script_path={script_path}, python_line={python_line}, frame_str={frame_str}")
        if script_path and python_line is not None:
            script_name = os.path.basename(script_path)
            print(f"{accelerator_name} kernel breakpoint hit at {script_name}:{python_line}")
        elif frame_str:
            print(f"{accelerator_name} kernel breakpoint hit at {frame_str}")
        else:
            print(f"{accelerator_name} kernel breakpoint")
        self._show_accelerator_context()
        self._command_mode()

    def _show_accelerator_context(self):
        """Display current accelerator thread context via GDB CLI (no MI)."""
        self._send_gdb_command(f"info {self.accelerator_prefix} kernels")
