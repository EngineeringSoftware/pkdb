import os
import json
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

from pygdbmi import gdbmiparser


from .pty_handler import PTYHandler
from .tty_command_line import read_tty_command_line
from ..core import BreakpointManager, get_debug_properties, native_debuggers
from ..core.debug_properties import verbose_out
from ..commands import register_gdb_commands
from ..commands.gdb_helpers import (
    get_current_python_location,
    get_frame_info,
    get_current_thread_id,
    get_openmp_threads,
)


def _allow_ptrace_attach():
    """Allow any process to ptrace the calling process (Yama LSM restriction bypass)."""
    try:
        import prctl

        prctl.set_ptracer(prctl.PR_SET_PTRACER_ANY)
        return
    except Exception:
        pass
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        libc.prctl(0x59616D61, ctypes.c_ulong(-1).value, 0, 0, 0)  # PR_SET_PTRACER_ANY
    except Exception:
        pass


def _wait_for_ready(ready_read_fd: int) -> None:
    """Block until the helper signals ready, or until it dies.

    The helper writes one byte once its gdb is set up; if it exits first, every write end of
    the pipe closes and the read returns EOF. Both outcomes are a real event, so this never
    needs a timeout or a poll interval.
    """
    try:
        os.read(ready_read_fd, 1)
    finally:
        os.close(ready_read_fd)


# Attaches native debugger (gdb / cuda-gdb / rocgdb) to our debuggee process and
# return attached process PID in order to control this process (using TTY + MI3 interface)
def launch_attached_debugger(
    pid,
    script_path,
    accelerator_info=None,
    handoff_payload: Optional[Dict[str, Any]] = None,
):
    _allow_ptrace_attach()

    debugger_name = "GDB controller"
    if accelerator_info is not None:
        debugger_name = f"{accelerator_info['name']}-GDB controller"
    verbose_out(f"\n[pkdb] Handing off to {debugger_name}.")

    cmd = [
        sys.executable,
        "-m",
        "pkdb.runtime.gdb_attach_helper",
        "--pid",
        str(pid),
        "--script-path",
        script_path,
    ]

    # forward all original script arguments (sys.argv[1:]) to the helper
    for script_arg in sys.argv[1:]:
        encoded = script_arg
        if script_arg.startswith("-"):
            encoded = f"@@{script_arg}"
        cmd.extend(["--script-arg", encoded])
    if handoff_payload is not None:
        from ..runtime.handoff_shm import handoff_payload_write_shm

        shm_name, shm_nbytes = handoff_payload_write_shm(handoff_payload)
        cmd.extend(
            [
                "--handoff-shm-name",
                shm_name,
                "--handoff-shm-nbytes",
                str(shm_nbytes),
            ]
        )
    if accelerator_info is not None:
        cmd.extend(["--accelerator-type", accelerator_info["type"]])
        cmd.extend(["--accelerator-name", accelerator_info["name"]])

    # Forward current debug properties so attach-helper process starts with
    # the same settings configured in the parent command loop.
    debug_props = get_debug_properties()
    cmd.extend(["--debug-properties-json", json.dumps(debug_props.list_properties())])
    if debug_props.verbose:
        cmd.append("--verbose")

    cwd = os.path.dirname(script_path) or None
    # Ack pipe for live breakpoint updates: helper writes one byte per
    # applied update; EOF (helper death) unblocks the waiting pusher.
    bp_ack_read_fd, bp_ack_write_fd = os.pipe()
    cmd.extend(["--bp-ack-fd", str(bp_ack_write_fd)])
    # Ready pipe: the helper writes one byte when its gdb is up; EOF covers it dying first.
    ready_read_fd, ready_write_fd = os.pipe()
    cmd.extend(["--ready-fd", str(ready_write_fd)])
    proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True, pass_fds=(bp_ack_write_fd, ready_write_fd))
    os.close(bp_ack_write_fd)
    os.close(ready_write_fd)
    setattr(proc, "pkdb_bp_ack_read_fd", bp_ack_read_fd)
    _wait_for_ready(ready_read_fd)
    return proc


_BP_UPDATE_MAX_BYTES = 8 * 1024 * 1024


class _BreakpointUpdateSignal(Exception):
    """Raised from the SIGUSR1 handler to unblock a pending read/select immediately."""


def _push_control_message(helper_proc: subprocess.Popen, seq: int, message: Dict[str, Any]) -> None:
    """
    Send a control message to an attached helper, then block until it acks.
    The caller is the debuggee itself, so blocking here is what guarantees
    the target cannot proceed (e.g. launch a kernel) before the message has
    actually been applied.
    """
    from ..runtime.handoff_shm import handoff_payload_write_shm

    name = f"pkdb-bpupdate-{helper_proc.pid}-{seq}"
    handoff_payload_write_shm(message, name)
    try:
        os.kill(helper_proc.pid, signal.SIGUSR1)
    except (ProcessLookupError, PermissionError, OSError):
        return
    ack_fd = getattr(helper_proc, "pkdb_bp_ack_read_fd", None)
    if ack_fd is None:
        return
    # One byte per applied message; b"" (EOF) means the helper died.
    os.read(ack_fd, 1)


def push_breakpoint_update(helper_proc: subprocess.Popen, seq: int, spec: Dict[str, Any]) -> None:
    """Install additional C++ breakpoints in an already-attached attach-helper session."""
    _push_control_message(helper_proc, seq, {"action": "install_breakpoints", "spec": spec})


def push_control_detach(helper_proc: subprocess.Popen, seq: int) -> None:
    """Release ptrace on the target; the helper (and its gdb) stays alive for a future reattach."""
    _push_control_message(helper_proc, seq, {"action": "detach"})


def push_control_reattach(helper_proc: subprocess.Popen, seq: int, spec: Dict[str, Any]) -> None:
    """Re-attach an already-launched helper to the pid it was originally given, then reinstall its breakpoint spec."""
    _push_control_message(helper_proc, seq, {"action": "reattach", "spec": spec})


class GDBController:
    def __init__(
        self,
        args,
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
        self.args = args
        self._handoff_breakpoint_spec: Optional[Dict[str, Any]] = handoff_breakpoint_spec
        self._bp_ack_write_fd: Optional[int] = bp_ack_write_fd
        self.breakpoint_manager = breakpoint_manager or BreakpointManager(
            script_path=script_path,
            script_args=script_args,
            gdb_command_sender=self._send_mi_command,
        )
        self.breakpoint_manager.set_gdb_command_sender(self._send_mi_command)
        self.current_breakpoint_line: Optional[int] = None
        self.current_thread_id: Optional[str] = None
        self.openmp_threads = []  # Store OpenMP thread IDs
        self.script_path = script_path
        self.script_args = list(script_args or [])

        # DEPRECATE ME: with latest architecture we can extract globals/locals/functions via prompting pdb
        self.script_globals = script_globals or {}
        self.script_variables = script_variables or {}

        self.process = None
        self._target_pid: int = 0
        self.token_counter = 1000
        self.running = False
        self.pty_handler = PTYHandler()
        self.debug_properties = get_debug_properties()
        self._is_gdb_attached = False
        self.cwd = cwd
        self.commands = {}
        self.command_prompt = "(pkdb) "
        # Interface mode: "mi3" (GDB MI) or "annotate3" (CLI with annotations). Set per debugger.
        self.command_execution_mode = "mi3"
        register_gdb_commands(self)
        self._cmd_history: list[str] = []
        self._installed_cpp_locations: Set[Tuple[str, int]] = set()
        self._bp_update_pending = False
        self._next_bp_update_seq = 1

        if breakpoints:
            for bp in breakpoints:
                self.breakpoint_manager.add_breakpoint(bp)
        if all_locs:
            self.breakpoint_manager.set_all_locs(all_locs)

    def _install_breakpoint_spec(self, spec: Dict[str, Any]) -> None:
        """Install cpp breakpoints from a mapping+enabled_python spec (initial attach or a live update)."""
        cpp_to_python = spec.get("cpp_to_python")
        if cpp_to_python:
            self.breakpoint_manager.apply_handoff_cpp_to_python(cpp_to_python)

        cpp_locations_by_python_loc: Dict[Tuple[str, int], List[List[Any]]] = {
            (entry["py_file"], entry["py_line"]): entry.get("cpp") or [] for entry in spec.get("mapping") or []
        }
        for py_file, py_line in spec.get("enabled_python") or []:
            for cpp_file, cpp_line in cpp_locations_by_python_loc.get((py_file, py_line), []):
                loc = (cpp_file, cpp_line)
                if loc in self._installed_cpp_locations:
                    continue
                self._installed_cpp_locations.add(loc)
                self._send_mi_command(f'break-insert -f "{cpp_file}:{cpp_line}"')

    def _install_attached_breakpoints(self) -> None:
        """Set device breakpoints after attach: prefer JIT Python to C++ mapping + enabled Python LOCs."""
        spec = self._handoff_breakpoint_spec
        if spec:
            self._install_breakpoint_spec(spec)
            return
        self.breakpoint_manager.sync_breakpoints_to_gdb()

    def _on_breakpoint_update_signal(self, signum, frame) -> None:
        self._bp_update_pending = True
        if not self.running:
            return
        raise _BreakpointUpdateSignal()

    def _install_breakpoint_update_signal_handler(self) -> None:
        """SIGUSR1 means "a new breakpoint spec is waiting"; see push_breakpoint_update()."""
        signal.signal(signal.SIGUSR1, self._on_breakpoint_update_signal)

    def _apply_breakpoint_update_spec(self, spec: Dict[str, Any]) -> None:
        """With mi-async on, break-insert is processed even while the target runs."""
        self._install_breakpoint_spec(spec)

    def _apply_detach(self) -> None:
        """Release ptrace on the target; this gdb process stays alive for a future reattach."""
        self._send_mi_command("exec-interrupt")
        self._send_mi_command("target-detach")
        self._is_gdb_attached = False

    def _apply_reattach(self, spec: Dict[str, Any]) -> None:
        """Re-attach to the target pid this helper was launched with, and reinstall its breakpoint spec."""
        self._send_mi_command(f"target-attach {self._target_pid}")
        self._is_gdb_attached = True
        self._install_breakpoint_spec(spec)
        self._send_mi_command("exec-continue")

    def _apply_control_message(self, message: Dict[str, Any]) -> None:
        action = message.get("action", "install_breakpoints")
        if action == "install_breakpoints":
            self._apply_breakpoint_update_spec(message["spec"])
        elif action == "detach":
            self._apply_detach()
        elif action == "reattach":
            self._apply_reattach(message["spec"])

    def _ack_breakpoint_update(self) -> None:
        """One ack byte per applied message; the pusher blocks on this (see push_breakpoint_update() et al.)."""
        if self._bp_ack_write_fd is not None:
            os.write(self._bp_ack_write_fd, b"\x01")

    def _drain_pending_breakpoint_updates(self) -> None:
        """Apply any control message(s) pushed in since the last check (see push_breakpoint_update() et al.)."""
        if not self._bp_update_pending:
            return
        self._bp_update_pending = False
        from ..runtime.handoff_shm import handoff_payload_read_shm

        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
        try:
            while True:
                name = f"pkdb-bpupdate-{os.getpid()}-{self._next_bp_update_seq}"
                message = handoff_payload_read_shm(name, _BP_UPDATE_MAX_BYTES)
                if message is None:
                    break
                self._apply_control_message(message)
                self._next_bp_update_seq += 1
                self._ack_breakpoint_update()
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR1})

    def _setup_gdb(self, debug_pid: int):
        if debug_pid <= 0:
            print("error: debugger PID is incorrect. The debugger process probably can not start")
            sys.exit(1)
        self._target_pid = debug_pid
        self._send_mi_command("gdb-set pagination off")
        self._send_mi_command("gdb-set confirm off")
        self._send_mi_command("gdb-set print thread-events off")
        self._send_mi_command("gdb-set breakpoint pending on")
        self._send_mi_command("gdb-set mi-async on")
        self._send_mi_command(f"gdb-set inferior-tty {self.pty_handler.get_slave_name()}")
        # attach launched debugger process
        self._send_mi_command(f"target-attach {debug_pid}")
        self._install_attached_breakpoints()

    def _get_attach_gdb_cmd(self):
        self.gdb_executable = native_debuggers.executable_for("gdb")
        return [self.gdb_executable, "--interpreter=mi3"]

    def _readline_gdb(self):
        """Read and parse one line directly from gdb stdout. Returns None on EOF."""
        if not self.process or self.process.stdout is None:
            return None
        line = self.process.stdout.readline()
        if not line:
            return None
        verbose_out({line.rstrip()}, True)
        try:
            return gdbmiparser.parse_response(line)
        except Exception:
            return None

    def _signal_ready(self, ready_fd: Optional[int]) -> None:
        """Unblock the parent waiting in `_wait_for_ready`; the close hands it EOF if we die later."""
        if ready_fd is not None:
            os.write(ready_fd, b"\x01")
            os.close(ready_fd)

    def start_attached(self, pid: int, ready_fd: Optional[int] = None):
        gdb_cmd = self._get_attach_gdb_cmd()
        self._is_gdb_attached = True

        self.process = subprocess.Popen(
            gdb_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=self.cwd,
        )
        self.running = True
        self._install_breakpoint_update_signal_handler()
        self._setup_gdb(debug_pid=pid)
        self._signal_ready(ready_fd)
        self._send_mi_command("exec-continue")
        self._transparent_loop()

    def _send_mi_command(self, command, **kwargs) -> list:
        if not self.process or self.process.poll() is not None:
            return []
        # kwargs may contain wait_for_prompt, but base GDBController always waits
        token = self.token_counter
        self.token_counter += 1

        actual_command = f"{token}-{command}"
        verbose_out(f"[SEND MI3] {command}", False)

        if self.process.stdin is not None:
            self.process.stdin.write(f"{actual_command}\n")
            self.process.stdin.flush()
        else:
            return []

        responses = []
        while True:
            parsed = self._readline_gdb()
            if parsed is None:
                break
            responses.append(parsed)
            verbose_out(f"[RECV MI3] {parsed}", True)
            if parsed.get("type") == "result" and parsed.get("token") == token:
                break
            if parsed.get("type") == "notify" and parsed.get("message") in ("exited", "exited-normally"):
                break

        return responses

    # main debugger loop: read GDB MI output line by line
    def _transparent_loop(self):
        while self.running and self.process and self.process.poll() is None:
            try:
                # Always check if there are some pending updates
                self._drain_pending_breakpoint_updates()
                parsed = self._readline_gdb()
                if parsed is None:
                    break

                msg_type = parsed.get("type")
                message = parsed.get("message")
                payload = parsed.get("payload", {})
                verbose_out(message, True, payload)

                if msg_type == "notify" and message == "stopped":
                    reason = payload.get("reason")
                    signal_name = payload.get("signal-name")
                    # Covers exited-normally, exited, and exited-signalled (killed inferior).
                    if reason and reason.startswith("exited"):
                        self.running = False
                        break

                    elif reason == "breakpoint-hit":
                        self._handle_workunit_breakpoint(payload)

                    elif reason == "signal-received" and signal_name and signal_name != "SIGTRAP":
                        frame_info = get_frame_info(self)
                        if frame_info:
                            print(f"\nProgram stopped with {signal_name} at {frame_info}")
                        else:
                            print(f"\nProgram stopped with {signal_name}")
                        self._command_mode()

            except KeyboardInterrupt:
                print("\nInterrupted")
                break
            except Exception:
                continue

        self.stop()

    def _handle_workunit_breakpoint(self, payload):
        # Command mode already set in _transparent_loop before this is called
        self.current_thread_id = get_current_thread_id(self)
        self.openmp_threads = get_openmp_threads(self)

        if self.debug_properties.verbose:
            verbose_msg = (
                "OpenMP threads detected: {' '.join(self.openmp_threads)}"
                if self.openmp_threads
                else "Warning: No OpenMP threads detected!"
            )
            verbose_out(verbose_msg)

        source_file, python_lineno = get_current_python_location(self)
        if python_lineno is not None:
            display = os.path.basename(source_file) if source_file else os.path.basename(self.script_path)
            print(f"\nWorkunit breakpoint, {display}:{python_lineno}")
            self.current_breakpoint_line = python_lineno
        else:
            frame_info = get_frame_info(self)
            if frame_info:
                print(f"\nWorkunit breakpoint, {frame_info}")

        self._command_mode()

    def command(self, *names):
        def decorator(func):
            for name in names:
                self.commands[name] = func
            return func

        return decorator

    def parse_and_execute(self, cmd_line) -> bool:
        """Execute one command line. Returns True when command mode should exit."""
        if not cmd_line:
            return False

        # First word is the command name, the rest is passed to the handler as args.
        cmd_name, _, args = cmd_line.strip().partition(" ")
        if not cmd_name:
            return False

        handler = self.commands.get(cmd_name)
        if handler is None:
            print(f"Unknown command: {cmd_name}")
            print("Type 'help' for available commands")
            return False

        try:
            handler(self, args.strip())
        except StopIteration:
            # Handlers raise StopIteration to leave command mode (quit/continue/step).
            return True
        except Exception as e:
            print(f"Error executing {cmd_name}: {e}")
        return False

    def _command_mode(self):
        self.pty_handler.set_gdb_command_mode(True)
        try:
            while True:
                try:
                    cmd = self._read_command_line(self.command_prompt).strip()
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    continue

                if not cmd:
                    continue

                if self.parse_and_execute(cmd):
                    break
        finally:
            self.pty_handler.set_gdb_command_mode(False)

    def _read_command_line(self, prompt: str) -> str:
        return read_tty_command_line(prompt, self._cmd_history)

    def stop(self):
        self.running = False
        self.pty_handler.stop()
        if self.process:
            if self.process.poll() is not None:
                try:
                    if self.process.stdin is not None:
                        self.process.stdin.close()
                except Exception:
                    pass
                self.process = None
                return
            try:
                if self.process.stdin is not None:
                    if self._is_gdb_attached:
                        self.process.stdin.write("detach\n")
                        self.process.stdin.flush()
                    self.process.stdin.write("quit\n")
                    self.process.stdin.flush()
                self.process.wait()
            except Exception:
                if self.process.poll() is None:
                    self.process.terminate()
            finally:
                if self.process and self.process.stdin is not None:
                    self.process.stdin.close()
            self.process = None
