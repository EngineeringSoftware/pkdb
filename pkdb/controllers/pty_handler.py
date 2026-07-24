"""
PTY handler for inferior process I/O
Allows pdb to work normally when inferior runs under GDB
"""

import os
import sys
import pty
import select
import threading


# Manages PTY for inferior process stdin/stdout forwarding.
class PTYHandler:
    def __init__(self):
        # Create pty for inferior (so pdb can work)
        # slave_fd is automatically closed when GDB takes ownership
        self.master_fd, slave_fd = pty.openpty()
        self.slave_name = os.ttyname(slave_fd)

        self.running = False
        self.in_gdb_command_mode = False
        self.reader_thread = None
        self.writer_thread = None

    # Start PTY I/O threads
    def start(self):
        self.running = True
        self.reader_thread = threading.Thread(target=self._read_pty_output, daemon=True)
        self.reader_thread.start()
        self.writer_thread = threading.Thread(target=self._write_pty_input, daemon=True)
        self.writer_thread.start()

    # Stop PTY I/O threads and close pty
    def stop(self):
        self.running = False
        if hasattr(self, "master_fd"):
            try:
                os.close(self.master_fd)
            except Exception:
                pass

    # Set whether we're in GDB command mode (stops stdin forwarding)
    def set_gdb_command_mode(self, enabled):
        self.in_gdb_command_mode = enabled

    # returns: the slave pty name for debugger to use
    def get_slave_name(self):
        return self.slave_name

    # Read inferior's output from pty and display it
    def _read_pty_output(self):
        while self.running:
            try:
                if not hasattr(self, "master_fd"):
                    break
                r, _, _ = select.select([self.master_fd], [], [], 0.1)
                if r:
                    data = os.read(self.master_fd, 4096)
                    if data:
                        sys.stdout.write(data.decode("utf-8", errors="replace"))
                        sys.stdout.flush()
            except Exception:
                break

    # Forward stdin to inferior's pty when not in GDB command mode
    # (TODO: add more debuggers later)
    def _write_pty_input(self):
        while self.running:
            try:
                if self.in_gdb_command_mode:
                    # In GDB mode - stdin handled by GDB controller
                    continue

                if not hasattr(self, "master_fd"):
                    break

                # Check if stdin has data (only if it's a tty)
                if not os.isatty(sys.stdin.fileno()):
                    continue

                r, _, _ = select.select([sys.stdin], [], [], 0)
                if r:
                    data = os.read(sys.stdin.fileno(), 1024)
                    if data:
                        os.write(self.master_fd, data)
            except Exception:
                pass
