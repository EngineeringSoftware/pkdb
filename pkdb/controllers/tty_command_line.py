"""
TTY command-line input with history and arrow-key editing for pkdb prompts.

Kept separate from GDBController so debugger controllers stay small; call
read_tty_command_line(prompt, history) from _read_command_line.
"""

import os
import sys
import termios
import tty


def read_tty_command_line(prompt: str, history: list[str]) -> str:
    """
    Read one line. On a TTY: raw mode, arrows, history. Otherwise: input().

    history is mutated: non-empty stripped lines are appended on Enter.
    """
    if not os.isatty(sys.stdin.fileno()):
        return input(prompt)

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    history_index = len(history)
    buf: list[str] = []
    cursor = 0

    def render():
        line = "".join(buf)
        sys.stdout.write("\r")
        sys.stdout.write(prompt + line)
        sys.stdout.write("\x1b[K")
        move_left = len(line) - cursor
        if move_left > 0:
            sys.stdout.write("\b" * move_left)
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        sys.stdout.write("\r\x1b[2K")
        sys.stdout.write(prompt)
        sys.stdout.flush()

        while True:
            ch = sys.stdin.read(1)
            if not ch:
                raise EOFError

            if ch in ("\r", "\n"):
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                line = "".join(buf)
                if line.strip():
                    history.append(line)
                return line

            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x04":
                raise EOFError

            if ch in ("\x7f", "\b"):
                if cursor > 0:
                    del buf[cursor - 1]
                    cursor -= 1
                    render()
                continue

            if ch == "\x1b":
                n1 = sys.stdin.read(1)
                n2 = sys.stdin.read(1)
                if n1 == "[":
                    if n2 == "A":
                        if history and history_index > 0:
                            history_index -= 1
                            buf = list(history[history_index])
                            cursor = len(buf)
                            render()
                    elif n2 == "B":
                        if history_index < len(history) - 1:
                            history_index += 1
                            buf = list(history[history_index])
                        else:
                            history_index = len(history)
                            buf = []
                        cursor = len(buf)
                        render()
                    elif n2 == "C":
                        if cursor < len(buf):
                            cursor += 1
                            render()
                    elif n2 == "D":
                        if cursor > 0:
                            cursor -= 1
                            render()
                continue

            if ch >= " ":
                buf.insert(cursor, ch)
                cursor += 1
                render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
