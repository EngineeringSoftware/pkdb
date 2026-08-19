"""
External helper process for attach-mode GDBController handoff.
THIS IS NOT THE MAIN FILE FOR PKDB.
pkdb launches this as a subprocess from `gdb_controller.py:76`
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from ..commands.gdb_commands import extract_script_functions, extract_script_variables
from ..controllers.accelerator_gdb_controller import AcceleratorGDBController
from ..controllers.gdb_controller import GDBController
from .handoff_shm import handoff_payload_read_shm


def _parse_args():
    parser = argparse.ArgumentParser(description="Attach GDBController to a running pkdb PID")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--script-path", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--script-arg", action="append", default=[])
    parser.add_argument("--handoff-shm-name", default="")
    parser.add_argument("--handoff-shm-nbytes", type=int, default=0)
    parser.add_argument("--accelerator-type", choices=["cuda", "hip"])
    parser.add_argument("--accelerator-name")
    parser.add_argument("--bp-ack-fd", type=int, default=None)
    parser.add_argument("--go-pipe-fd", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug-properties-json", default="")
    return parser.parse_args()


def _apply_handoff_cpp_to_python(controller, handoff_spec: Optional[Dict[str, Any]]) -> None:
    """Apply parent-built C++ -> Python table; no functor file scan."""
    if not handoff_spec:
        return
    bm = getattr(controller, "breakpoint_manager", None)
    if bm is None:
        return
    entries = handoff_spec.get("cpp_to_python")
    if not isinstance(entries, list) or not entries:
        return
    bm.apply_handoff_cpp_to_python(entries)


def main() -> int:
    args = _parse_args()
    # decode script args that were prefixed to avoid treating them as options
    decoded_script_args = []
    for a in args.script_arg:
        if isinstance(a, str) and a.startswith("@@"):
            decoded_script_args.append(a[2:])
        else:
            decoded_script_args.append(a)
    args.script_arg = decoded_script_args
    script_path = os.path.abspath(args.script_path)
    cwd = os.path.dirname(script_path) or None

    script_globals = extract_script_functions(script_path)
    script_variables = extract_script_variables(script_path)

    handoff_spec: Optional[Dict[str, Any]] = None
    if args.handoff_shm_name and args.handoff_shm_nbytes > 0:
        handoff_spec = handoff_payload_read_shm(args.handoff_shm_name, args.handoff_shm_nbytes)

    if args.accelerator_type is None:
        controller = GDBController(
            args=[],
            breakpoints=[],
            script_path=script_path,
            script_args=args.script_arg,
            script_globals=script_globals,
            script_variables=script_variables,
            cwd=cwd,
            handoff_breakpoint_spec=handoff_spec,
            bp_ack_write_fd=args.bp_ack_fd,
        )
    else:
        controller = AcceleratorGDBController(
            accelerator_type=args.accelerator_type,
            args=[],
            breakpoints=[],
            script_path=script_path,
            script_args=args.script_arg,
            script_globals=script_globals,
            script_variables=script_variables,
            cwd=cwd,
            handoff_breakpoint_spec=handoff_spec,
            bp_ack_write_fd=args.bp_ack_fd,
        )

    if args.debug_properties_json:
        try:
            props = json.loads(args.debug_properties_json)
            if isinstance(props, dict):
                for name, value in props.items():
                    controller.debug_properties.set_property(name, value)
        except Exception:
            pass

    # Apply verbose flag from parent (pdb) side if provided.
    if args.verbose:
        controller.debug_properties.verbose = True
    _apply_handoff_cpp_to_python(controller, handoff_spec)
    controller.start_attached(args.pid, ready_file=args.ready_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
