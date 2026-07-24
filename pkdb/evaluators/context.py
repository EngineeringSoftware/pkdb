"""
Evaluation context for expression evaluation.

For example, when we need to switch contexts (from cuda to python) this file
helps us to find variables across different contexts
"""

from enum import Enum
from typing import Optional, Any, Dict


class ContextType(Enum):
    """Types of debugging contexts"""

    GDB = "gdb"
    CUDA = "cuda"
    HIP = "hip"
    PDB = "pdb"


class EvaluationContext:
    def __init__(
        self,
        context_type: ContextType,
        controller: Any,
        frame: Optional[Any] = None,
        script_globals: Optional[Dict] = None,
    ):
        self.type = context_type
        self.controller = controller
        self.frame = frame
        self.script_globals = script_globals or {}

    def resolve_variable(self, name: str) -> Any:
        from .variable_resolver import resolve_variable

        return resolve_variable(self, name)

    def execute_gdb_command(self, cmd: str) -> list:
        if self.type == ContextType.PDB:
            raise RuntimeError("Cannot execute GDB commands in PDB context")
        return self.controller._send_mi_command(cmd)

    def get_variable_info(self, var: str) -> Optional[dict]:
        if self.type == ContextType.PDB:
            if self.frame:
                all_vars = {**self.frame.f_globals, **self.frame.f_locals}
                if var in all_vars:
                    val = all_vars[var]
                    return {"type": type(val).__name__, "value": str(val), "python_type": type(val)}
            return None
        responses = self.controller._send_mi_command(f'data-evaluate-expression "{var}"')
        for resp in responses:
            if resp.get("type") == "result" and resp.get("message") == "done":
                return {"value": resp.get("payload", {}).get("value", "?")}
        return None
