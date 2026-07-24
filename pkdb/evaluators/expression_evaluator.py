"""
Expression evaluation orchestrator
Get expression and parse and evaluates it.
"""

import ast
from typing import Any, Optional
from dataclasses import dataclass

from .context import EvaluationContext
from .variable_resolver import resolve_and_evaluate_variable
from .device_functions import DeviceFunctionRegistry
from .ipc_utils import extract_ipc_from_arg, _resolve_from_context
import time

import builtins as _builtins


class KokkosViewProxy:
    """
    Proxy object for Kokkos Views/arrays that allows Python-style operations.

    When accessed like a[0], it fetches the value from GDB.
    When sliced like a[0:10], it returns a list from GDB.
    Supports iteration for functions like sum(), max(), etc.
    """

    def __init__(self, var_name: str, context):
        self.var_name = var_name
        self.context = context

    def __getitem__(self, key):
        """Handle subscript access: a[0], a[1:5], etc."""
        if isinstance(key, slice):
            # slices: a[0:10]
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else None
            if stop is None:
                raise RuntimeError(f"Slice end required for {self.var_name}[{key}]")
            expr = f"{self.var_name}[{start}:{stop}]"
            result = resolve_and_evaluate_variable(self.context, expr)
            return result
        elif isinstance(key, int):
            # single index: a[0]
            expr = f"{self.var_name}[{key}]"
            return resolve_and_evaluate_variable(self.context, expr)
        else:
            raise RuntimeError(f"Unsupported subscript type: {type(key)}")

    def __len__(self):
        """Get length of the view."""
        from ..translators.variables import get_view_dimensions

        dimensions = get_view_dimensions(self.context.controller, self.var_name)
        if dimensions:
            self._length_cache = dimensions[0]
        else:
            self._length_cache = 0
        return self._length_cache

    def __iter__(self):
        """Support iteration for functions like sum(), max(), etc."""
        # fetch all elements at once as a slice command
        length = len(self)
        if length > 0:
            elements = self[0:length]
            yield from elements

    def __repr__(self):
        return f"<KokkosView: {self.var_name}>"


@dataclass
class EvaluationResult:
    value: Any
    type_info: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def __str__(self):
        if not self.success:
            return f"Error: {self.error}"
        return str(self.value)


class ExpressionEvaluator:
    _current_context: EvaluationContext | None = None

    def __init__(self, context: EvaluationContext):
        self.context = context
        self.controller = context.controller
        self.device_functions = DeviceFunctionRegistry()
        # defined set of supported python instructions (TODO: find a way to use
        # any python instruction without modifying this list)
        self.safe_builtins = {
            "abs": abs,
            "max": max,
            "min": min,
            "sum": self._sum_for_eval,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "any": any,
            "all": all,
            "round": round,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "set": set,
        }

    def _sum_for_eval(self, iterable, start=0):
        """
        sum() used in eval: if iterable is a device-array proxy, resolve it
        (retrieve variable from device to host via context), then use Python
        built-in sum on the host array.
        """
        ctx = ExpressionEvaluator._current_context or self.context
        if getattr(iterable, "var_name", None) and ctx:
            resolved = _resolve_from_context(iterable, ctx)
            if resolved is not None:
                if hasattr(resolved, "get"):
                    resolved = resolved.get()
                iterable = resolved
        return _builtins.sum(iterable, start)

    def evaluate(self, expression: str) -> EvaluationResult:
        try:
            ExpressionEvaluator._current_context = self.context
            try:
                expr_ast = self._parse_expression(expression)
                transformed_expr, namespace = self._prepare_expression(expr_ast, expression)
                eval_namespace = {**self.safe_builtins, **namespace}

                start_time = time.perf_counter()
                value = eval(transformed_expr, {"__builtins__": {}}, eval_namespace)
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"elapsed: {elapsed_time:.4f} seconds")

                type_info = self._get_type_info(value)
                return EvaluationResult(value=value, type_info=type_info, success=True)
            finally:
                ExpressionEvaluator._current_context = None
        except Exception as e:
            return EvaluationResult(value=None, success=False, error=str(e))

    def _parse_expression(self, expr: str) -> ast.AST:
        try:
            tree = ast.parse(expr, mode="eval")
            return tree.body
        except SyntaxError as e:
            raise RuntimeError(f"Syntax error in expression: {e}")

    def _prepare_expression(self, node: ast.AST, original_expr: str) -> tuple[str, dict]:
        """
        Prepare expression for Python evaluation by resolving all base variables.

        Example:
          Input: "a[0] + b / c"
          Resolve: a = KokkosView('a')
                   b = 10
                   c = 2
          Namespace: {'a': KokkosViewProxy('a'), 'b': 10, 'c': 2}
          Eval: "a[0] + b / c" with namespace = 5.0
          note: KokkosViewProxy allows to use standard array ops on its
        """
        namespace = {}

        # TODO: here is a chance for micro-optimization.
        # Basically if we have some expression like
        # `deviceFunction(Y) + deviceFunction(X)`
        # we can do everything on-device without copying the temporary results
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                self._handle_variable_node(child, isinstance(node, ast.Name), namespace)
            elif isinstance(child, ast.Call):
                self._handle_function_call_node(child, namespace)

        return original_expr, namespace

    def _handle_variable_node(self, child: ast.Name, is_bare_variable: bool, namespace: dict) -> None:
        """Handle a variable node in the AST"""
        var_name = child.id
        if var_name in self.safe_builtins or var_name in namespace:
            return

        # Special case: if expression is just a bare variable name that's an array,
        # fetch the actual array contents instead of returning a proxy
        if is_bare_variable and self._is_kokkos_view_or_array(var_name):
            value = self._resolve_bare_array(var_name)
        else:
            value = self._resolve_variable(var_name)
        namespace[var_name] = value

    def _resolve_bare_array(self, var_name: str) -> Any:
        """Resolve a bare array variable to its full contents"""
        proxy = KokkosViewProxy(var_name, self.context)
        try:
            length = len(proxy)
            return proxy[0:length] if length > 0 else []
        except Exception:
            return proxy

    def _handle_function_call_node(self, child: ast.Call, namespace: dict) -> None:
        """Handle a function call node in the AST"""
        if not isinstance(child.func, ast.Name):
            return

        func_name = child.func.id
        if func_name in self.safe_builtins or func_name in namespace:
            return

        # Priority 1: Try to resolve from program context first
        try:
            func_obj = self._resolve_callable(func_name)
            namespace[func_name] = func_obj
            return
        except RuntimeError:
            pass

        # Priority 2: Check if it's a special on-device function
        if self.device_functions.is_special_function(func_name):
            handler = self.device_functions.get_handler(func_name)
            func_obj = self._create_device_function_wrapper(handler)
            namespace[func_name] = func_obj

    def _resolve_variable(self, var_name: str) -> Any:
        """
        Resolve a variable from the context (GDB/Kokkos or Python).

        For Kokkos Views/arrays, return a proxy object that allows Python subscripting.
        For scalar values, return the actual resolved value.
        """
        from .context import ContextType

        if self.context.type == ContextType.PDB:
            return resolve_and_evaluate_variable(self.context, var_name)

        if self._is_kokkos_view_or_array(var_name):
            return KokkosViewProxy(var_name, self.context)
        else:
            return resolve_and_evaluate_variable(self.context, var_name)

    def _is_kokkos_view_or_array(self, var_name: str) -> bool:
        """Check if a variable is a Kokkos View or array."""
        from ..translators.variables import get_view_dimensions

        try:
            dimensions = get_view_dimensions(self.context.controller, var_name)
            return dimensions is not None and len(dimensions) > 0
        except:
            return False

    def _create_device_function_wrapper(self, handler):
        """
        Create a wrapper that intercepts device function calls, extracts IPC
        tokens from array arguments, and passes them to the handler.
        Resolves KokkosViewProxy to actual Python variable (CuPy array) via context.frame.
        """

        def wrapper(*args, **kwargs):
            ctx = ExpressionEvaluator._current_context
            if ctx is None:
                raise RuntimeError("No evaluation context for device function")
            ipc_tokens = []
            for arg in args:
                token = extract_ipc_from_arg(arg, ctx)
                if token is not None:
                    ipc_tokens.append(token)
                # Scalars could be passed in kwargs if needed
            return handler(ipc_tokens, ctx, **kwargs)

        return wrapper

    def _resolve_callable(self, func_name: str) -> Any:
        """Resolve a function/callable from the context."""
        from .context import ContextType

        # python context
        if self.context.type == ContextType.PDB:
            if self.context.frame:
                if func_name in self.context.frame.f_locals:
                    return self.context.frame.f_locals[func_name]
                if func_name in self.context.frame.f_globals:
                    return self.context.frame.f_globals[func_name]

        if func_name in self.context.script_globals:
            return self.context.script_globals[func_name]
        raise RuntimeError(f"Function '{func_name}' not found")

    def _get_type_info(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return None
        return type(value).__name__
