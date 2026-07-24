"""
Device function handlers for special pre-defined functions.

These handlers receive IPC tokens (not raw Python objects) and execute
operations directly on device memory. They are invoked via a wrapper that
extracts IPC info from array arguments before calling.
"""

from typing import Any, Callable, List, Optional

from .ipc_utils import IpcToken

# Registry: public_name -> method_name (populated by @device_function decorator)
_DEVICE_FUNCTION_REGISTRY: dict[str, str] = {}


def device_function(name: str) -> Callable:
    """Decorator that registers a method as a device function handler."""

    def decorator(method: Callable) -> Callable:
        _DEVICE_FUNCTION_REGISTRY[name] = method.__name__
        return method

    return decorator


class DeviceFunctionRegistry:
    """Registry for special device functions that can be executed on accelerators."""

    def is_special_function(self, func_name: str) -> bool:
        """Check if function name is in our special pre-defined functions."""
        return func_name in _DEVICE_FUNCTION_REGISTRY

    def get_handler(self, func_name: str) -> Optional[Callable]:
        """Get handler for special pre-defined functions."""
        method_name = _DEVICE_FUNCTION_REGISTRY.get(func_name)
        if method_name is None:
            return None
        return getattr(self, method_name, None)

    @device_function("device_sum")
    def _device_sum(self, ipc_tokens: List[IpcToken], context, **kwargs) -> Any:
        """
        Execute sum operation on device.
        Receives IPC tokens, opens handles, runs kernel, returns sum (e.g. int/float).
        """
        if not ipc_tokens:
            raise ValueError("device_sum requires at least one array argument")
        token = ipc_tokens[0]
        from ..scripts.cuda_device_sum import sum_array_from_handle

        return sum_array_from_handle(
            token["handle_hex"],
            token["num_elements"],
            token.get("dtype_name", "int64"),
        )

    # TODO: implement the rest of the functions same way
    def _device_max(self, ipc_tokens: List[IpcToken], context, **kwargs) -> Any:
        """Execute max operation on device."""
        if not ipc_tokens:
            raise ValueError("device_max requires at least one array argument")
        pass

    def _device_min(self, ipc_tokens: List[IpcToken], context, **kwargs) -> Any:
        """Execute min operation on device."""
        if not ipc_tokens:
            raise ValueError("device_min requires at least one array argument")
        pass

    def _device_reduce(self, ipc_tokens: List[IpcToken], context, **kwargs) -> Any:
        """Execute reduce operation on device (custom op)."""
        if not ipc_tokens:
            raise ValueError("device_reduce requires at least one array argument")
        pass
