"""
Debug properties for PyKokkos debugger
Manages global debugger settings accessible by all debugger backends (GDB, PDB, CUDA-GDB, etc.)
"""

import time


class DebugProperties:
    """
    Singleton class to manage debug properties across all debugger backends.
    Properties control debugger behavior like verbose output, etc.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.verbose = False  # Controls verbose debug output (GDB MI messages, etc.)
        self.profile = False  # Enable cProfile profiling to identify bottlenecks
        self.print_elements = 128  # Max number of array elements to print/fetch in debugger output

    # ------------------------------------------------------------------
    # Instance-level helpers
    # ------------------------------------------------------------------

    def verbose_time(self) -> str:
        """Return formatted current time for verbose output."""
        return f"[time: {time.strftime('%Y-%m-%d %H:%M:%S')}]"

    def verbose_out(
        self,
        verbose_message,
        is_debug_output: bool | None = None,
        payload: str | None = None,
    ) -> None:
        """
        Conditionally print a verbose debug message if verbose mode is enabled.

        :param verbose_message: The message or command being logged
        :param is_debug_output: True when logging debugger output (>>),
                                False when logging commands sent (<<)
        :param payload: Optional extra payload to append to the message
        """
        if self.verbose == False:
            return
        print(self.verbose_time())
        prefix = ""
        if is_debug_output is not None:
            prefix = "[pkdb-gdb" + (">>" if is_debug_output else "<<") + "]"
        print(f"{prefix} {verbose_message} {'' if not payload else payload}", flush=True)

    def set_property(self, name: str, value) -> bool:
        """
        Set a debug property value.

        :param name: Property name (e.g., 'verbose')
        :param value: Property value to set
        :returns: True if property was set successfully, False if property doesn't exist
        """
        if not hasattr(self, name):
            return False

        setattr(self, name, value)
        return True

    def get_property(self, name: str):
        """
        Get a debug property value.

        :param name: Property name (e.g., 'verbose')
        :returns: Property value, or False if property doesn't exist
        """
        return getattr(self, name, False)

    def list_properties(self) -> dict:
        """
        List all available debug properties and their current values.

        :returns: Dictionary of property names and values
        """
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("_") and not callable(getattr(self, attr))
        }


# ----------------------------------------------------------------------
# Singleton accessors and module-level convenience wrappers
# ----------------------------------------------------------------------


def get_debug_properties() -> DebugProperties:
    """Get the global debug properties instance"""
    return DebugProperties()


def run_pkdb_set_command(arg: str, *, writeln=print) -> None:
    """
    Handle ``set`` / ``set <name>`` / ``set <name> <value>`` for pkdb debug properties.

    Used by the pdb-side ``(pkdb)`` prompt (``PdbCommandsMixin.do_set``).
    ``writeln`` should accept a single string (e.g. ``print`` or ``Pdb.message``).
    """
    debug_props = get_debug_properties()

    if not arg:
        props = debug_props.list_properties()
        writeln("Debug properties:")
        for name, value in sorted(props.items()):
            if isinstance(value, bool):
                shown = "on" if value else "off"
            else:
                shown = value
            writeln(f"  {name}: {shown}")
        return

    parts = arg.strip().split(maxsplit=1)
    if len(parts) == 1:
        prop_name = parts[0]
        current_props = debug_props.list_properties()
        if prop_name not in current_props:
            writeln(f"Unknown property: {prop_name}")
            writeln("\nAvailable properties:")
            for name in sorted(current_props.keys()):
                writeln(f"  {name}")
            return
        value = current_props[prop_name]
        if isinstance(value, bool):
            shown = "on" if value else "off"
        else:
            shown = value
        writeln(f"{prop_name}: {shown}")
        return

    prop_name = parts[0]
    prop_value_raw = parts[1].strip()

    current_props = debug_props.list_properties()
    if prop_name not in current_props:
        writeln(f"Unknown property: {prop_name}")
        writeln("\nAvailable properties:")
        for name in sorted(current_props.keys()):
            writeln(f"  {name}")
        return

    current_value = current_props[prop_name]
    if isinstance(current_value, bool):
        prop_value = prop_value_raw.lower()
        if prop_value in ["on", "1", "true"]:
            value = True
        elif prop_value in ["off", "0", "false"]:
            value = False
        else:
            writeln(f"Invalid value: {prop_value_raw}")
            writeln("Usage: set <property> <on|off>")
            return
    elif isinstance(current_value, int):
        try:
            value = int(prop_value_raw)
            if value <= 0:
                raise ValueError
        except ValueError:
            writeln(f"Invalid value: {prop_value_raw}")
            writeln("Usage: set <property> <positive-integer>")
            return
    else:
        value = prop_value_raw

    if debug_props.set_property(prop_name, value):
        if isinstance(value, bool):
            shown = "on" if value else "off"
        else:
            shown = value
        writeln(f"{prop_name}: {shown}")
    else:
        writeln(f"Unknown property: {prop_name}")
        writeln("\nAvailable properties:")
        props = debug_props.list_properties()
        for name in sorted(props.keys()):
            writeln(f"  {name}")


def properties() -> DebugProperties:
    """Alias for get_debug_properties() for shorter access."""
    return DebugProperties()


def verbose_time():
    """
    Module-level helper for backward compatibility.
    Returns formatted current time using the global debug properties instance.
    """
    return get_debug_properties().verbose_time()


def verbose_out(
    verbose_message,
    is_debug_output: bool | None = None,
    payload: str | None = None,
) -> None:
    """
    Module-level helper for backward compatibility.
    Delegates to the global debug properties instance.
    """
    get_debug_properties().verbose_out(
        verbose_message=verbose_message,
        is_debug_output=is_debug_output,
        payload=payload,
    )
