import re


def infer_python_type(cpp_type: str) -> str:
    """
    Infer Python type from C++ type signature.

    Args:
        cpp_type: C++ type string from GDB

    Returns:
        Inferred Python type name
    """
    if not cpp_type:
        return cpp_type

    # Handle Kokkos Views
    if "Kokkos" in cpp_type and "View" in cpp_type:
        if "CudaSpace" in cpp_type or "CudaUVMSpace" in cpp_type:
            return "cupy.ndarray"
        elif "HIPSpace" in cpp_type:
            return "cupy.ndarray"
        elif "HostSpace" in cpp_type:
            return "numpy.ndarray"
        else:
            return "numpy.ndarray"

    # Handle register types and basic types
    if "@register" in cpp_type:
        cpp_type = cpp_type.replace("@register", "").strip()

    # Handle const/generic qualifiers
    cpp_type = re.sub(r"const\s+", "", cpp_type)
    cpp_type = re.sub(r"@generic\s+", "", cpp_type)

    # Map C++ types to Python types
    type_map = {
        "int32_t": "int",
        "int64_t": "int",
        "int16_t": "int",
        "int8_t": "int",
        "uint32_t": "int",
        "uint64_t": "int",
        "uint16_t": "int",
        "uint8_t": "int",
        "int": "int",
        "long": "int",
        "short": "int",
        "float": "float",
        "double": "float",
        "bool": "bool",
        "char": "str",
    }

    for cpp, py in type_map.items():
        if cpp in cpp_type:
            return py

    return cpp_type
