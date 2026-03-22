"""Error mapping from gdb.error to JSON-RPC errors."""

from typing import Optional, Dict, Any

from .codes import (
    GDB_ERROR,
    OPTIMIZED_OUT,
    NO_ACTIVE_KERNEL,
    MODALITY_FORBIDDEN,
    TARGET_RUNNING,
)


def map_gdb_error(error: Exception, method: str = "unknown") -> Dict[str, Any]:
    """
    Map gdb.error to structured JSON-RPC error.

    Args:
        error: The gdb.error exception
        method: The RPC method that was being called

    Returns:
        JSON-RPC error response dict with code, message, and data fields
    """
    error_msg = str(error)
    error_type = "unknown"
    hint = "Check GDB documentation for details."

    # Determine error type from message patterns
    if "not within" in error_msg or "invalid coordinates" in error_msg.lower():
        error_type = "invalid_coordinates"
        hint = "Verify the thread/block coordinates are valid for the current kernel."
    elif "optimized out" in error_msg.lower():
        error_type = "optimized_out"
        hint = "Variable was optimized away. Try compiling with -O0."
    elif "no active kernel" in error_msg.lower():
        error_type = "no_active_kernel"
        hint = "No GPU kernel is currently active. Use 'cuda kernel' to switch to a kernel."
    elif "Cannot access memory" in error_msg:
        error_type = "memory_access_denied"
        hint = "Memory access denied. Check address space permissions."
    elif "not accessible" in error_msg.lower():
        error_type = "inaccessible"
        hint = "Variable is not accessible in the current context."
    elif "unavailable" in error_msg.lower():
        error_type = "unavailable"
        hint = "Value is currently unavailable."
    else:
        error_type = "gdb_internal"
        hint = "Internal GDB error. Check GDB output for details."

    return {
        "code": GDB_ERROR,
        "message": f"GDB error: {error_msg}",
        "data": {
            "source": "gdb_internal",
            "error_type": error_type,
            "method": method,
            "hint": hint,
            "details": {"original_error": error_msg},
        },
    }


def error_invalid_coordinates(msg: str) -> Dict[str, Any]:
    """Create error for invalid thread/block coordinates."""
    return {
        "code": MODALITY_FORBIDDEN,
        "message": f"Invalid coordinates: {msg}",
        "data": {
            "source": "modality_guard",
            "error_type": "invalid_coordinates",
            "hint": "Verify block/thread coordinates are valid.",
        },
    }


def error_no_active_kernel(msg: str = "No active kernel") -> Dict[str, Any]:
    """Create error for no active kernel."""
    return {
        "code": NO_ACTIVE_KERNEL,
        "message": msg,
        "data": {
            "source": "modality_guard",
            "error_type": "no_active_kernel",
            "hint": "Switch to a kernel with 'cuda kernel <id>' or list kernels with 'info cuda kernels'.",
        },
    }


def error_modality_forbidden(method: str, reason: str) -> Dict[str, Any]:
    """Create error for modality-forbidden operation."""
    return {
        "code": MODALITY_FORBIDDEN,
        "message": f"Operation '{method}' forbidden: {reason}",
        "data": {
            "source": "modality_guard",
            "error_type": "modality_forbidden",
            "method": method,
            "hint": f"Cannot execute '{method}' in current mode. {reason}",
        },
    }


def error_target_running() -> Dict[str, Any]:
    """Create error for target already running."""
    return {
        "code": TARGET_RUNNING,
        "message": "Target is already running",
        "data": {
            "source": "gdb_executor",
            "error_type": "target_running",
            "hint": "Interrupt the target first with 'interrupt' or wait for it to stop.",
        },
    }


def error_optimized_out(var_name: str) -> Dict[str, Any]:
    """Create error for optimized out variable."""
    return {
        "code": OPTIMIZED_OUT,
        "message": f"Variable '{var_name}' is optimized out",
        "data": {
            "source": "serializer",
            "error_type": "optimized_out",
            "hint": "Recompile with -O0 to see variable values.",
        },
    }


def error_generic_gdb(error_msg: str, method: str = "unknown") -> Dict[str, Any]:
    """Create generic GDB error response."""
    return {
        "code": GDB_ERROR,
        "message": f"GDB error: {error_msg}",
        "data": {
            "source": "gdb_internal",
            "error_type": "generic",
            "method": method,
            "hint": "Check GDB output for details.",
            "details": {"original_error": error_msg},
        },
    }