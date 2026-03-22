"""Error types for CUDA-GDB-CLI."""

from typing import Optional, Dict, Any


class CUDAGDBError(Exception):
    """Base exception for CUDA-GDB-CLI."""
    pass


class SessionError(CUDAGDBError):
    """Session-related errors."""
    pass


class ConnectionError(CUDAGDBError):
    """Connection errors."""
    pass


class CommandError(CUDAGDBError):
    """Command execution errors."""
    pass


class SafetyError(CUDAGDBError):
    """Safety policy violations."""
    pass


class CUDAError(CUDAGDBError):
    """CUDA-specific errors."""
    pass


class CUDAMemoryError(CUDAError):
    """CUDA memory access errors."""
    pass


# JSON-RPC error codes
class ErrorCodes:
    """JSON-RPC 2.0 error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Custom error codes (-32000 to -32099)
    GDB_ERROR = -32000
    TIMEOUT = -32001
    PROCESS_CRASHED = -32002
    MODALITY_FORBIDDEN = -32003
    TARGET_RUNNING = -32004
    OPTIMIZED_OUT = -32005
    NO_ACTIVE_KERNEL = -32006
    MEMORY_TRUNCATED = -32007
    CUDA_ERROR = -32010
    CUDA_MEMORY_ERROR = -32011


def error_response(code: int, message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a JSON-RPC error response.

    Args:
        code: Error code
        message: Error message
        data: Optional additional data

    Returns:
        Error dict
    """
    error = {
        "code": code,
        "message": message,
    }
    if data:
        error["data"] = data
    return {"error": error}
