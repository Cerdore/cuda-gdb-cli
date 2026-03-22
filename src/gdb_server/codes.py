# JSON-RPC 2.0 Error Code Constants

# JSON-RPC 2.0 standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# GDB-specific error codes (reserved range -32000 to -32099)
GDB_ERROR = -32000
TIMEOUT = -32001
PROCESS_CRASHED = -32002
MODALITY_FORBIDDEN = -32003
TARGET_RUNNING = -32004
OPTIMIZED_OUT = -32005
NO_ACTIVE_KERNEL = -32006
MEMORY_TRUNCATED = -32007

# Error code to name mapping for logging/debugging
ERROR_CODE_NAMES = {
    PARSE_ERROR: "ParseError",
    INVALID_REQUEST: "InvalidRequest",
    METHOD_NOT_FOUND: "MethodNotFound",
    INVALID_PARAMS: "InvalidParams",
    INTERNAL_ERROR: "InternalError",
    GDB_ERROR: "GdbError",
    TIMEOUT: "Timeout",
    PROCESS_CRASHED: "ProcessCrashed",
    MODALITY_FORBIDDEN: "ModalityForbidden",
    TARGET_RUNNING: "TargetRunning",
    OPTIMIZED_OUT: "OptimizedOut",
    NO_ACTIVE_KERNEL: "NoActiveKernel",
    MEMORY_TRUNCATED: "MemoryTruncated",
}


def get_error_name(code: int) -> str:
    """Get human-readable name for error code."""
    return ERROR_CODE_NAMES.get(code, f"UnknownError({code})")