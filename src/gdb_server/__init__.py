"""GDB Server package - embedded RPC server for cuda-gdb.

This package contains the core components for the cuda-gdb RPC server:

- gdb_rpc_server: Main RPC server implementation
- gdb_executor: Thread-safe GDB command execution
- cuda_handlers: CUDA-specific command handlers
- modality_guard: FSM for mode detection and permissions
- value_formatter: gdb.Value to JSON serialization
- json_rpc: JSON-RPC 2.0 encoding/decoding
- focus_tracker: GPU thread focus tracking
"""

from .gdb_rpc_server import start_server, stop_server, get_server, GdbRpcServer
from .gdb_executor import GdbExecutor
from .modality_guard import get_modality_guard, ModalityGuard, DebugModality
from .cuda_handlers import CUDA_HANDLERS, get_cuda_handler, list_cuda_handlers
from .value_formatter import serialize_gdb_value, GdbValueSerializer
from .json_rpc import (
    encode_response,
    encode_error_response,
    decode_request,
    create_success_response,
    create_error_response,
)
from .focus_tracker import get_focus_tracker, FocusTracker
from .codes import get_error_name

__all__ = [
    # RPC Server
    "start_server",
    "stop_server",
    "get_server",
    "GdbRpcServer",
    # Executor
    "GdbExecutor",
    # Modality Guard
    "get_modality_guard",
    "ModalityGuard",
    "DebugModality",
    # CUDA Handlers
    "CUDA_HANDLERS",
    "get_cuda_handler",
    "list_cuda_handlers",
    # Value Formatter
    "serialize_gdb_value",
    "GdbValueSerializer",
    # JSON-RPC
    "encode_response",
    "encode_error_response",
    "decode_request",
    "create_success_response",
    "create_error_response",
    # Focus Tracker
    "get_focus_tracker",
    "FocusTracker",
    # Utils
    "get_error_name",
]