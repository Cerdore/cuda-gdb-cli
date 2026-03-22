"""GDB RPC Server - Embedded JSON-RPC server for cuda-gdb.

This module implements an RPC server that runs inside cuda-gdb's Python
interpreter. It listens on a Unix Domain Socket and handles JSON-RPC 2.0
requests from the cuda-gdb-cli client.

Key Design:
- RPC listener runs in a background thread
- All GDB API calls are dispatched to the main thread via gdb.post_event()
- Thread safety is enforced through GdbExecutor
- ModalityGuard checks permissions before handler execution
"""

import json
import os
import socket
import threading
import traceback
from typing import Any, Callable, Dict, Optional

import gdb

from .codes import (
    GDB_ERROR,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    MODALITY_FORBIDDEN,
    PARSE_ERROR,
    TIMEOUT,
)
from .gdb_errors import map_gdb_error
from .gdb_executor import GdbExecutor
from .json_rpc import (
    create_error_response,
    create_success_response,
    decode_request,
    encode_error_response,
    encode_response,
)
from .modality_guard import get_modality_guard, OperationCategory
from .cuda_handlers import CUDA_HANDLERS


# =============================================================================
# CPU Handlers (inherited from gdb-cli)
# =============================================================================

def handle_backtrace(**kwargs) -> Dict[str, Any]:
    """Show backtrace."""
    full = kwargs.get("full", False)
    try:
        cmd = "backtrace"
        if full:
            cmd += " full"
        output = gdb.execute(cmd, to_string=True)
        return {"output": output, "frames": _parse_backtrace(output)}
    except gdb.error as e:
        return {"error": map_gdb_error(e, "backtrace")}


def handle_threads(**kwargs) -> Dict[str, Any]:
    """List threads."""
    try:
        output = gdb.execute("info threads", to_string=True)
        return {"output": output, "threads": _parse_threads(output)}
    except gdb.error as e:
        return {"error": map_gdb_error(e, "threads")}


def handle_evaluate(**kwargs) -> Dict[str, Any]:
    """Evaluate an expression."""
    expr = kwargs.get("expression")
    if not expr:
        return {"error": {"code": INVALID_PARAMS, "message": "Missing expression"}}
    try:
        val = gdb.parse_and_eval(expr)
        from .value_formatter import serialize_gdb_value
        return {"expression": expr, "value": serialize_gdb_value(val)}
    except gdb.error as e:
        return {"error": map_gdb_error(e, "evaluate")}


def handle_locals(**kwargs) -> Dict[str, Any]:
    """Show local variables."""
    try:
        frame = gdb.selected_frame()
        block = frame.block()
        locals_dict = {}
        from .value_formatter import serialize_gdb_value

        for symbol in block:
            if symbol.is_argument or symbol.is_variable:
                try:
                    val = symbol.value(frame)
                    locals_dict[symbol.name] = serialize_gdb_value(val)
                except Exception:
                    locals_dict[symbol.name] = {"error": "Cannot read value"}

        return {"locals": locals_dict}
    except gdb.error as e:
        return {"error": map_gdb_error(e, "locals")}


def handle_memory(**kwargs) -> Dict[str, Any]:
    """Read memory."""
    address = kwargs.get("address")
    count = kwargs.get("count", 16)
    if not address:
        return {"error": {"code": INVALID_PARAMS, "message": "Missing address"}}
    try:
        cmd = f"x/{count}xb {address}"
        output = gdb.execute(cmd, to_string=True)
        return {"address": address, "count": count, "output": output}
    except gdb.error as e:
        return {"error": map_gdb_error(e, "memory")}


def handle_disassemble(**kwargs) -> Dict[str, Any]:
    """Disassemble current location."""
    count = kwargs.get("count", 10)
    try:
        frame = gdb.selected_frame()
        pc = frame.pc()
        cmd = f"x/{count}i {hex(pc)}"
        output = gdb.execute(cmd, to_string=True)
        return {"address": hex(pc), "count": count, "output": output}
    except gdb.error as e:
        return {"error": map_gdb_error(e, "disassemble")}


def handle_exec(**kwargs) -> Dict[str, Any]:
    """Execute a raw GDB command."""
    command = kwargs.get("command")
    safety_level = kwargs.get("safety_level", "readonly")

    if not command:
        return {"error": {"code": INVALID_PARAMS, "message": "Missing command"}}

    # Check safety
    from .safety import check_command, SafetyLevel
    level_map = {
        "readonly": SafetyLevel.READONLY,
        "readwrite": SafetyLevel.READWRITE,
        "full": SafetyLevel.FULL,
    }
    error = check_command(command, level_map.get(safety_level, SafetyLevel.READONLY))
    if error:
        return {"error": {"code": MODALITY_FORBIDDEN, "message": error}}

    try:
        output = gdb.execute(command, to_string=True)
        return {"command": command, "output": output}
    except gdb.error as e:
        return {"error": map_gdb_error(e, "exec")}


def handle_stop(**kwargs) -> Dict[str, Any]:
    """Stop the debugging session."""
    try:
        gdb.execute("quit", to_string=True)
    except:
        pass
    return {"status": "stopping"}


# Helper functions
def _parse_backtrace(output: str) -> list:
    """Parse backtrace output into frame list."""
    frames = []
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('#'):
            frames.append({"raw": line})
    return frames


def _parse_threads(output: str) -> list:
    """Parse info threads output into thread list."""
    threads = []
    for line in output.split('\n'):
        line = line.strip()
        if line and line[0].isdigit() or line.startswith('*'):
            threads.append({"raw": line})
    return threads


# CPU Handler registry
CPU_HANDLERS: Dict[str, Callable] = {
    "backtrace": handle_backtrace,
    "threads": handle_threads,
    "evaluate": handle_evaluate,
    "locals": handle_locals,
    "memory": handle_memory,
    "disassemble": handle_disassemble,
    "exec": handle_exec,
    "stop": handle_stop,
}


# =============================================================================
# RPC Server
# =============================================================================

class GdbRpcServer:
    """JSON-RPC server embedded in cuda-gdb.

    Runs in a background thread, dispatches all GDB operations
    to the main thread via gdb.post_event().
    """

    def __init__(self, socket_path: str):
        """Initialize the RPC server.

        Args:
            socket_path: Path to Unix Domain Socket
        """
        self.socket_path = socket_path
        self.handlers: Dict[str, Callable] = {}
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.listener_thread: Optional[threading.Thread] = None

        # Register handlers
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register all handlers."""
        # CPU handlers
        self.handlers.update(CPU_HANDLERS)

        # CUDA handlers
        self.handlers.update(CUDA_HANDLERS)

    def start(self) -> None:
        """Start the RPC server."""
        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create Unix Domain Socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # Allow periodic checks

        self.running = True

        # Start listener thread
        self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listener_thread.start()

        print(f"[GDB RPC] Server started on {self.socket_path}")

    def stop(self) -> None:
        """Stop the RPC server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        print("[GDB RPC] Server stopped")

    def _listen_loop(self) -> None:
        """Main listener loop (runs in background thread)."""
        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
                # Handle each connection in a separate thread
                handler_thread = threading.Thread(
                    target=self._handle_connection,
                    args=(client_socket,),
                    daemon=True
                )
                handler_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[GDB RPC] Accept error: {e}")

    def _handle_connection(self, client_socket: socket.socket) -> None:
        """Handle a client connection."""
        try:
            # Receive request
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            request_str = data.decode("utf-8").strip()
            if not request_str:
                return

            # Parse request
            try:
                request = decode_request(request_str)
            except ValueError as e:
                response = encode_error_response(None, PARSE_ERROR, str(e))
                client_socket.sendall((response + "\n").encode())
                return

            # Process request (via main thread)
            response = self._process_request(request)

            # Send response
            if isinstance(response, dict):
                response = json.dumps(response)
            client_socket.sendall((response + "\n").encode())

        except Exception as e:
            print(f"[GDB RPC] Connection error: {e}")
            traceback.print_exc()
        finally:
            client_socket.close()

    def _process_request(self, request: Dict[str, Any]) -> str:
        """Process a JSON-RPC request.

        Dispatches handler execution to the main GDB thread.
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        # Lookup handler
        handler = self.handlers.get(method)
        if not handler:
            return encode_error_response(
                request_id,
                METHOD_NOT_FOUND,
                f"Method not found: {method}"
            )

        # Check modality permissions
        modality_guard = get_modality_guard()
        permission_error = modality_guard.check_permission(method)
        if permission_error:
            return encode_response(create_error_response(
                request_id,
                permission_error["code"],
                permission_error["message"],
                permission_error.get("data")
            ))

        # Execute handler on main thread
        result = GdbExecutor.execute_sync(handler, params)

        # Build response
        if "error" in result:
            error = result["error"]
            if isinstance(error, dict):
                return encode_response(create_error_response(
                    request_id,
                    error.get("code", GDB_ERROR),
                    error.get("message", str(error)),
                    error.get("data")
                ))
            return encode_error_response(request_id, GDB_ERROR, str(error))

        return encode_response(create_success_response(request_id, result.get("result", result)))


# =============================================================================
# Module-level Functions
# =============================================================================

_server: Optional[GdbRpcServer] = None


def start_server(socket_path: str) -> None:
    """Start the RPC server (called from GDB initialization)."""
    global _server
    _server = GdbRpcServer(socket_path)
    _server.start()


def stop_server() -> None:
    """Stop the RPC server."""
    global _server
    if _server:
        _server.stop()
        _server = None


def get_server() -> Optional[GdbRpcServer]:
    """Get the current server instance."""
    return _server