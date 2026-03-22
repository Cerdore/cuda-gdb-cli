"""
RPC Listener Thread - Socket I/O for JSON-RPC communication.

Runs in a gdb.Thread subclass that automatically blocks signals.
All GDB API calls must go through the command queue.
"""

import json
import socket
import threading
import queue
from typing import Dict, Any, Callable, Optional

# GDB Python API - available when running inside cuda-gdb
import gdb


class CommandTask:
    """
    Encapsulates a pending RPC command.

    Attributes:
        request_id: JSON-RPC request ID
        method: Method name
        params: Method parameters
        result: Result data (set by executor)
        error: Error dict (set by executor)
        completed: Event signaling completion
    """

    def __init__(self, request_id: Optional[Any], method: str, params: Dict[str, Any]):
        self.request_id = request_id
        self.method = method
        self.params = params
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None
        self.completed = threading.Event()


class RPCListenerThread(gdb.Thread):
    """
    RPC listener running in a separate gdb.Thread.

    This thread handles socket I/O for JSON-RPC communication.
    It NEVER calls GDB APIs directly - all GDB operations
    are dispatched through the command queue via gdb.post_event().

    Thread Safety:
        - Inherits from gdb.Thread which automatically calls gdb.block_signals()
        - Uses a separate command queue for thread-safe communication
        - Condition variables or events synchronize with GDB main thread
    """

    def __init__(
        self,
        socket_path: str,
        command_queue: queue.Queue,
        tool_handlers: Dict[str, Callable],
        gdb_executor: Any,
        modality_guard: Any,
        debug_mode: bool = False
    ):
        """
        Initialize the RPC listener thread.

        Args:
            socket_path: Unix Domain Socket path for IPC
            command_queue: Thread-safe queue for command tasks
            tool_handlers: Dict mapping method names to handler functions
            gdb_executor: GDB executor instance for scheduling tasks
            modality_guard: Modality guard for permission checking
            debug_mode: Enable debug logging
        """
        super().__init__()
        self.socket_path = socket_path
        self.command_queue = command_queue
        self.tool_handlers = tool_handlers
        self.gdb_executor = gdb_executor
        self.modality_guard = modality_guard
        self.debug_mode = debug_mode

        self._server_socket: Optional[socket.socket] = None
        self._running = threading.Event()
        self._running.clear()

    def run(self) -> None:
        """
        Main loop: accept connections, receive requests, send responses.

        This method runs in the RPC Listener Thread (gdb.Thread).
        It handles socket accept/read/write but dispatches all
        GDB operations to the main thread via the command queue.
        """
        try:
            self._setup_socket()
            self._running.set()

            if self.debug_mode:
                gdb.write(f"[RPC Listener] Started on {self.socket_path}\n")

            while self._running.is_set():
                try:
                    self._server_socket.settimeout(1.0)
                    conn, addr = self._server_socket.accept()
                    if self.debug_mode:
                        gdb.write(f"[RPC Listener] Connection from {addr}\n")

                    # Handle connection in separate method
                    self._handle_connection(conn)

                except socket.timeout:
                    # Accept timeout - check if we should exit
                    continue
                except OSError as e:
                    if self._running.is_set():
                        if self.debug_mode:
                            gdb.write(f"[RPC Listener] Socket error: {e}\n")
                    break

        except Exception as e:
            gdb.write(f"[RPC Listener] Fatal error: {e}\n")
        finally:
            self._cleanup_socket()

    def _setup_socket(self) -> None:
        """Create and bind the Unix Domain Socket."""
        # Create socket
        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        # Set socket options
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind to path
        self._server_socket.bind(self.socket_path)
        self._server_socket.listen(5)

        # Send ready notification
        self._send_ready_notification()

    def _cleanup_socket(self) -> None:
        """Clean up socket resources."""
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None

        # Remove socket file
        try:
            import os
            if hasattr(self, 'socket_path'):
                os.unlink(self.socket_path)
        except Exception:
            pass

    def _send_ready_notification(self) -> None:
        """
        Send RPC ready notification to indicate server is initialized.

        This is sent directly through the socket after binding.
        """
        ready_msg = {
            "jsonrpc": "2.0",
            "method": "__rpc_ready",
            "params": {
                "socket": self.socket_path,
                "version": "1.0"
            }
        }
        # Note: In practice, this would be sent via the transport proxy
        # For now, just log that we're ready
        if self.debug_mode:
            gdb.write("[RPC Listener] Ready notification sent\n")

    def _handle_connection(self, conn: socket.socket) -> None:
        """
        Handle a single client connection.

        Reads JSON-RPC requests, dispatches to command queue,
        waits for result, sends response.

        Args:
            conn: Connected socket
        """
        try:
            while True:
                # Read message
                raw_data = self._read_message(conn)
                if not raw_data:
                    break

                # Parse JSON-RPC request
                try:
                    request = json.loads(raw_data)
                except json.JSONDecodeError as e:
                    self._send_error(conn, None, -32700, f"Parse error: {e}")
                    continue

                # Validate request structure
                if not self._validate_request(request):
                    continue

                # Dispatch to command queue
                response = self._dispatch_request(request)
                if response is not None:
                    self._send_message(conn, json.dumps(response))

        except Exception as e:
            if self.debug_mode:
                gdb.write(f"[RPC Listener] Connection error: {e}\n")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _read_message(self, conn: socket.socket) -> Optional[str]:
        """
        Read a JSON-RPC message from the socket.

        Messages are length-prefixed (4-byte big-endian integer).

        Args:
            conn: Connected socket

        Returns:
            Message string or None if disconnected
        """
        try:
            # Read length prefix (4 bytes)
            length_data = b""
            while len(length_data) < 4:
                chunk = conn.recv(4 - len(length_data))
                if not chunk:
                    return None
                length_data += chunk

            length = int.from_bytes(length_data, byteorder='big')

            # Read message body
            body = b""
            while len(body) < length:
                chunk = conn.recv(length - len(body))
                if not chunk:
                    return None
                body += chunk

            return body.decode('utf-8')

        except Exception as e:
            if self.debug_mode:
                gdb.write(f"[RPC Listener] Read error: {e}\n")
            return None

    def _send_message(self, conn: socket.socket, message: str) -> None:
        """
        Send a JSON-RPC message to the socket.

        Messages are length-prefixed (4-byte big-endian integer).

        Args:
            conn: Connected socket
            message: JSON message string
        """
        try:
            data = message.encode('utf-8')
            length = len(data)
            length_prefix = length.to_bytes(4, byteorder='big')
            conn.sendall(length_prefix + data)
        except Exception as e:
            if self.debug_mode:
                gdb.write(f"[RPC Listener] Send error: {e}\n")

    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """
        Validate JSON-RPC request structure.

        Args:
            request: Parsed request dict

        Returns:
            True if valid, False if invalid
        """
        # Check jsonrpc version
        if request.get("jsonrpc") != "2.0":
            self._send_error(None, request.get("id"), -32600, "Invalid Request: jsonrpc must be '2.0'")
            return False

        # Check method exists
        if "method" not in request:
            self._send_error(None, request.get("id"), -32600, "Invalid Request: missing method")
            return False

        return True

    def _dispatch_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Dispatch request to GDB main thread via command queue.

        Args:
            request: Valid JSON-RPC request

        Returns:
            JSON-RPC response dict, or None for notifications
        """
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        # Notifications (no id) are not responded to
        if request_id is None:
            # But still process them for side effects
            if method in self.tool_handlers:
                self._queue_task(request_id, method, params)
            return None

        # Check if method exists
        if method not in self.tool_handlers:
            return self._error_response(
                request_id, -32601, f"Method not found: {method}"
            )

        # Queue the task and wait for result
        task = self._queue_task(request_id, method, params)

        # Wait for completion
        timeout = getattr(self.gdb_executor, 'timeout', 25)
        if not task.completed.wait(timeout=timeout):
            return self._error_response(
                request_id, -32001, f"Command timed out after {timeout}s"
            )

        # Build response
        if task.error:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": task.error
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": task.result
            }

    def _queue_task(
        self,
        request_id: Optional[Any],
        method: str,
        params: Dict[str, Any]
    ) -> CommandTask:
        """
        Queue a command task to the GDB main thread.

        Args:
            request_id: JSON-RPC request ID
            method: Method name
            params: Method parameters

        Returns:
            CommandTask (with result/error set when completed)
        """
        task = CommandTask(request_id, method, params)

        # Put in queue
        self.command_queue.put(task)

        # Schedule processing in GDB main thread
        gdb.post_event(lambda: self._process_task())

        return task

    def _process_task(self) -> None:
        """
        Process next task from queue (runs in GDB main thread).

        This is called via gdb.post_event() from the RPC listener thread.
        """
        try:
            # Get task from queue (non-blocking)
            try:
                task = self.command_queue.get_nowait()
            except queue.Empty:
                return

            # Check modality permission
            if self.modality_guard:
                permission_error = self.modality_guard.check_permission(task.method)
                if permission_error:
                    task.error = permission_error
                    task.completed.set()
                    return

            # Execute handler
            handler = self.tool_handlers.get(task.method)
            if handler:
                try:
                    task.result = handler(task.params)
                except gdb.error as gdb_err:
                    task.error = self._map_gdb_error(gdb_err, task.method)
                except Exception as exc:
                    task.error = {
                        "code": -32603,
                        "message": str(exc),
                        "data": {"source": "rpc_engine"}
                    }
            else:
                task.error = {
                    "code": -32601,
                    "message": f"Method not found: {task.method}"
                }

        except Exception as e:
            gdb.write(f"[RPC Listener] Task processing error: {e}\n")
        finally:
            # Signal completion
            if 'task' in locals():
                task.completed.set()

    def _map_gdb_error(self, error: Exception, method: str) -> Dict[str, Any]:
        """
        Map gdb.error to JSON-RPC error format.

        Args:
            error: gdb.error exception
            method: Method that was being executed

        Returns:
            JSON-RPC error dict
        """
        error_msg = str(error)

        # Map common error patterns
        if "not within" in error_msg or "invalid" in error_msg.lower():
            return {
                "code": -32000,
                "message": error_msg,
                "data": {
                    "source": "gdb_internal",
                    "error_type": "invalid_coordinates",
                    "hint": "Block/thread coordinates outside valid range"
                }
            }

        if "optimized out" in error_msg:
            return {
                "code": -32005,
                "message": f"Variable '{error_msg.split()[0]}' has been optimized out",
                "data": {
                    "source": "gdb_internal",
                    "error_type": "optimized_out",
                    "hint": "Recompile with -g -G flags"
                }
            }

        if "no active kernel" in error_msg.lower():
            return {
                "code": -32006,
                "message": "No CUDA kernel is currently active",
                "data": {
                    "source": "gdb_internal",
                    "error_type": "no_active_kernel"
                }
            }

        # Default GDB error
        return {
            "code": -32000,
            "message": error_msg,
            "data": {"source": "gdb_internal"}
        }

    def _error_response(self, request_id: Any, code: int, message: str) -> Dict[str, Any]:
        """Build an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

    def _send_error(
        self,
        conn: Optional[socket.socket],
        request_id: Any,
        code: int,
        message: str
    ) -> None:
        """Send error response to client."""
        if conn:
            response = self._error_response(request_id, code, message)
            self._send_message(conn, json.dumps(response))

    def send_notification(self, notification: Dict[str, Any]) -> None:
        """
        Send async notification to client.

        This is called from GDB main thread via callbacks
        (gdb.events.stop, etc.).

        Args:
            notification: JSON-RPC notification dict
        """
        # For now, we need the socket connection to send notifications
        # In a full implementation, this would write to a notification channel
        # that the transport proxy monitors
        if self.debug_mode:
            gdb.write(f"[RPC Listener] Notification: {notification.get('method')}\n")

    def stop(self) -> None:
        """Stop the RPC listener."""
        self._running.clear()
        self._cleanup_socket()
        if self.debug_mode:
            gdb.write("[RPC Listener] Stopped\n")