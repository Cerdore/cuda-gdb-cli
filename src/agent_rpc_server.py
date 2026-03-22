"""
CUDA-GDB-CLI Agent RPC Server

Entry point for the embedded Python RPC engine that runs inside cuda-gdb.
Parses environment variables, detects debug modality, and initializes the RPC system.

Usage:
    cuda-gdb -x agent_rpc_server.py --args <executable> [args...]
    cuda-gdb -x agent_rpc_server.py -p <pid>
    cuda-gdb -x agent_rpc_server.py <executable> -c <coredump_file>
"""

import os
import sys
import json
import socket
import threading
import queue
from typing import Optional, Dict, Any, Callable

# These imports are available inside cuda-gdb's embedded Python
import gdb

# Import our modules
from rpc_listener import RPCListenerThread
from gdb_executor import GDBExecutor, CommandTask
from modality_guard import ModalityGuard, DebugModality
from serializer import GdbValueSerializer


class AgentRPCServer:
    """
    Main RPC server that coordinates all components.
    Runs inside cuda-gdb's Python interpreter.
    """

    # Configuration from environment variables
    DEFAULT_SOCKET_PATH_TEMPLATE = "/tmp/cuda-gdb-agent-{pid}.sock"
    DEFAULT_INTERNAL_TIMEOUT = 25  # seconds

    def __init__(self):
        self.socket_path: str = ""
        self.internal_timeout: int = self.DEFAULT_INTERNAL_TIMEOUT
        self.debug_mode: bool = False

        # Core components
        self.modality_guard: Optional[ModalityGuard] = None
        self.gdb_executor: Optional[GDBExecutor] = None
        self.rpc_listener: Optional[RPCListenerThread] = None

        # Thread synchronization
        self.command_queue: queue.Queue = queue.Queue()
        self.startup_complete = threading.Event()

        # Tool handlers registry
        self.tool_handlers: Dict[str, Callable] = {}

    def parse_environment(self):
        """Parse configuration from environment variables."""
        # Socket path for IPC
        socket_template = os.environ.get(
            "CUDA_GDB_AGENT_SOCKET",
            self.DEFAULT_SOCKET_PATH_TEMPLATE
        )
        pid = os.getpid()
        self.socket_path = socket_template.format(pid=pid)

        # Timeout configuration
        timeout_str = os.environ.get("RPC_TIMEOUT_SECONDS", "")
        if timeout_str:
            try:
                self.internal_timeout = int(timeout_str)
            except ValueError:
                pass

        # Debug mode
        self.debug_mode = os.environ.get("LOG_LEVEL", "") == "DEBUG"

    def detect_debug_modality(self) -> Dict[str, Any]:
        """
        Detect whether we're in Live debugging or Coredump (post-mortem) mode.

        Returns:
            Dict with mode information and capability flags
        """
        if self.modality_guard is None:
            self.modality_guard = ModalityGuard()

        return self.modality_guard.detect_modality()

    def register_tool_handlers(self):
        """Register all available tool handlers."""
        # Import tool handlers (will be implemented in separate module)
        try:
            from tool_handlers import register_all_handlers
            self.tool_handlers = register_all_handlers(self.gdb_executor, self.modality_guard)
        except ImportError:
            # Tool handlers not yet implemented - register minimal set
            self.tool_handlers = {
                "__rpc_ready": self._handle_rpc_ready,
                "__get_capabilities": self._handle_get_capabilities,
            }

    def _handle_rpc_ready(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RPC ready notification (internal use)."""
        return {"status": "ready", "socket": self.socket_path}

    def _handle_get_capabilities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get server capabilities."""
        modality_info = self.detect_debug_modality()
        return {
            "capabilities": modality_info.get("capabilities", {}),
            "mode": modality_info.get("mode", "UNKNOWN"),
            "server_version": "1.0",
        }

    def initialize(self):
        """Initialize all components."""
        # Parse environment configuration
        self.parse_environment()

        # Initialize modality guard
        self.modality_guard = ModalityGuard()

        # Initialize GDB executor
        self.gdb_executor = GDBExecutor(
            command_queue=self.command_queue,
            modality_guard=self.modality_guard,
            timeout=self.internal_timeout
        )

        # Register tool handlers
        self.register_tool_handlers()

        # Detect debug modality
        modality_info = self.detect_debug_modality()
        if self.debug_mode:
            print(f"[RPC Server] Debug modality: {modality_info.get('mode', 'UNKNOWN')}", file=sys.stderr)

    def start(self):
        """Start the RPC server and listener thread."""
        if self.debug_mode:
            print(f"[RPC Server] Starting with socket: {self.socket_path}", file=sys.stderr)

        # Initialize RPC listener thread
        self.rpc_listener = RPCListenerThread(
            socket_path=self.socket_path,
            command_queue=self.command_queue,
            tool_handlers=self.tool_handlers,
            gdb_executor=self.gdb_executor,
            modality_guard=self.modality_guard,
            debug_mode=self.debug_mode
        )

        # Start the listener (this runs in a separate thread managed by gdb)
        self.rpc_listener.start()

        # Wait for startup to complete
        self.startup_complete.wait(timeout=5)

        if self.debug_mode:
            print("[RPC Server] Started successfully", file=sys.stderr)

    def shutdown(self):
        """Gracefully shutdown the RPC server."""
        if self.rpc_listener:
            self.rpc_listener.stop()

        if self.debug_mode:
            print("[RPC Server] Shutdown complete", file=sys.stderr)


# Global server instance
_server: Optional[AgentRPCServer] = None


def initialize():
    """Initialize the RPC server (called by cuda-gdb on script load)."""
    global _server
    _server = AgentRPCServer()
    _server.initialize()
    return _server


def start():
    """Start the RPC server (called after target is loaded)."""
    global _server
    if _server:
        _server.start()


def get_server() -> Optional[AgentRPCServer]:
    """Get the global server instance."""
    return _server


def get_modality_guard() -> Optional[ModalityGuard]:
    """Get the modality guard instance."""
    global _server
    return _server.modality_guard if _server else None


def get_gdb_executor() -> Optional[GDBExecutor]:
    """Get the GDB executor instance."""
    global _server
    return _server.gdb_executor if _server else None


# Auto-initialize when loaded in cuda-gdb
def register():
    """
    Register the RPC server with cuda-gdb.
    This function is called automatically when the script is loaded.
    """
    # Initialize the server
    server = initialize()

    # Register stop event handler for async notifications
    from async_event_dispatcher import AsyncEventDispatcher
    dispatcher = AsyncEventDispatcher(server.rpc_listener)

    # Connect to GDB events
    gdb.events.stop.connect(dispatcher.on_stop)
    gdb.events.exited.connect(dispatcher.on_exit)

    # Start the server
    start()

    if server.debug_mode:
        print("[RPC Server] Registered with cuda-gdb", file=sys.stderr)


# This is the entry point that cuda-gdb will call
if __name__ == "__register__":
    register()