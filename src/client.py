"""Unix Domain Socket client for communicating with GDB RPC server."""

import json
import socket
import os
from typing import Any, Dict, Optional
from pathlib import Path


class GDBClient:
    """Client for communicating with the embedded GDB RPC server.
    
    The RPC server runs inside cuda-gdb and listens on a Unix Domain Socket.
    This client connects to it and sends JSON-RPC requests.
    """

    DEFAULT_TIMEOUT = 30.0

    def __init__(self, session_id: str, timeout: float = DEFAULT_TIMEOUT):
        """Initialize the client.
        
        Args:
            session_id: Session identifier
            timeout: Socket timeout in seconds
        """
        self.session_id = session_id
        self.timeout = timeout
        self._socket_path = self._get_socket_path(session_id)
        self._socket: Optional[socket.socket] = None

    def _get_socket_path(self, session_id: str) -> str:
        """Get the Unix socket path for a session."""
        # Socket path is based on session ID
        return f"/tmp/cuda-gdb-{session_id}.sock"

    def connect(self) -> None:
        """Connect to the RPC server."""
        if self._socket is not None:
            return

        if not os.path.exists(self._socket_path):
            raise ConnectionError(f"Socket not found: {self._socket_path}")

        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(self.timeout)
        self._socket.connect(self._socket_path)

    def close(self) -> None:
        """Close the connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def call(self, method: str, **params) -> Dict[str, Any]:
        """Send a JSON-RPC request and return the response.
        
        Args:
            method: RPC method name
            **params: Method parameters
            
        Returns:
            Response dict
        """
        self.connect()

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }

        # Send request
        message = json.dumps(request) + "\n"
        self._socket.sendall(message.encode("utf-8"))

        # Receive response
        response_data = b""
        while True:
            chunk = self._socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
            if b"\n" in response_data:
                break

        response_str = response_data.decode("utf-8").strip()
        response = json.loads(response_str)

        if "error" in response:
            return {"error": response["error"]}
        return response.get("result", {})

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def get_client(session_id: str) -> GDBClient:
    """Get a client for the given session."""
    return GDBClient(session_id)
