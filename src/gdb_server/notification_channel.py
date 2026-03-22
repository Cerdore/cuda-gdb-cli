"""Async notification channel for sending events to the agent."""

import json
import socket
from typing import Dict, Any, Optional


class NotificationChannel:
    """
    Async notification sender for JSON-RPC events.

    Sends asynchronous notifications (not responses) back to the
    agent via the Unix Domain Socket. Used for stop events,
    breakpoint hits, and other GDB-generated events.
    """

    def __init__(self, socket_path: str):
        """
        Initialize the notification channel.

        Args:
            socket_path: Path to the Unix Domain Socket for IPC
        """
        self._socket_path = socket_path
        self._socket: Optional[socket.socket] = None

    def _get_socket(self) -> socket.socket:
        """Get or create the socket connection."""
        if self._socket is None:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._socket.connect(self._socket_path)
        return self._socket

    def send(self, notification: Dict[str, Any]) -> bool:
        """
        Send a JSON-RPC notification.

        Args:
            notification: The notification dict with "jsonrpc", "method", and "params"

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            sock = self._get_socket()
            data = json.dumps(notification) + "\n"
            sock.sendall(data.encode("utf-8"))
            return True
        except Exception:
            return False

    def send_stop_notification(
        self,
        reason: str,
        focus: Optional[Dict[str, Any]] = None,
        exception: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a stop event notification.

        Args:
            reason: The reason for the stop (breakpoint, exception, etc.)
            focus: Current focus coordinates
            exception: Exception details if applicable

        Returns:
            True if sent successfully
        """
        notification = {
            "jsonrpc": "2.0",
            "method": "cuda/stop",
            "params": {
                "reason": reason,
            }
        }

        if focus:
            notification["params"]["focus"] = focus
        if exception:
            notification["params"]["exception"] = exception

        return self.send(notification)

    def close(self) -> None:
        """Close the socket connection."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None