"""
Transport Proxy - Process lifecycle management, timeout handling.

Manages cuda-gdb subprocess lifecycle, IPC communication, and timeout handling.
This runs as a standalone process outside cuda-gdb.
"""

import os
import sys
import json
import socket
import signal
import subprocess
import threading
import time
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransportProxy:
    """
    Transport Proxy - Host-side process lifecycle manager.

    Responsibilities:
    1. Start cuda-gdb subprocess with embedded RPC server
    2. Manage IPC channel (Unix Domain Socket)
    3. Handle timeouts and crash recovery
    4. Forward MCP client requests to RPC engine
    """

    DEFAULT_TIMEOUT = 30  # seconds
    GRACE_PERIOD = 5  # seconds after SIGINT

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cuda_gdb_path = self.config.get("cuda_gdb_path", "cuda-gdb")
        self.timeout = self.config.get("timeout", self.DEFAULT_TIMEOUT)
        self.grace_period = self.config.get("grace_period", self.GRACE_PERIOD)

        self.socket_path = self.config.get("socket_path")
        self.process: Optional[subprocess.Popen] = None
        self.rpc_socket: Optional[socket.socket] = None
        self.is_running = False

    def start_live(self, executable: str, args: list = None, pid: int = None):
        """
        Start live debugging session.

        Args:
            executable: Path to CUDA executable
            args: Command line arguments for executable
            pid: Optional PID to attach to
        """
        if self.socket_path is None:
            self.socket_path = f"/tmp/cuda-gdb-agent-{os.getpid()}.sock"

        # Build cuda-gdb command
        rpc_script = os.path.join(
            os.path.dirname(__file__),
            "..",
            "agent_rpc_server.py"
        )

        cmd = [
            self.cuda_gdb_path,
            "-x", rpc_script,
        ]

        if pid:
            cmd.extend(["-p", str(pid)])
        elif executable:
            cmd.append(executable)
            if args:
                cmd.extend(args)

        # Set environment for RPC server
        env = os.environ.copy()
        env["CUDA_GDB_AGENT_SOCKET"] = self.socket_path

        logger.info(f"Starting cuda-gdb: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        # Wait for RPC server to be ready
        self._wait_for_rpc_ready()
        self.is_running = True
        logger.info("cuda-gdb started successfully")

    def start_coredump(self, executable: str, corefile: str):
        """Start coredump analysis session."""
        if self.socket_path is None:
            self.socket_path = f"/tmp/cuda-gdb-agent-{os.getpid()}.sock"

        rpc_script = os.path.join(
            os.path.dirname(__file__),
            "..",
            "agent_rpc_server.py"
        )

        cmd = [
            self.cuda_gdb_path,
            "-x", rpc_script,
            executable,
            "-c", corefile
        ]

        env = os.environ.copy()
        env["CUDA_GDB_AGENT_SOCKET"] = self.socket_path

        logger.info(f"Starting cuda-gdb for coredump: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )

        self._wait_for_rpc_ready()
        self.is_running = True
        logger.info("cuda-gdb coredump session started")

    def _wait_for_rpc_ready(self, timeout: int = 10):
        """Wait for RPC server to send ready notification."""
        start_time = time.time()

        # Create UDS server to receive ready signal
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.settimeout(timeout)
        server.bind(self.socket_path)
        server.listen(1)

        try:
            conn, _ = server.accept()
            data = conn.recv(1024)
            if data:
                msg = json.loads(data)
                if msg.get("method") == "__rpc_ready":
                    logger.info("RPC server ready")
                    return
        except socket.timeout:
            raise RuntimeError("Timeout waiting for RPC server ready")
        finally:
            server.close()

    def send_request(self, method: str, params: Dict = None, request_id: int = 1) -> Dict:
        """
        Send JSON-RPC request to RPC engine and get response.

        Args:
            method: RPC method name
            params: Method parameters
            request_id: Request ID

        Returns:
            JSON-RPC response dict
        """
        if not self.is_running:
            raise RuntimeError("Transport proxy not started")

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }

        # Connect to RPC server
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(self.timeout)

        try:
            client.connect(self.socket_path)
            client.sendall(json.dumps(request).encode())

            # Wait for response with timeout handling
            response = self._recv_with_timeout(client)

            return json.loads(response)

        except socket.timeout:
            # Timeout - try to interrupt
            self._handle_timeout()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,
                    "message": f"Command timed out after {self.timeout}s",
                    "data": {
                        "hint": "The debugger may be blocked by a long-running kernel. "
                                "Consider using cuda_execution_control with action='interrupt'."
                    }
                }
            }
        finally:
            client.close()

    def _recv_with_timeout(self, sock: socket.socket) -> str:
        """Receive data with timeout handling."""
        chunks = []
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            except socket.timeout:
                break
        return b"".join(chunks).decode()

    def _handle_timeout(self):
        """Handle request timeout - try to interrupt blocking command."""
        logger.warning("Request timeout, attempting to interrupt...")

        if self.process and self.process.poll() is None:
            # Send SIGINT to try to unblock
            self.process.send_signal(signal.SIGINT)

            # Wait grace period
            time.sleep(self.grace_period)

            # Check if still running
            if self.process.poll() is None:
                logger.error("Failed to interrupt, process still running")

    def check_crash(self) -> Optional[Dict]:
        """
        Check if cuda-gdb process has crashed.

        Returns:
            Crash info dict if crashed, None otherwise
        """
        if self.process is None:
            return None

        returncode = self.process.poll()
        if returncode is not None:
            self.is_running = False

            # Get stderr
            try:
                stderr = self.process.stderr.read() if self.process.stderr else ""
            except:
                stderr = ""

            return {
                "exit_code": returncode,
                "crashed": returncode < 0,
                "signal": signal.Signals(-returncode).name if returncode < 0 else None,
                "stderr": stderr[-2000:]  # Last 2000 chars
            }

        return None

    def shutdown(self):
        """Shutdown transport proxy and cuda-gdb process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        # Clean up socket
        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except:
                pass

        self.is_running = False
        logger.info("Transport proxy shutdown complete")


def create_from_args(args) -> TransportProxy:
    """Create TransportProxy from command line arguments."""
    config = {
        "cuda_gdb_path": args.cuda_gdb_path,
        "timeout": args.timeout,
        "socket_path": args.socket_path,
    }
    return TransportProxy(config)


def main():
    """CLI entry point for transport proxy."""
    import argparse

    parser = argparse.ArgumentParser(description="CUDA-GDB-CLI Transport Proxy")
    parser.add_argument("--mode", choices=["live", "coredump"], required=True)
    parser.add_argument("--executable", help="Path to CUDA executable")
    parser.add_argument("--args", nargs="*", help="Executable arguments")
    parser.add_argument("--pid", type=int, help="PID to attach to")
    parser.add_argument("--corefile", help="Coredump file path")
    parser.add_argument("--cuda-gdb-path", default="cuda-gdb")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--socket-path")

    args = parser.parse_args()

    proxy = create_from_args(args)

    try:
        if args.mode == "live":
            proxy.start_live(args.executable, args.args, args.pid)
        else:
            proxy.start_coredump(args.executable, args.corefile)

        # Keep running until interrupted
        while proxy.is_running:
            time.sleep(1)
            crash = proxy.check_crash()
            if crash:
                logger.error(f"cuda-gdb crashed: {crash}")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        proxy.shutdown()


if __name__ == "__main__":
    main()