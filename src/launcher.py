"""Launcher for cuda-gdb processes.

This module handles starting cuda-gdb with the embedded RPC server,
for both coredump analysis and live process attach modes.
"""

import os
import subprocess
import tempfile
import time
from typing import Optional, Dict, Any
from pathlib import Path

from .session import get_session_manager, SessionMeta


class GDBProcess:
    """Represents a running cuda-gdb process."""

    def __init__(self, session: SessionMeta, process: subprocess.Popen):
        """Initialize GDB process."""
        self.session = session
        self.process = process
        self.socket_path = f"/tmp/cuda-gdb-{session.session_id}.sock"

    def is_running(self) -> bool:
        """Check if the process is still running."""
        return self.process.poll() is None

    def terminate(self) -> None:
        """Terminate the process."""
        if self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def wait_for_socket(self, timeout: float = 10.0) -> bool:
        """Wait for the Unix socket to be created."""
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self.socket_path):
                return True
            time.sleep(0.1)
        return False


# Global process registry
_processes: Dict[str, GDBProcess] = {}


def launch_core(
    binary: str,
    core: str,
    gdb_path: str = "cuda-gdb",
    sysroot: Optional[str] = None,
    solib_prefix: Optional[str] = None,
    source_dir: Optional[str] = None,
    timeout: int = 600,
    cuda_memcheck: bool = False,
) -> Dict[str, Any]:
    """Launch cuda-gdb in coredump mode.

    Args:
        binary: Path to the binary file
        core: Path to the core dump file
        gdb_path: Path to cuda-gdb executable
        sysroot: Optional sysroot path
        solib_prefix: Optional shared library prefix
        source_dir: Optional source directory
        timeout: Session timeout in seconds
        cuda_memcheck: Enable CUDA memcheck

    Returns:
        Dict with session_id and status
    """
    # Create session
    session_manager = get_session_manager()
    session = session_manager.create_session(
        mode="core",
        binary=binary,
        core_file=core
    )

    # Build GDB commands
    gdb_commands = _build_core_commands(
        session=session,
        binary=binary,
        core=core,
        sysroot=sysroot,
        solib_prefix=solib_prefix,
        source_dir=source_dir,
        cuda_memcheck=cuda_memcheck,
    )

    # Start cuda-gdb
    process = _start_gdb_process(gdb_path, gdb_commands)

    # Register process
    gdb_proc = GDBProcess(session, process)
    _processes[session.session_id] = gdb_proc

    # Wait for socket
    if gdb_proc.wait_for_socket():
        return {
            "session_id": session.session_id,
            "mode": "core",
            "status": "started",
            "socket_path": gdb_proc.socket_path,
        }
    else:
        gdb_proc.terminate()
        return {
            "error": "Failed to start RPC server",
            "session_id": session.session_id,
        }


def launch_attach(
    pid: int,
    binary: Optional[str] = None,
    gdb_path: str = "cuda-gdb",
    scheduler_locking: bool = True,
    non_stop: bool = True,
    timeout: int = 600,
    allow_write: bool = False,
    cuda_software_preemption: bool = False,
) -> Dict[str, Any]:
    """Launch cuda-gdb in attach mode.

    Args:
        pid: Process ID to attach
        binary: Optional binary file path
        gdb_path: Path to cuda-gdb executable
        scheduler_locking: Enable scheduler locking
        non_stop: Use non-stop mode
        timeout: Session timeout in seconds
        allow_write: Allow memory writes
        cuda_software_preemption: Enable CUDA software preemption

    Returns:
        Dict with session_id and status
    """
    # Create session
    session_manager = get_session_manager()
    session = session_manager.create_session(
        mode="attach",
        binary=binary,
        pid=pid
    )

    # Build GDB commands
    gdb_commands = _build_attach_commands(
        session=session,
        pid=pid,
        binary=binary,
        scheduler_locking=scheduler_locking,
        non_stop=non_stop,
        cuda_software_preemption=cuda_software_preemption,
    )

    # Start cuda-gdb
    process = _start_gdb_process(gdb_path, gdb_commands)

    # Register process
    gdb_proc = GDBProcess(session, process)
    _processes[session.session_id] = gdb_proc

    # Wait for socket
    if gdb_proc.wait_for_socket():
        return {
            "session_id": session.session_id,
            "mode": "attach",
            "status": "started",
            "socket_path": gdb_proc.socket_path,
        }
    else:
        gdb_proc.terminate()
        return {
            "error": "Failed to start RPC server",
            "session_id": session.session_id,
        }


def stop_session(session_id: str) -> Dict[str, Any]:
    """Stop a debugging session.

    Args:
        session_id: Session identifier

    Returns:
        Dict with status
    """
    if session_id not in _processes:
        return {"error": f"Session {session_id} not found"}

    gdb_proc = _processes[session_id]
    gdb_proc.terminate()

    # Clean up
    del _processes[session_id]
    session_manager = get_session_manager()
    session_manager.delete_session(session_id)

    # Remove socket
    if os.path.exists(gdb_proc.socket_path):
        os.unlink(gdb_proc.socket_path)

    return {
        "session_id": session_id,
        "status": "stopped"
    }


def _build_core_commands(
    session: SessionMeta,
    binary: str,
    core: str,
    sysroot: Optional[str] = None,
    solib_prefix: Optional[str] = None,
    source_dir: Optional[str] = None,
    cuda_memcheck: bool = False,
) -> list:
    """Build GDB commands for coredump mode."""
    commands = [
        "set pagination off",
        "set print elements 0",
        "set confirm off",
    ]

    if cuda_memcheck:
        commands.append("set cuda memcheck on")

    if sysroot:
        commands.append(f"set sysroot {sysroot}")
    if solib_prefix:
        commands.append(f"set solib-absolute-prefix {solib_prefix}")
    if source_dir:
        commands.append(f"directory {source_dir}")

    commands.append(f"file {binary}")
    commands.append(f"core-file {core}")

    # Add RPC server startup commands
    commands.extend(_build_server_commands(session))

    return commands


def _build_attach_commands(
    session: SessionMeta,
    pid: int,
    binary: Optional[str] = None,
    scheduler_locking: bool = True,
    non_stop: bool = True,
    cuda_software_preemption: bool = False,
) -> list:
    """Build GDB commands for attach mode."""
    commands = [
        "set pagination off",
        "set print elements 0",
        "set confirm off",
    ]

    if cuda_software_preemption:
        commands.append("set cuda software_preemption on")

    if scheduler_locking:
        commands.append("set scheduler-locking on")

    if non_stop:
        commands.append("set target-async on")
        commands.append("set non-stop on")

    if binary:
        commands.append(f"file {binary}")

    commands.append(f"attach {pid}")

    # Add RPC server startup commands
    commands.extend(_build_server_commands(session))

    return commands


def _build_server_commands(session: SessionMeta) -> list:
    """Build commands to start the embedded RPC server."""
    socket_path = f"/tmp/cuda-gdb-{session.session_id}.sock"

    # The RPC server is a Python script injected into GDB
    server_script = _get_server_script_path()

    return [
        f"python import sys; sys.path.insert(0, '{server_script}')",
        f"python from gdb_server.gdb_rpc_server import start_server; start_server('{socket_path}')",
    ]


def _get_server_script_path() -> str:
    """Get the path to the gdb_server module."""
    # The gdb_server module is in the same package
    return str(Path(__file__).parent)


def _start_gdb_process(gdb_path: str, commands: list) -> subprocess.Popen:
    """Start a cuda-gdb process with the given commands."""
    args = [gdb_path, "-nx", "-q"]

    for cmd in commands:
        args.extend(["-ex", cmd])

    # Create a FIFO for keeping GDB alive
    fifo_path = tempfile.mktemp(prefix="cuda-gdb-fifo-")
    os.mkfifo(fifo_path)

    # Start process
    process = subprocess.Popen(
        args,
        stdin=open(fifo_path, "r"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return process
