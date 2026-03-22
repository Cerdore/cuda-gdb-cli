"""Session-wide state container for CUDA-GDB-CLI."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import os


@dataclass
class SessionState:
    """Container for session-wide state.

    Holds all state information for a debugging session including
    mode, target info, CUDA metadata, and runtime configuration.
    """

    # Session identification
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    # Target info
    mode: str = "INITIALIZING"  # INITIALIZING, MUTABLE, IMMUTABLE, RUNNING, STOPPED
    target_type: str = ""  # load, attach, coredump
    pid: Optional[int] = None
    executable: Optional[str] = None
    core_file: Optional[str] = None

    # CUDA-specific
    cuda_version: Optional[str] = None
    gpu_device: Optional[str] = None
    compute_capability: Optional[str] = None

    # GDB info
    gdb_version: Optional[str] = None

    # Runtime config
    socket_path: str = ""
    timeout: float = 25.0

    # State flags
    is_running: bool = False
    has_cuda_kernel: bool = False
    last_exception: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "mode": self.mode,
            "target_type": self.target_type,
            "pid": self.pid,
            "executable": self.executable,
            "core_file": self.core_file,
            "cuda_version": self.cuda_version,
            "gpu_device": self.gpu_device,
            "compute_capability": self.compute_capability,
            "gdb_version": self.gdb_version,
            "socket_path": self.socket_path,
            "timeout": self.timeout,
            "is_running": self.is_running,
            "has_cuda_kernel": self.has_cuda_kernel,
        }

    @classmethod
    def from_env(cls) -> "SessionState":
        """Create session state from environment variables.

        Environment variables:
            CUDA_GDB_SOCKET_PATH: Unix socket path for RPC
            CUDA_GDB_SESSION_ID: Session identifier
            CUDA_GDB_TIMEOUT: RPC timeout in seconds
        """
        return cls(
            session_id=os.environ.get("CUDA_GDB_SESSION_ID", ""),
            socket_path=os.environ.get("CUDA_GDB_SOCKET_PATH", ""),
            timeout=float(os.environ.get("CUDA_GDB_TIMEOUT", "25.0")),
        )

    def update_from_target(self, target_info: Dict[str, Any]) -> None:
        """Update state from target detection info."""
        self.mode = target_info.get("mode", self.mode)
        self.target_type = target_info.get("target_type", self.target_type)
        self.pid = target_info.get("pid")
        self.executable = target_info.get("executable")

    def set_running(self, running: bool) -> None:
        """Update running state."""
        self.is_running = running
        if running:
            self.mode = "RUNNING"
        else:
            self.mode = "STOPPED"


# Global session state
_session_state: Optional[SessionState] = None


def get_session_state() -> SessionState:
    """Get the global session state."""
    global _session_state
    if _session_state is None:
        _session_state = SessionState.from_env()
    return _session_state


def reset_session_state() -> None:
    """Reset the global session state."""
    global _session_state
    _session_state = None
