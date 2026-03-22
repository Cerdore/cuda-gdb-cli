"""Session State - Session-wide state container."""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionState:
    """
    Session-wide state container for CUDA-GDB debugging.

    Holds all persistent state for a debugging session.
    """

    session_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # Mode and target info
    mode: str = "initializing"
    target_type: str = "unknown"  # load, attach, coredump

    # CUDA-specific state
    cuda_version: Optional[str] = None
    gpu_device: Optional[Dict[str, Any]] = None

    # Execution state
    is_running: bool = False
    has_active_kernel: bool = False

    # Focus state
    current_kernel: Optional[int] = None
    current_block: Optional[tuple] = None
    current_thread: Optional[tuple] = None

    # Breakpoints
    breakpoints: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Statistics
    request_count: int = 0
    last_request_time: Optional[datetime] = None

    # Custom data
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "mode": self.mode,
            "target_type": self.target_type,
            "cuda_version": self.cuda_version,
            "gpu_device": self.gpu_device,
            "is_running": self.is_running,
            "has_active_kernel": self.has_active_kernel,
            "current_kernel": self.current_kernel,
            "current_block": self.current_block,
            "current_thread": self.current_thread,
            "breakpoints": self.breakpoints,
            "request_count": self.request_count,
            "last_request_time": (
                self.last_request_time.isoformat()
                if self.last_request_time else None
            ),
            "custom": self.custom,
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def increment_request(self) -> None:
        """Increment request counter."""
        self.request_count += 1
        self.last_request_time = datetime.now()


class SessionStateManager:
    """Manages session state."""

    _sessions: Dict[str, SessionState] = {}

    @classmethod
    def create_session(cls, session_id: str) -> SessionState:
        """Create a new session."""
        session = SessionState(session_id=session_id)
        cls._sessions[session_id] = session
        return session

    @classmethod
    def get_session(cls, session_id: str) -> Optional[SessionState]:
        """Get session by ID."""
        return cls._sessions.get(session_id)

    @classmethod
    def remove_session(cls, session_id: str) -> None:
        """Remove session."""
        cls._sessions.pop(session_id, None)

    @classmethod
    def list_sessions(cls) -> list:
        """List all session IDs."""
        return list(cls._sessions.keys())


def get_or_create_session(session_id: str) -> SessionState:
    """Get or create a session."""
    return SessionStateManager.get_session(session_id) or SessionStateManager.create_session(session_id)