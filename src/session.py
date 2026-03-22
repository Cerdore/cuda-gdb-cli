"""Session management for CUDA-GDB-CLI."""

import os
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class SessionMeta:
    """Session metadata."""
    session_id: str
    mode: str  # "core" or "attach"
    binary: Optional[str] = None
    core_file: Optional[str] = None
    pid: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)
    cuda_version: Optional[str] = None
    gpu_device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "binary": self.binary,
            "core_file": self.core_file,
            "pid": self.pid,
            "created_at": self.created_at.isoformat(),
            "cuda_version": self.cuda_version,
            "gpu_device": self.gpu_device,
        }


class SessionManager:
    """Manages debugging sessions.

    Sessions are stored in ~/.cuda-gdb-cli/sessions/
    Each session has a metadata file and a Unix socket.
    """

    SESSION_DIR = Path.home() / ".cuda-gdb-cli" / "sessions"

    def __init__(self):
        """Initialize session manager."""
        self.SESSION_DIR.mkdir(parents=True, exist_ok=True)

    def create_session(self, mode: str, **kwargs) -> SessionMeta:
        """Create a new session.

        Args:
            mode: "core" or "attach"
            **kwargs: Additional metadata

        Returns:
            SessionMeta object
        """
        session_id = self._generate_session_id()
        session = SessionMeta(
            session_id=session_id,
            mode=mode,
            **kwargs
        )
        self._save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[SessionMeta]:
        """Get session by ID."""
        meta_path = self.SESSION_DIR / f"{session_id}.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r") as f:
            data = json.load(f)

        return SessionMeta(
            session_id=data["session_id"],
            mode=data["mode"],
            binary=data.get("binary"),
            core_file=data.get("core_file"),
            pid=data.get("pid"),
            created_at=datetime.fromisoformat(data["created_at"]),
            cuda_version=data.get("cuda_version"),
            gpu_device=data.get("gpu_device"),
        )

    def list_sessions(self) -> list:
        """List all sessions."""
        sessions = []
        for meta_path in self.SESSION_DIR.glob("*.json"):
            try:
                session = self.get_session(meta_path.stem)
                if session:
                    sessions.append(session)
            except Exception:
                pass
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        meta_path = self.SESSION_DIR / f"{session_id}.json"
        if meta_path.exists():
            meta_path.unlink()
            return True
        return False

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return uuid.uuid4().hex[:8]

    def _save_session(self, session: SessionMeta) -> None:
        """Save session metadata to disk."""
        meta_path = self.SESSION_DIR / f"{session.session_id}.json"
        with open(meta_path, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def get_socket_path(self, session_id: str) -> str:
        """Get the Unix socket path for a session."""
        return f"/tmp/cuda-gdb-{session_id}.sock"


# Global session manager
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
