"""State management for CUDA-GDB CLI."""

from .modality_guard import (
    ModalityGuard,
    DebugModality,
    OperationCategory,
    get_modality_guard,
)
from .focus_tracker import (
    FocusTracker,
    SoftwareCoords,
    HardwareCoords,
    get_focus_tracker,
    reset_focus_tracker,
)
from .session_state import (
    SessionState,
    SessionStateManager,
    get_or_create_session,
)

__all__ = [
    # modality_guard
    "ModalityGuard",
    "DebugModality",
    "OperationCategory",
    "get_modality_guard",
    # focus_tracker
    "FocusTracker",
    "SoftwareCoords",
    "HardwareCoords",
    "get_focus_tracker",
    "reset_focus_tracker",
    # session_state
    "SessionState",
    "SessionStateManager",
    "get_or_create_session",
]