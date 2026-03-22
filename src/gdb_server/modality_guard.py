"""Modality Guard - FSM for load/attach mode detection."""

from enum import Enum, auto
from typing import Optional, Dict, Any, List, Callable
import gdb


class DebugModality(Enum):
    """Debugging mode states."""

    INITIALIZING = auto()  # Startup phase
    MUTABLE = auto()    # Live mode - full access
    IMMUTABLE = auto()  # Coredump mode - read only
    RUNNING = auto()     # Live mode substate - target executing
    STOPPED = auto()    # Live mode substate - target paused
    CRASHED = auto()    # cuda-gdb process crashed


class OperationCategory(Enum):
    """Categories of operations with different permission requirements."""

    READ_ONLY = auto()      # Always allowed
    READ_WRITE = auto()     # Requires MUTABLE or STOPPED
    EXECUTION = auto()     # Requires STOPPED (can't modify while running)


# Method to operation category mapping
METHOD_PERMISSIONS: Dict[str, OperationCategory] = {
    # Read-only methods (allowed in all modes including IMMUTABLE/coredump)
    # CPU handlers
    "backtrace": OperationCategory.READ_ONLY,
    "threads": OperationCategory.READ_ONLY,
    "stack": OperationCategory.READ_ONLY,
    "registers": OperationCategory.READ_ONLY,
    "breakpoints": OperationCategory.READ_ONLY,
    "locals": OperationCategory.READ_ONLY,
    "info_args": OperationCategory.READ_ONLY,
    "info_frame": OperationCategory.READ_ONLY,
    "evaluate": OperationCategory.READ_ONLY,
    "memory": OperationCategory.READ_ONLY,
    "disassemble": OperationCategory.READ_ONLY,
    # CUDA handlers (read-only)
    "cuda_threads": OperationCategory.READ_ONLY,
    "cuda_kernels": OperationCategory.READ_ONLY,
    "cuda_devices": OperationCategory.READ_ONLY,
    "cuda_exceptions": OperationCategory.READ_ONLY,
    "cuda_warps": OperationCategory.READ_ONLY,
    "cuda_lanes": OperationCategory.READ_ONLY,
    "cuda_memory": OperationCategory.READ_ONLY,
    # CUDA handlers (read-write)
    "cuda_focus": OperationCategory.READ_WRITE,
}


class ModalityGuard:
    """
    Finite State Machine for debugging mode and permission checking.

    Singleton - shared across all handlers.
    Enforces modality-aware operation permissions.
    """

    _instance: Optional["ModalityGuard"] = None

    def __init__(self):
        """Initialize modality guard."""
        self.current_mode = DebugModality.INITIALIZING
        self.last_focus_snapshot: Optional[Dict[str, Any]] = None
        self._target_info: Optional[Dict[str, Any]] = None
        self._callbacks: List[Callable[[DebugModality, DebugModality], None]] = []

    @classmethod
    def get_instance(cls) -> "ModalityGuard":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = ModalityGuard()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def detect_modality(self) -> Dict[str, Any]:
        """
        Detect Live vs Coredump mode from target info.

        Returns:
            Dict with mode, target_type, and capabilities
        """
        try:
            # Get target information from GDB
            inferior = gdb.selected_inferior()
            pid = inferior.pid
            was_attached = inferior.was_attached()

            # Try to get executable
            exe = inferior.executable_path if hasattr(inferior, 'executable_path') else None

            # Check if we have a core file
            try:
                core_file = gdb.execute("info program", to_string=True)
                is_core = "core file" in core_file.lower()
            except:
                is_core = False

            if is_core:
                self.current_mode = DebugModality.IMMUTABLE
                target_type = "coredump"
            elif was_attached:
                self.current_mode = DebugModality.MUTABLE
                target_type = "attach"
            else:
                self.current_mode = DebugModality.MUTABLE
                target_type = "load"

            self._target_info = {
                "mode": self.current_mode.name,
                "target_type": target_type,
                "pid": pid,
                "executable": exe,
                "capabilities": self._get_capabilities(),
            }

            return self._target_info

        except Exception as e:
            # Fallback to INITIALIZING if detection fails
            self.current_mode = DebugModality.INITIALIZING
            return {
                "mode": "INITIALIZING",
                "error": str(e),
            }

    def _get_capabilities(self) -> Dict[str, bool]:
        """Get capabilities based on current mode."""
        if self.current_mode in (DebugModality.IMMUTABLE, DebugModality.CRASHED):
            return {
                "can_read_memory": True,
                "can_write_memory": False,
                "can_execute": False,
                "can_set_breakpoints": False,
                "can_modify_registers": False,
            }
        else:
            return {
                "can_read_memory": True,
                "can_write_memory": True,
                "can_execute": True,
                "can_set_breakpoints": True,
                "can_modify_registers": True,
            }

    def check_permission(self, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Check if method is allowed in current mode.

        Args:
            method_name: The RPC method name

        Returns:
            None if allowed, error dict if forbidden
        """
        from ..errors import error_modality_forbidden

        # Get operation category for method
        category = METHOD_PERMISSIONS.get(method_name, OperationCategory.READ_ONLY)

        # Check based on current mode and required category
        if category == OperationCategory.READ_ONLY:
            return None

        if self.current_mode == DebugModality.INITIALIZING:
            return error_modality_forbidden(
                method_name,
                "debugger is still initializing"
            )

        if self.current_mode == DebugModality.CRASHED:
            return error_modality_forbidden(
                method_name,
                "debugger process has crashed"
            )

        if self.current_mode == DebugModality.IMMUTABLE:
            return error_modality_forbidden(
                method_name,
                "coredump mode is read-only"
            )

        if self.current_mode == DebugModality.RUNNING:
            if category == OperationCategory.EXECUTION:
                return error_modality_forbidden(
                    method_name,
                    "target is currently running"
                )
            # READ_WRITE is allowed while running (can switch focus, etc.)
            return None

        if self.current_mode in (DebugModality.MUTABLE, DebugModality.STOPPED):
            return None

        # Default: allow
        return None

    def on_target_stopped(self) -> None:
        """Transition to STOPPED when target stops."""
        if self.current_mode == DebugModality.RUNNING:
            self.current_mode = DebugModality.STOPPED
            self._notify_callbacks(DebugModality.RUNNING, DebugModality.STOPPED)

    def on_target_running(self) -> None:
        """Transition to RUNNING when target continues."""
        if self.current_mode == DebugModality.STOPPED:
            self.current_mode = DebugModality.RUNNING
            self._notify_callbacks(DebugModality.STOPPED, DebugModality.RUNNING)

    def on_target_exited(self) -> None:
        """Transition to STOPPED when target exits."""
        if self.current_mode in (DebugModality.RUNNING, DebugModality.STOPPED):
            self.current_mode = DebugModality.STOPPED

    def on_crash(self) -> None:
        """Mark debugger as crashed."""
        self.current_mode = DebugModality.CRASHED
        self._notify_callbacks(self.current_mode, DebugModality.CRASHED)

    def add_mode_callback(
        self,
        callback: Callable[[DebugModality, DebugModality], None]
    ) -> None:
        """Register a callback for mode changes."""
        self._callbacks.append(callback)

    def _notify_callbacks(
        self,
        old_mode: DebugModality,
        new_mode: DebugModality,
    ) -> None:
        """Notify all registered callbacks of mode change."""
        for callback in self._callbacks:
            try:
                callback(old_mode, new_mode)
            except Exception:
                pass

    def get_current_mode(self) -> DebugModality:
        """Get current mode."""
        return self.current_mode

    def is_read_only_mode(self) -> bool:
        """Check if current mode is read-only (coredump)."""
        return self.current_mode == DebugModality.IMMUTABLE

    def is_running_mode(self) -> bool:
        """Check if target is running."""
        return self.current_mode == DebugModality.RUNNING


# Convenience function
def get_modality_guard() -> ModalityGuard:
    """Get the modality guard singleton."""
    return ModalityGuard.get_instance()