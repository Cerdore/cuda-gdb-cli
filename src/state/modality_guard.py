"""
Modality Guard - Finite State Machine for debugging mode detection and permission control.

This module implements the FSM that detects and enforces the current debugging mode
(Live vs Coredump) and controls which operations are permitted in each state.
"""

from enum import Enum, auto
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass


class DebugModality(Enum):
    """Debugging modality states for the FSM."""
    INITIALIZING = auto()  # Startup, mode not yet detected
    MUTABLE = auto()       # Live mode - full read/write access
    IMMUTABLE = auto()     # Coredump mode - read only
    RUNNING = auto()       # Live mode substate - target is executing
    STOPPED = auto()       # Live mode substate - target is paused
    CRASHED = auto()       # cuda-gdb itself crashed


@dataclass
class ModalityInfo:
    """Information about the current modality."""
    mode: DebugModality
    description: str
    capabilities: Dict[str, bool]


class ModalityGuard:
    """
    Finite State Machine for debugging mode and permission checking.

    This is a singleton class that manages the debugging mode state
    and enforces which operations are permitted in each mode.

    State Transitions:
        INITIALIZING → MUTABLE (Live mode detected)
        INITIALIZING → IMMUTABLE (Coredump mode detected)
        MUTABLE → RUNNING (continue/step executed)
        RUNNING → STOPPED (breakpoint/exception hit)
        STOPPED → RUNNING (continue/step executed)
        * → CRASHED (cuda-gdb crashed)

    Permission Rules:
        - IMMUTABLE mode: No execution control, no memory modification
        - RUNNING state: No read operations (target is executing)
        - STOPPED/MUTABLE: All operations permitted
    """

    # Methods that require MUTABLE mode (Live debugging)
    EXECUTION_METHODS: Set[str] = {
        "cuda_execution_control",
        "cuda_set_breakpoint",
        "cuda_remove_breakpoint",
    }

    # Methods forbidden when target is RUNNING
    READ_METHODS: Set[str] = {
        "cuda_set_focus",
        "cuda_evaluate_var",
        "cuda_dump_warp_registers",
        "cuda_analyze_exception",
        "cuda_read_shared_memory",
        "cuda_list_kernels",
        "cuda_backtrace",
        "cuda_disassemble",
        "cuda_device_info",
    }

    _instance: Optional['ModalityGuard'] = None

    def __new__(cls) -> 'ModalityGuard':
        """Singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the modality guard."""
        if self._initialized:
            return

        self.current_mode: DebugModality = DebugModality.INITIALIZING
        self.last_focus_snapshot: Optional[Dict[str, Any]] = None
        self._initialized = True

    def detect_modality(self) -> ModalityInfo:
        """
        Detect the current debugging modality from GDB target info.

        This method inspects the GDB target to determine if we are
        in Live mode (attached to a running process or about to launch)
        or Coredump mode (analyzing a post-mortem dump file).

        Returns:
            ModalityInfo with mode, description, and capabilities.
        """
        try:
            # Import gdb here to allow testing without gdb module
            import gdb

            # Check target information
            target_info = gdb.execute("info target", to_string=True)

            if "cudacore" in target_info.lower() or ".cudacore" in target_info:
                self.current_mode = DebugModality.IMMUTABLE
                return ModalityInfo(
                    mode=DebugModality.IMMUTABLE,
                    description="Post-mortem coredump analysis mode. "
                               "Execution control is disabled. All data is read-only.",
                    capabilities={
                        "read_variables": True,
                        "read_registers": True,
                        "read_memory": True,
                        "set_focus": True,
                        "execution_control": False,
                        "modify_memory": False,
                        "set_breakpoints": False,
                    }
                )
            else:
                self.current_mode = DebugModality.STOPPED
                return ModalityInfo(
                    mode=DebugModality.MUTABLE,
                    description="Live interactive debugging mode. "
                               "Full execution control available.",
                    capabilities={
                        "read_variables": True,
                        "read_registers": True,
                        "read_memory": True,
                        "set_focus": True,
                        "execution_control": True,
                        "modify_memory": True,
                        "set_breakpoints": True,
                    }
                )

        except Exception as e:
            # If we can't detect, stay in INITIALIZING
            return ModalityInfo(
                mode=DebugModality.INITIALIZING,
                description=f"Unable to detect modality: {str(e)}",
                capabilities={}
            )

    def check_permission(self, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Check if the given method is permitted in the current mode.

        Args:
            method_name: The JSON-RPC method name to check.

        Returns:
            None if the method is permitted.
            Error dict if the method is forbidden in current mode.
        """
        # Coredump mode (IMMUTABLE) - block execution and modification
        if self.current_mode == DebugModality.IMMUTABLE:
            if method_name in self.EXECUTION_METHODS:
                return {
                    "code": -32003,
                    "message": f"Method '{method_name}' is forbidden in "
                               f"Coredump (IMMUTABLE) mode",
                    "data": {
                        "current_mode": "IMMUTABLE",
                        "hint": "This is a post-mortem coredump analysis session. "
                                "Execution control and memory modification are "
                                "not available. Only read-only inspection tools "
                                "can be used.",
                        "available_methods": sorted(self.READ_METHODS)
                    }
                }

        # Target is RUNNING - block read operations
        if self.current_mode == DebugModality.RUNNING:
            if method_name in self.READ_METHODS:
                return {
                    "code": -32004,
                    "message": f"Method '{method_name}' requires the target "
                               f"to be stopped",
                    "data": {
                        "current_mode": "RUNNING",
                        "hint": "The target is currently running. Send "
                                "cuda_execution_control with action='interrupt' "
                                "to pause execution first."
                    }
                }

        # CRASHED state - block everything
        if self.current_mode == DebugModality.CRASHED:
            return {
                "code": -32002,
                "message": "cuda-gdb process crashed",
                "data": {
                    "current_mode": "CRASHED",
                    "hint": "The debugger has crashed. The session needs to be "
                            "restarted."
                }
            }

        # All checks passed - method is permitted
        return None

    def on_target_stopped(self) -> None:
        """
        Transition to STOPPED state when target stops.

        Called when the target hits a breakpoint, exception, or is
        interrupted. Also captures the current focus snapshot.
        """
        if self.current_mode == DebugModality.RUNNING:
            self.current_mode = DebugModality.STOPPED

        # Capture focus snapshot for notifications
        self._capture_focus_snapshot()

    def on_target_running(self) -> None:
        """
        Transition to RUNNING state when target starts executing.

        Called when continue/step commands are issued.
        """
        if self.current_mode in (DebugModality.STOPPED, DebugModality.MUTABLE):
            self.current_mode = DebugModality.RUNNING

    def on_crashed(self) -> None:
        """
        Transition to CRASHED state when cuda-gdb crashes.
        """
        self.current_mode = DebugModality.CRASHED

    def _capture_focus_snapshot(self) -> None:
        """Capture the current GPU thread focus coordinates."""
        try:
            import gdb
            output = gdb.execute("cuda kernel block thread", to_string=True)
            self.last_focus_snapshot = {
                "raw_output": output.strip()
            }
        except Exception:
            self.last_focus_snapshot = None

    def get_mode_info(self) -> Dict[str, Any]:
        """
        Get information about the current mode.

        Returns:
            Dict with mode name and state information.
        """
        return {
            "mode": self.current_mode.name,
            "is_live": self.current_mode in (
                DebugModality.MUTABLE,
                DebugModality.RUNNING,
                DebugModality.STOPPED
            ),
            "is_coredump": self.current_mode == DebugModality.IMMUTABLE,
            "can_execute": self.current_mode in (
                DebugModality.MUTABLE,
                DebugModality.STOPPED
            ),
        }


# Singleton instance for convenience
def get_modality_guard() -> ModalityGuard:
    """Get the singleton ModalityGuard instance."""
    return ModalityGuard()