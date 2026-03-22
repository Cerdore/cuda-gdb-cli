"""GPU thread focus tracker for CUDA debugging."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class SoftwareCoords:
    """Software coordinates for GPU threads."""
    block: Tuple[int, int, int] = (0, 0, 0)
    thread: Tuple[int, int, int] = (0, 0, 0)
    kernel: Optional[int] = None


@dataclass
class HardwareCoords:
    """Hardware coordinates for GPU execution units."""
    device: int = 0
    sm: int = 0
    warp: int = 0
    lane: int = 0


class FocusTracker:
    """Tracks current GPU thread focus coordinates.

    Maintains both software coordinates (kernel, block, thread)
    and hardware coordinates (device, SM, warp, lane).
    """

    def __init__(self):
        """Initialize focus tracker."""
        self.software_coords = SoftwareCoords()
        self.hardware_coords = HardwareCoords()
        self._focus_history: List[SoftwareCoords] = []

    def update(
        self,
        block: Optional[List[int]] = None,
        thread: Optional[List[int]] = None,
        kernel: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update focus coordinates.

        Args:
            block: Block coordinates [x, y, z]
            thread: Thread coordinates [x, y, z]
            kernel: Kernel ID

        Returns:
            Dict with software and hardware coordinates
        """
        # Save previous focus to history
        self._focus_history.append(SoftwareCoords(
            block=self.software_coords.block,
            thread=self.software_coords.thread,
            kernel=self.software_coords.kernel
        ))

        # Update software coordinates
        if block is not None and len(block) == 3:
            self.software_coords.block = tuple(block)
        if thread is not None and len(thread) == 3:
            self.software_coords.thread = tuple(thread)
        if kernel is not None:
            self.software_coords.kernel = kernel

        # Hardware coordinates would be queried from GDB
        # This is a placeholder - actual implementation queries cuda-gdb
        return self.get_snapshot()

    def get_snapshot(self) -> Dict[str, Any]:
        """Return current focus for notifications/responses.

        Returns:
            Dict with software_coords and hardware_coords
        """
        return {
            "software_coords": {
                "block": list(self.software_coords.block),
                "thread": list(self.software_coords.thread),
                "kernel": self.software_coords.kernel,
            },
            "hardware_coords": {
                "device": self.hardware_coords.device,
                "sm": self.hardware_coords.sm,
                "warp": self.hardware_coords.warp,
                "lane": self.hardware_coords.lane,
            }
        }

    def update_hardware_coords(
        self,
        device: int,
        sm: int,
        warp: int,
        lane: int
    ) -> None:
        """Update hardware coordinates after GDB query."""
        self.hardware_coords.device = device
        self.hardware_coords.sm = sm
        self.hardware_coords.warp = warp
        self.hardware_coords.lane = lane

    def get_block_str(self) -> str:
        """Get block coordinates as string for GDB commands."""
        return f"({self.software_coords.block[0]},{self.software_coords.block[1]},{self.software_coords.block[2]})"

    def get_thread_str(self) -> str:
        """Get thread coordinates as string for GDB commands."""
        return f"({self.software_coords.thread[0]},{self.software_coords.thread[1]},{self.software_coords.thread[2]})"

    def get_previous_focus(self) -> Optional[SoftwareCoords]:
        """Get the previous focus from history."""
        if self._focus_history:
            return self._focus_history[-1]
        return None

    def clear_history(self) -> None:
        """Clear focus history."""
        self._focus_history.clear()


# Singleton instance
_focus_tracker: Optional[FocusTracker] = None


def get_focus_tracker() -> FocusTracker:
    """Get the focus tracker singleton."""
    global _focus_tracker
    if _focus_tracker is None:
        _focus_tracker = FocusTracker()
    return _focus_tracker
