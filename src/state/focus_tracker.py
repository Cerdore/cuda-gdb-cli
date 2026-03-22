"""Focus Tracker - GPU thread focus state."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import gdb


@dataclass
class SoftwareCoords:
    """Software coordinates for GPU thread."""

    block: Tuple[int, int, int] = (0, 0, 0)
    thread: Tuple[int, int, int] = (0, 0, 0)
    kernel: Optional[int] = None


@dataclass
class HardwareCoords:
    """Hardware coordinates for GPU thread."""

    device: int = 0
    sm: int = 0
    warp: int = 0
    lane: int = 0


class FocusTracker:
    """
    Tracks current GPU thread focus coordinates.

    Maintains both software (block/thread/kernel) and
    hardware (device/sm/warp/lane) coordinate systems.
    """

    def __init__(self):
        """Initialize focus tracker."""
        self.software_coords = SoftwareCoords()
        self.hardware_coords = HardwareCoords()
        self._frame_info: Optional[Dict[str, Any]] = None

    def update(
        self,
        block: Optional[List[int]] = None,
        thread: Optional[List[int]] = None,
        kernel: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update focus and return hardware mapping.

        Args:
            block: New block coordinates [x, y, z]
            thread: New thread coordinates [x, y, z]
            kernel: New kernel id

        Returns:
            Dict with software coords, hardware mapping, and verification
        """
        # Update software coordinates
        if block is not None:
            if len(block) != 3:
                raise ValueError("block must have 3 coordinates")
            self.software_coords.block = tuple(block)

        if thread is not None:
            if len(thread) != 3:
                raise ValueError("thread must have 3 coordinates")
            self.software_coords.thread = tuple(thread)

        if kernel is not None:
            self.software_coords.kernel = kernel

        # Query hardware coordinates from GDB
        self._update_hardware_coords()

        # Get frame info
        self._update_frame_info()

        return self.get_snapshot()

    def _update_hardware_coords(self) -> None:
        """Query hardware coordinates from GDB."""
        try:
            # Get device
            output = gdb.execute("cuda device", to_string=True)
            self.hardware_coords.device = self._parse_single_int(output)
        except gdb.error:
            pass

        try:
            # Get SM
            output = gdb.execute("cuda sm", to_string=True)
            self.hardware_coords.sm = self._parse_single_int(output)
        except gdb.error:
            pass

        try:
            # Get warp
            output = gdb.execute("cuda warp", to_string=True)
            self.hardware_coords.warp = self._parse_single_int(output)
        except gdb.error:
            pass

        try:
            # Get lane
            output = gdb.execute("cuda lane", to_string=True)
            self.hardware_coords.lane = self._parse_single_int(output)
        except gdb.error:
            pass

    def _parse_single_int(self, output: str) -> int:
        """Parse a single integer from GDB output."""
        import re
        match = re.search(r'\d+', output)
        return int(match.group()) if match else 0

    def _update_frame_info(self) -> None:
        """Update current frame information."""
        try:
            frame = gdb.selected_frame()
            self._frame_info = {
                "function": frame.name() or "??",
                "address": hex(frame.pc()) if frame.pc() else None,
            }

            try:
                sal = frame.sal()
                if sal and sal.symtab:
                    self._frame_info["file"] = sal.symtab.filename
                    self._frame_info["line"] = sal.line
            except Exception:
                pass

        except Exception:
            self._frame_info = None

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Return current focus for notifications.

        Returns:
            Dict with software_coords, hardware_coords, and frame
        """
        result = {
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
            },
        }

        if self._frame_info:
            result["frame"] = self._frame_info

        return result

    def get_current_focus(self) -> Dict[str, Any]:
        """
        Get current focus from GDB (query-based).

        Returns:
            Current focus information from GDB
        """
        focus = {}

        # Query software coordinates
        for dim in ["kernel", "block", "thread"]:
            try:
                output = gdb.execute(f"cuda {dim}", to_string=True)
                focus[dim] = self._parse_focus_output(dim, output)
            except gdb.error:
                focus[dim] = None

        # Query hardware coordinates
        for dim in ["device", "sm", "warp", "lane"]:
            try:
                output = gdb.execute(f"cuda {dim}", to_string=True)
                focus[dim] = self._parse_single_int(output)
            except gdb.error:
                focus[dim] = None

        return focus

    def _parse_focus_output(self, dimension: str, output: str) -> Any:
        """Parse focus output for a dimension."""
        import re

        if dimension == "kernel":
            # Kernel is a single integer
            match = re.search(r'\d+', output)
            return int(match.group()) if match else None

        # Block and thread are coordinates like "(1,2,3)"
        match = re.search(r'\((\d+),(\d+),(\d+)\)', output)
        if match:
            return [int(m) for m in match.groups()]

        return None

    def verify_focus(
        self,
        block: List[int],
        thread: List[int],
    ) -> Dict[str, Any]:
        """
        Verify focus matches expected coordinates.

        Args:
            block: Expected block coordinates
            thread: Expected thread coordinates

        Returns:
            Verification result with actual coordinates
        """
        actual_focus = self.get_current_focus()

        actual_block = actual_focus.get("block")
        actual_thread = actual_focus.get("thread")

        verified = (
            actual_block == block and
            actual_thread == thread
        )

        return {
            "verified": verified,
            "expected_block": block,
            "actual_block": actual_block,
            "expected_thread": thread,
            "actual_thread": actual_thread,
        }


# Singleton instance
_instance: Optional[FocusTracker] = None


def get_focus_tracker() -> FocusTracker:
    """Get the focus tracker singleton."""
    global _instance
    if _instance is None:
        _instance = FocusTracker()
    return _instance


def reset_focus_tracker() -> None:
    """Reset focus tracker (for testing)."""
    global _instance
    _instance = None