"""
CUDA Focus Manager - GPU thread focus management with coordinate validation.

Implements cuda_set_focus tool handler per API schema section 2.1.
"""

import re
import gdb


class CudaFocusManager:
    """
    GPU thread focus manager.
    Encapsulates cuda thread/block/kernel commands with coordinate validation and hardware mapping.
    """

    @staticmethod
    def set_focus(params: dict) -> dict:
        """
        Switch GPU thread focus.

        Args:
            params: {
                "block": [x, y, z],    # Thread Block coordinates
                "thread": [x, y, z],   # Thread coordinates within block
                "kernel": int          # Optional, kernel number
            }

        Returns:
            {
                "status": "ok",
                "software_coords": {...},
                "hardware_mapping": {"device": 0, "sm": 7, "warp": 3, "lane": 15},
                "verification": {...}
            }
        """
        block = params.get("block", [0, 0, 0])
        thread = params.get("thread", [0, 0, 0])
        kernel = params.get("kernel")

        # Validate coordinates
        if not isinstance(block, list) or len(block) != 3:
            return {
                "status": "error",
                "error_type": "invalid_coordinates",
                "message": "block must be a list of 3 integers [x, y, z]"
            }

        if not isinstance(thread, list) or len(thread) != 3:
            return {
                "status": "error",
                "error_type": "invalid_coordinates",
                "message": "thread must be a list of 3 integers [x, y, z]"
            }

        for coord in block + thread:
            if not isinstance(coord, int) or coord < 0:
                return {
                    "status": "error",
                    "error_type": "invalid_coordinates",
                    "message": "Coordinates must be non-negative integers"
                }

        # Build cuda command
        commands = []
        if kernel is not None:
            commands.append(f"cuda kernel {kernel}")
        commands.append(
            f"cuda block {block[0]},{block[1]},{block[2]} "
            f"thread {thread[0]},{thread[1]},{thread[2]}"
        )

        try:
            for cmd in commands:
                gdb.execute(cmd, to_string=True)

            # Get hardware mapping
            hardware_mapping = CudaFocusManager._get_hardware_mapping()

            # Verify focus was switched
            verification = CudaFocusManager._verify_focus(block, thread)

            return {
                "status": "ok",
                "software_coords": {
                    "block": block,
                    "thread": thread,
                    "kernel": kernel
                },
                "hardware_mapping": hardware_mapping,
                "verification": verification
            }

        except gdb.error as err:
            error_msg = str(err)

            # Detect common focus switch failure reasons
            if "not within" in error_msg or "invalid" in error_msg.lower():
                return {
                    "status": "error",
                    "error_type": "invalid_coordinates",
                    "message": error_msg,
                    "hint": "The specified block/thread coordinates are outside "
                            "the active grid dimensions. Use cuda_list_kernels "
                            "to check the current grid configuration."
                }
            if "no active" in error_msg.lower() or "no kernel" in error_msg.lower():
                return {
                    "status": "error",
                    "error_type": "no_active_kernel",
                    "message": error_msg,
                    "hint": "No CUDA kernel is currently active on the GPU. "
                            "The program may be executing host (CPU) code. "
                            "Set a breakpoint inside a __global__ function first."
                }
            return {
                "status": "error",
                "error_type": "focus_switch_failed",
                "message": error_msg
            }

    @staticmethod
    def _get_hardware_mapping() -> dict:
        """Get current focus hardware coordinate mapping."""
        mapping = {}
        try:
            output = gdb.execute("cuda device sm warp lane", to_string=True)
            for key in ["device", "sm", "warp", "lane"]:
                match = re.search(rf'{key}\s+(\d+)', output, re.IGNORECASE)
                if match:
                    mapping[key] = int(match.group(1))
        except gdb.error:
            pass
        return mapping

    @staticmethod
    def _verify_focus(expected_block: list, expected_thread: list) -> dict:
        """Verify focus was successfully switched to expected coordinates."""
        try:
            thread_x = int(gdb.parse_and_eval("threadIdx.x"))
            thread_y = int(gdb.parse_and_eval("threadIdx.y"))
            thread_z = int(gdb.parse_and_eval("threadIdx.z"))
            block_x = int(gdb.parse_and_eval("blockIdx.x"))
            block_y = int(gdb.parse_and_eval("blockIdx.y"))
            block_z = int(gdb.parse_and_eval("blockIdx.z"))

            actual_thread = [thread_x, thread_y, thread_z]
            actual_block = [block_x, block_y, block_z]

            return {
                "verified": (actual_thread == expected_thread
                             and actual_block == expected_block),
                "actual_thread": actual_thread,
                "actual_block": actual_block
            }
        except gdb.error:
            return {"verified": False, "reason": "Unable to read threadIdx/blockIdx"}


# Tool handler registration
TOOL_HANDLER = CudaFocusManager.set_focus