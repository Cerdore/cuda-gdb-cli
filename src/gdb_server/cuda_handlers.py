"""CUDA-specific GDB command handlers.

Parses cuda-gdb text output and returns structured JSON responses.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import gdb

from .focus_tracker import get_focus_tracker
from .value_formatter import serialize_gdb_value


# CUDA Exception type mappings
CUDA_EXCEPTION_MAP: Dict[str, str] = {
    "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS": "GPU thread accessed illegal memory",
    "CUDA_EXCEPTION_WARP_ASSERT": "GPU assert() triggered",
    "CUDA_EXCEPTION_LANE_INVALID_PC": "GPU thread executed invalid instruction",
    "CUDA_EXCEPTION_WARP_MISALIGNED": "GPU warp encountered misaligned access",
    "CUDA_EXCEPTION_DEVICE_LIMIT": "GPU device limit exceeded",
    "CUDA_EXCEPTION_LANE_OUT_OF_RANGE": "GPU lane index out of range",
    "CUDA_EXCEPTION_HARDWARE_STACK_OVERFLOW": "GPU hardware stack overflow",
    "CUDA_EXCEPTION_DEVICE_HARDWARE_ERROR": "GPU device hardware error",
    "CUDA_EXCEPTION_MISALIGNED_ADDRESS": "GPU misaligned memory address",
    "CUDA_EXCEPTION_INVALID_ADDRESS_SPACE": "GPU invalid address space access",
}

# Valid address spaces for CUDA memory
VALID_ADDRESS_SPACES = frozenset(["shared", "global", "local", "generic", "const", "param"])


def _parse_3d_coords(coord_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse 3D coordinates like (0,1,2) into tuple."""
    match = re.match(r"\((\d+),\s*(\d+),\s*(\d+)\)", coord_str.strip())
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _execute_cuda_command(cmd: str) -> Tuple[bool, str]:
    """Execute a cuda-gdb command and return (success, output)."""
    try:
        output = gdb.execute(cmd, to_string=True)
        return True, output or ""
    except gdb.error as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def _parse_table_output(output: str, header_pattern: str) -> Tuple[List[str], List[List[str]]]:
    """Parse tabular GDB output into headers and rows."""
    lines = output.strip().split("\n")
    headers = []
    rows = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Check for header line
        if re.search(header_pattern, line, re.IGNORECASE):
            headers = re.split(r"\s{2,}", line)
            continue

        # Skip separator lines
        if re.match(r"^[-=]+$", line):
            continue

        # Parse data row
        if headers and line:
            # Handle lines starting with * (current marker)
            row = re.split(r"\s{2,}", line.strip())
            rows.append(row)

    return headers, rows


# =============================================================================
# Handler 1: CUDA Threads
# =============================================================================

def handle_cuda_threads(**kwargs) -> Dict[str, Any]:
    """Parse `info cuda threads` output and return GPU thread list.

    Returns:
        Dict with cuda_threads list, total_count, and truncated flag.
    """
    success, output = _execute_cuda_command("info cuda threads")

    if not success:
        return {
            "cuda_threads": [],
            "total_count": 0,
            "truncated": False,
            "error": output
        }

    threads = []

    # Pattern matches:
    # * (0,0,0)  (0,0,0)    -   matmul_kernel  matmul.cu  28
    #   (0,0,0)  (1,0,0)    -   matmul_kernel  matmul.cu  28
    thread_pattern = re.compile(
        r"^(\*?\s*)"
        r"\((\d+),\s*(\d+),\s*(\d+)\)\s+"  # BlockIdx
        r"\((\d+),\s*(\d+),\s*(\d+)\)\s+"  # ThreadIdx
        r"(\S+)?\s+"  # To (often -)
        r"(\S+)\s+"  # Name
        r"(\S+)\s+"  # Filename
        r"(\d+)"  # Line
    )

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("BlockIdx") or re.match(r"^[-=]+$", line):
            continue

        match = thread_pattern.match(line)
        if match:
            is_current = match.group(1).strip() == "*"
            block_idx = [int(match.group(2)), int(match.group(3)), int(match.group(4))]
            thread_idx = [int(match.group(5)), int(match.group(6)), int(match.group(7))]
            name = match.group(9) or ""
            filename = match.group(10) or ""
            try:
                line_num = int(match.group(11))
            except (ValueError, TypeError):
                line_num = 0

            threads.append({
                "kernel": None,  # Will be filled from context
                "block_idx": block_idx,
                "thread_idx": thread_idx,
                "name": name,
                "file": filename,
                "line": line_num,
                "is_current": is_current,
                "exception": None
            })

    return {
        "cuda_threads": threads,
        "total_count": len(threads),
        "truncated": False
    }


# =============================================================================
# Handler 2: CUDA Kernels
# =============================================================================

def handle_cuda_kernels(**kwargs) -> Dict[str, Any]:
    """Parse `info cuda kernels` output.

    Returns:
        Dict with cuda_kernels list containing kernel info.
    """
    success, output = _execute_cuda_command("info cuda kernels")

    if not success:
        return {
            "cuda_kernels": [],
            "error": output
        }

    kernels = []

    # Pattern matches:
    # Kernel  Function       GridDim      BlockDim     Device  Status
    # 0       matmul_kernel  (32,32,1)    (16,16,1)    0       running
    kernel_pattern = re.compile(
        r"^(\d+)\s+"  # Kernel ID
        r"(\S+)\s+"  # Function name
        r"\((\d+),\s*(\d+),\s*(\d+)\)\s+"  # GridDim
        r"\((\d+),\s*(\d+),\s*(\d+)\)\s+"  # BlockDim
        r"(\d+)\s+"  # Device
        r"(\S+)"  # Status
    )

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("Kernel") or re.match(r"^[-=]+$", line):
            continue

        match = kernel_pattern.match(line)
        if match:
            kernel_id = int(match.group(1))
            function = match.group(2)
            grid_dim = [int(match.group(3)), int(match.group(4)), int(match.group(5))]
            block_dim = [int(match.group(6)), int(match.group(7)), int(match.group(8))]
            device = int(match.group(9))
            status = match.group(10)

            kernels.append({
                "id": kernel_id,
                "function": function,
                "grid_dim": grid_dim,
                "block_dim": block_dim,
                "device": device,
                "status": status
            })

    return {
        "cuda_kernels": kernels,
        "total_count": len(kernels)
    }


# =============================================================================
# Handler 3: CUDA Focus
# =============================================================================

def handle_cuda_focus(**kwargs) -> Dict[str, Any]:
    """View or switch GPU focus.

    Kwargs:
        kernel: Optional kernel ID to switch to
        block: Optional [x, y, z] block coords to switch to
        thread: Optional [x, y, z] thread coords to switch to
        device: Optional device ID
        sm: Optional SM ID
        warp: Optional warp ID
        lane: Optional lane ID

    Returns:
        Dict with current focus coordinates.
    """
    tracker = get_focus_tracker()
    changes_made = False

    # Switch kernel if specified
    kernel = kwargs.get("kernel")
    if kernel is not None:
        success, output = _execute_cuda_command(f"cuda kernel {kernel}")
        if success:
            tracker.update(kernel=kernel)
            changes_made = True
        else:
            return {
                **tracker.get_snapshot(),
                "error": f"Failed to switch kernel: {output}"
            }

    # Switch block if specified
    block = kwargs.get("block")
    if block is not None and len(block) == 3:
        block_str = f"({block[0]},{block[1]},{block[2]})"
        success, output = _execute_cuda_command(f"cuda block {block_str}")
        if success:
            tracker.update(block=block)
            changes_made = True
        else:
            return {
                **tracker.get_snapshot(),
                "error": f"Failed to switch block: {output}"
            }

    # Switch thread if specified
    thread = kwargs.get("thread")
    if thread is not None and len(thread) == 3:
        thread_str = f"({thread[0]},{thread[1]},{thread[2]})"
        success, output = _execute_cuda_command(f"cuda thread {thread_str}")
        if success:
            tracker.update(thread=thread)
            changes_made = True
        else:
            return {
                **tracker.get_snapshot(),
                "error": f"Failed to switch thread: {output}"
            }

    # Switch hardware coordinates if specified
    device = kwargs.get("device")
    sm = kwargs.get("sm")
    warp = kwargs.get("warp")
    lane = kwargs.get("lane")

    if device is not None:
        success, output = _execute_cuda_command(f"cuda device {device}")
        if success:
            tracker.hardware_coords.device = device
            changes_made = True

    if sm is not None:
        success, output = _execute_cuda_command(f"cuda sm {sm}")
        if success:
            tracker.hardware_coords.sm = sm
            changes_made = True

    if warp is not None:
        success, output = _execute_cuda_command(f"cuda warp {warp}")
        if success:
            tracker.hardware_coords.warp = warp
            changes_made = True

    if lane is not None:
        success, output = _execute_cuda_command(f"cuda lane {lane}")
        if success:
            tracker.hardware_coords.lane = lane
            changes_made = True

    # Get current focus from cuda-gdb if no changes were made
    if not changes_made:
        success, output = _execute_cuda_command("info cuda focus")
        if success:
            # Parse output like:
            # Current CUDA focus: kernel 0, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0
            focus_match = re.search(
                r"kernel\s+(\d+).*?"
                r"block\s+\((\d+),(\d+),(\d+)\).*?"
                r"thread\s+\((\d+),(\d+),(\d+)\)",
                output,
                re.IGNORECASE
            )
            if focus_match:
                tracker.update(
                    kernel=int(focus_match.group(1)),
                    block=[int(focus_match.group(2)), int(focus_match.group(3)), int(focus_match.group(4))],
                    thread=[int(focus_match.group(5)), int(focus_match.group(6)), int(focus_match.group(7))]
                )

    return tracker.get_snapshot()


# =============================================================================
# Handler 4: CUDA Devices
# =============================================================================

def handle_cuda_devices(**kwargs) -> Dict[str, Any]:
    """Parse `info cuda devices` output.

    Returns:
        Dict with cuda_devices list.
    """
    success, output = _execute_cuda_command("info cuda devices")

    if not success:
        return {
            "cuda_devices": [],
            "error": output
        }

    devices = []

    # Pattern matches:
    # Device  Name                    SMs  Cap  Threads/SM  Regs/SM  Mem
    # 0       NVIDIA A100-SXM4-40GB   108  8.0  2048        65536    40GB
    device_pattern = re.compile(
        r"^(\d+)\s+"  # Device ID
        r"(.+?)\s{2,}"  # Name (non-greedy, followed by spaces)
        r"(\d+)\s+"  # SMs
        r"(\d+\.\d+)\s+"  # Capability
        r"(\d+)\s+"  # Threads/SM
        r"(\d+)\s+"  # Regs/SM
        r"(\S+)"  # Memory
    )

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("Device") or re.match(r"^[-=]+$", line):
            continue

        match = device_pattern.match(line)
        if match:
            devices.append({
                "id": int(match.group(1)),
                "name": match.group(2).strip(),
                "sms": int(match.group(3)),
                "capability": match.group(4),
                "threads_per_sm": int(match.group(5)),
                "regs_per_sm": int(match.group(6)),
                "memory": match.group(7)
            })

    return {
        "cuda_devices": devices,
        "total_count": len(devices)
    }


# =============================================================================
# Handler 5: CUDA Exceptions
# =============================================================================

def handle_cuda_exceptions(**kwargs) -> Dict[str, Any]:
    """Parse `info cuda exceptions` output.

    Returns:
        Dict with cuda_exceptions list.
    """
    success, output = _execute_cuda_command("info cuda exceptions")

    if not success:
        return {
            "cuda_exceptions": [],
            "error": output
        }

    exceptions = []

    # Pattern matches:
    # Kernel  Block      Thread     Device  SM  Warp  Lane  Exception
    # 0       (0,0,0)    (0,0,0)    0       0   0     0     CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS
    exception_pattern = re.compile(
        r"^(\d+)\s+"  # Kernel
        r"\((\d+),(\d+),(\d+)\)\s+"  # Block
        r"\((\d+),(\d+),(\d+)\)\s+"  # Thread
        r"(\d+)\s+"  # Device
        r"(\d+)\s+"  # SM
        r"(\d+)\s+"  # Warp
        r"(\d+)\s+"  # Lane
        r"(\S+)"  # Exception type
    )

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("Kernel") or re.match(r"^[-=]+$", line):
            continue

        match = exception_pattern.match(line)
        if match:
            exception_type = match.group(12)
            description = CUDA_EXCEPTION_MAP.get(
                exception_type,
                f"Unknown CUDA exception: {exception_type}"
            )

            exceptions.append({
                "kernel": int(match.group(1)),
                "block": [int(match.group(2)), int(match.group(3)), int(match.group(4))],
                "thread": [int(match.group(5)), int(match.group(6)), int(match.group(7))],
                "device": int(match.group(8)),
                "sm": int(match.group(9)),
                "warp": int(match.group(10)),
                "lane": int(match.group(11)),
                "type": exception_type,
                "description": description
            })

    return {
        "cuda_exceptions": exceptions,
        "total_count": len(exceptions)
    }


# =============================================================================
# Handler 6: CUDA Memory
# =============================================================================

def handle_cuda_memory(**kwargs) -> Dict[str, Any]:
    """Read GPU address space memory.

    Kwargs:
        expr: Memory expression/variable name (required)
        space: Address space - shared/global/local/generic/const/param (default: generic)
        element_type: Type to cast elements as (default: auto)
        count: Number of elements to read (default: 1)
        format: Output format hint

    Returns:
        Dict with memory data, address, and metadata.
    """
    expr = kwargs.get("expr")
    if not expr:
        return {
            "error": "Missing required parameter: expr",
            "data": None
        }

    space = kwargs.get("space", "generic").lower()
    if space not in VALID_ADDRESS_SPACES:
        return {
            "error": f"Invalid address space: {space}. Valid: {', '.join(VALID_ADDRESS_SPACES)}",
            "data": None
        }

    element_type = kwargs.get("element_type")
    count = kwargs.get("count", 1)
    format_hint = kwargs.get("format")

    try:
        # Build GDB expression with address space modifier
        if element_type:
            gdb_expr = f"@{space} ({element_type}*){expr}"
            if count > 1:
                gdb_expr = f"@{space} ({element_type}[{count}]){expr}"
        else:
            gdb_expr = f"@{space} {expr}"

        # Evaluate the expression
        val = gdb.parse_and_eval(gdb_expr)

        # Serialize the value
        result = serialize_gdb_value(val)
        result["address_space"] = space

        if format_hint:
            result["format_hint"] = format_hint

        if count > 1:
            result["element_count"] = count

        return {
            "data": result,
            "expression": expr,
            "address_space": space
        }

    except gdb.error as e:
        error_msg = str(e)
        return {
            "error": error_msg,
            "error_type": "gdb_error",
            "data": None,
            "expression": expr,
            "address_space": space
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": "unknown",
            "data": None,
            "expression": expr,
            "address_space": space
        }


# =============================================================================
# Handler 7: CUDA Warps
# =============================================================================

def handle_cuda_warps(**kwargs) -> Dict[str, Any]:
    """Parse `info cuda warps` output.

    Returns:
        Dict with cuda_warps list.
    """
    success, output = _execute_cuda_command("info cuda warps")

    if not success:
        return {
            "cuda_warps": [],
            "error": output
        }

    warps = []

    # Pattern matches:
    # Warp  Device  SM    Active  Status
    # 0     0       0     32      active
    warp_pattern = re.compile(
        r"^(\d+)\s+"  # Warp ID
        r"(\d+)\s+"  # Device
        r"(\d+)\s+"  # SM
        r"(\d+)\s+"  # Active lanes
        r"(\S+)"  # Status
    )

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("Warp") or re.match(r"^[-=]+$", line):
            continue

        match = warp_pattern.match(line)
        if match:
            warps.append({
                "id": int(match.group(1)),
                "device": int(match.group(2)),
                "sm": int(match.group(3)),
                "active_lanes": int(match.group(4)),
                "status": match.group(5)
            })

    return {
        "cuda_warps": warps,
        "total_count": len(warps)
    }


# =============================================================================
# Handler 8: CUDA Lanes
# =============================================================================

def handle_cuda_lanes(**kwargs) -> Dict[str, Any]:
    """Parse `info cuda lanes` output.

    Returns:
        Dict with cuda_lanes list.
    """
    success, output = _execute_cuda_command("info cuda lanes")

    if not success:
        return {
            "cuda_lanes": [],
            "error": output
        }

    lanes = []

    # Pattern matches:
    # Lane  ThreadIdx   Active  Status
    # 0     (0,0,0)     yes     active
    # 1     (1,0,0)     yes     active
    lane_pattern = re.compile(
        r"^(\d+)\s+"  # Lane ID
        r"\((\d+),(\d+),(\d+)\)\s+"  # ThreadIdx
        r"(yes|no)\s+"  # Active
        r"(\S+)"  # Status
    )

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("Lane") or re.match(r"^[-=]+$", line):
            continue

        match = lane_pattern.match(line)
        if match:
            lanes.append({
                "id": int(match.group(1)),
                "thread_idx": [int(match.group(2)), int(match.group(3)), int(match.group(4))],
                "active": match.group(5).lower() == "yes",
                "status": match.group(6)
            })

    return {
        "cuda_lanes": lanes,
        "total_count": len(lanes)
    }


# =============================================================================
# Handler Registry
# =============================================================================

CUDA_HANDLERS: Dict[str, callable] = {
    "cuda_threads": handle_cuda_threads,
    "cuda_kernels": handle_cuda_kernels,
    "cuda_focus": handle_cuda_focus,
    "cuda_devices": handle_cuda_devices,
    "cuda_exceptions": handle_cuda_exceptions,
    "cuda_memory": handle_cuda_memory,
    "cuda_warps": handle_cuda_warps,
    "cuda_lanes": handle_cuda_lanes,
}


def get_cuda_handler(name: str) -> Optional[callable]:
    """Get a CUDA handler by name."""
    return CUDA_HANDLERS.get(name)


def list_cuda_handlers() -> List[str]:
    """List all available CUDA handlers."""
    return list(CUDA_HANDLERS.keys())