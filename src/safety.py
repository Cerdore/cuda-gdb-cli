"""Safety module for command whitelisting and validation."""

from enum import Enum
from typing import Set, Optional


class SafetyLevel(Enum):
    """Safety levels for command execution."""
    READONLY = "readonly"
    READWRITE = "readwrite"
    FULL = "full"


# Commands that are always allowed
READONLY_COMMANDS: Set[str] = {
    # CPU commands
    "bt", "backtrace", "where",
    "info threads", "info registers", "info locals", "info args",
    "info frame", "info functions", "info variables",
    "list", "disassemble", "x/",
    "print", "p", "display",
    "ptype", "whatis",
    # CUDA commands
    "info cuda threads", "info cuda kernels", "info cuda devices",
    "info cuda exceptions", "info cuda warps", "info cuda lanes",
    "cuda kernel", "cuda block", "cuda thread", "cuda device",
}

# Commands that require readwrite level
READWRITE_COMMANDS: Set[str] = {
    "set", "set variable",
    "break", "b", "delete breakpoints", "disable", "enable",
    "watch", "rwatch", "awatch",
    "cuda set",
}

# Commands that require full level
FULL_COMMANDS: Set[str] = {
    "run", "start", "continue", "c", "fg",
    "step", "s", "next", "n", "finish", "return",
    "attach", "detach", "kill",
    "signal",
}

# Commands that are always blocked
BLOCKED_COMMANDS: Set[str] = {
    "quit", "q",
    "shell", "!",
    "pipe",
    "python", "pi",
}


def check_command(command: str, safety_level: SafetyLevel) -> Optional[str]:
    """Check if a command is allowed at the given safety level.

    Args:
        command: The GDB command to check
        safety_level: The current safety level

    Returns:
        None if allowed, error message if blocked
    """
    cmd = command.strip().lower()

    # Check blocked commands
    for blocked in BLOCKED_COMMANDS:
        if cmd.startswith(blocked):
            return f"Command '{command}' is blocked for safety"

    # Check readonly commands (always allowed)
    for readonly in READONLY_COMMANDS:
        if cmd.startswith(readonly):
            return None

    # Check readwrite commands
    for rw in READWRITE_COMMANDS:
        if cmd.startswith(rw):
            if safety_level in (SafetyLevel.READWRITE, SafetyLevel.FULL):
                return None
            return f"Command '{command}' requires readwrite or full safety level"

    # Check full commands
    for full in FULL_COMMANDS:
        if cmd.startswith(full):
            if safety_level == SafetyLevel.FULL:
                return None
            return f"Command '{command}' requires full safety level"

    # Default: allow if full level, otherwise warn
    if safety_level == SafetyLevel.FULL:
        return None

    return f"Command '{command}' not in whitelist for {safety_level.value} level"


# CUDA-specific command whitelist
CUDA_READONLY_COMMANDS: Set[str] = {
    "info cuda threads",
    "info cuda kernels",
    "info cuda devices",
    "info cuda exceptions",
    "info cuda warps",
    "info cuda lanes",
    "cuda kernel",
    "cuda block",
    "cuda thread",
    "cuda device",
}

CUDA_WRITE_COMMANDS: Set[str] = {
    "set cuda memcheck",
    "set cuda software_preemption",
    "set cuda break_on_launch",
}
