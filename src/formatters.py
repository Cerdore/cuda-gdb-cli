"""JSON output formatters for CUDA-GDB-CLI."""

import json
import sys
from typing import Any, Dict


def print_json(data: Dict[str, Any]) -> None:
    """Print data as JSON to stdout."""
    print(json.dumps(data, indent=2))


def format_value(value: Any, max_depth: int = 5) -> Any:
    """Format a value for JSON output.

    Args:
        value: The value to format
        max_depth: Maximum recursion depth

    Returns:
        JSON-serializable value
    """
    if max_depth <= 0:
        return "..."

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.hex()

    if isinstance(value, dict):
        return {k: format_value(v, max_depth - 1) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [format_value(v, max_depth - 1) for v in value]

    return str(value)


def truncate_array(arr: list, max_items: int = 256) -> Dict[str, Any]:
    """Truncate an array for output.

    Args:
        arr: The array to truncate
        max_items: Maximum items to display

    Returns:
        Dict with elements and metadata
    """
    total = len(arr)
    truncated = total > max_items

    return {
        "elements": arr[:max_items],
        "total_count": total,
        "displayed_count": min(total, max_items),
        "truncated": truncated,
    }


def format_hex(value: int) -> str:
    """Format an integer as hex string."""
    return f"0x{value:x}"


def parse_hex(value: str) -> int:
    """Parse a hex string to integer."""
    if value.startswith("0x") or value.startswith("0X"):
        return int(value, 16)
    return int(value)
