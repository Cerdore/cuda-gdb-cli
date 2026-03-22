"""
Shared Memory Accessor - GPU shared memory reading with @shared modifier.

Implements cuda_read_shared_memory tool handler per API schema section 2.6.
"""

import gdb

from .serializer import GdbValueSerializer


class SharedMemoryAccessor:
    """
    Shared memory safe accessor.
    Automatically injects @shared address space modifier and handles IPC memory access denial exceptions.
    """

    # CUDA IPC memory access denial characteristic error messages
    IPC_REJECTION_PATTERNS = [
        "Cannot access memory imported via CUDA IPC",
        "IPC memory access denied",
    ]

    @staticmethod
    def read_by_variable(variable_name: str, array_length: int = None) -> dict:
        """
        Read shared memory by variable name.

        Args:
            variable_name: Shared memory variable name (e.g., "s_data")
            array_length: If array, specify read length

        Returns:
            Serialized JSON-compatible structure
        """
        try:
            if array_length and array_length > 0:
                expression = f"({variable_name})@{array_length}"
            else:
                expression = variable_name

            gdb_value = gdb.parse_and_eval(expression)
            return {
                "status": "ok",
                "memory_space": "shared",
                "data": GdbValueSerializer.serialize(gdb_value)
            }
        except gdb.error as err:
            return SharedMemoryAccessor._handle_error(err)

    @staticmethod
    def read_by_address(address: str, data_type: str, count: int = 1) -> dict:
        """
        Read shared memory by physical address.
        Automatically injects @shared modifier.

        Args:
            address: Hex address string (e.g., "0x20") or integer
            data_type: C type string (e.g., "int", "float", "double")
            count: Number of elements to read

        Returns:
            Serialized JSON-compatible structure
        """
        if isinstance(address, int):
            address = hex(address)

        try:
            if count > 1:
                # Read contiguous array: *((@shared TYPE*)ADDR)@COUNT
                expression = f"*((@shared {data_type}*){address})@{count}"
            else:
                # Read single value: *(@shared TYPE*)ADDR
                expression = f"*(@shared {data_type}*){address}"

            gdb_value = gdb.parse_and_eval(expression)
            return {
                "status": "ok",
                "memory_space": "shared",
                "address": address,
                "data_type": data_type,
                "count": count,
                "data": GdbValueSerializer.serialize(gdb_value)
            }
        except gdb.error as err:
            return SharedMemoryAccessor._handle_error(err)

    @staticmethod
    def _handle_error(err: Exception) -> dict:
        """Unified shared memory error handling."""
        error_msg = str(err)

        # Detect IPC memory access denial
        for pattern in SharedMemoryAccessor.IPC_REJECTION_PATTERNS:
            if pattern in error_msg:
                return {
                    "status": "error",
                    "error_type": "ipc_access_denied",
                    "message": error_msg,
                    "hint": "This shared memory allocation was imported via "
                            "CUDA IPC (inter-process communication). cuda-gdb "
                            "explicitly prohibits accessing IPC-imported memory "
                            "for security reasons."
                }

        # Detect address out of bounds
        if "Address out of bounds" in error_msg or "out of bounds" in error_msg:
            return {
                "status": "error",
                "error_type": "address_out_of_bounds",
                "message": error_msg,
                "hint": "The address may be outside the shared memory "
                        "allocation for the current thread block. Verify "
                        "the block dimensions and shared memory size."
            }

        # Generic error
        return {
            "status": "error",
            "error_type": "shared_memory_read_failed",
            "message": error_msg
        }


# Tool handler registration
def handle_read_shared_memory(params: dict) -> dict:
    """
    Main entry point for cuda_read_shared_memory tool.
    Supports both address and variable name access.
    """
    address = params.get("address")
    if not address:
        return {
            "status": "error",
            "error_type": "invalid_params",
            "message": "address parameter is required"
        }

    data_type = params.get("data_type", "int")
    count = params.get("count", 1)

    # Check if address is a variable name (starts with letter) or hex address
    if isinstance(address, str) and address.startswith("0x"):
        return SharedMemoryAccessor.read_by_address(address, data_type, count)
    else:
        # Treat as variable name
        return SharedMemoryAccessor.read_by_variable(address, count)


TOOL_HANDLER = handle_read_shared_memory