"""
GDB Value Serializer - Convert gdb.Value objects to JSON-compatible structures.

Handles all CUDA types including pointers, arrays, structs, and optimized-out values.
"""

import gdb


class GdbValueSerializer:
    """Serialize gdb.Value objects to JSON-compatible Python objects."""

    MAX_ARRAY_ELEMENTS = 256  # Prevent large arrays from bloating agent context
    MAX_STRING_LENGTH = 4096  # String truncation threshold
    MAX_DEPTH = 5  # Maximum recursion depth for nested structures

    @staticmethod
    def serialize(gdb_value, depth: int = 0, max_depth: int = None) -> dict:
        """
        Recursively serialize gdb.Value to JSON-compatible structure.

        Returns:
            {
                "value": <actual value>,
                "type": <type string>,
                "address": <memory address, if available>,
                "meta": {<metadata like optimized_out marker>}
            }
        """
        if max_depth is None:
            max_depth = GdbValueSerializer.MAX_DEPTH

        if depth > max_depth:
            return {"value": "<max_depth_exceeded>", "type": "truncated"}

        result = {"type": str(gdb_value.type)}

        # Check if optimized out
        if gdb_value.is_optimized_out:
            result["value"] = None
            result["meta"] = {
                "optimized_out": True,
                "hint": "Variable was optimized out by the compiler. "
                        "Recompile with -g -G flags to preserve debug info."
            }
            return result

        # Get address if available
        try:
            if gdb_value.address is not None:
                result["address"] = hex(int(gdb_value.address))
        except gdb.error:
            pass

        type_code = gdb_value.type.strip_typedefs().code

        # Basic integer types
        if type_code in (gdb.TYPE_CODE_INT, gdb.TYPE_CODE_ENUM,
                         gdb.TYPE_CODE_CHAR, gdb.TYPE_CODE_BOOL):
            try:
                int_val = int(gdb_value)
                result["value"] = int_val
                result["hex"] = hex(int_val)
            except (gdb.error, OverflowError):
                result["value"] = str(gdb_value)
            return result

        # Float types
        if type_code == gdb.TYPE_CODE_FLT:
            try:
                result["value"] = float(gdb_value)
            except (gdb.error, ValueError):
                result["value"] = str(gdb_value)
            return result

        # Pointer types
        if type_code == gdb.TYPE_CODE_PTR:
            try:
                ptr_val = int(gdb_value)
                result["value"] = hex(ptr_val)
                if ptr_val == 0:
                    result["meta"] = {"null_pointer": True}
            except (gdb.error, OverflowError):
                result["value"] = str(gdb_value)
            return result

        # Array types
        if type_code == gdb.TYPE_CODE_ARRAY:
            try:
                array_type = gdb_value.type.strip_typedefs()
                range_type = array_type.range()
                length = range_type[1] - range_type[0] + 1
                actual_length = min(length, GdbValueSerializer.MAX_ARRAY_ELEMENTS)

                elements = []
                for i in range(actual_length):
                    elem = GdbValueSerializer.serialize(
                        gdb_value[i], depth + 1, max_depth
                    )
                    elements.append(elem)

                result["value"] = elements
                result["meta"] = {
                    "total_length": length,
                    "displayed_length": actual_length,
                    "truncated": length > actual_length
                }
            except gdb.error as err:
                result["value"] = None
                result["meta"] = {"read_error": str(err)}
            return result

        # Struct/Union types
        if type_code in (gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION):
            try:
                fields = {}
                for field in gdb_value.type.fields():
                    try:
                        field_val = gdb_value[field.name]
                        fields[field.name] = GdbValueSerializer.serialize(
                            field_val, depth + 1, max_depth
                        )
                    except gdb.error as err:
                        fields[field.name] = {
                            "value": None,
                            "meta": {"read_error": str(err)}
                        }
                result["value"] = fields
            except gdb.error as err:
                result["value"] = None
                result["meta"] = {"read_error": str(err)}
            return result

        # Fallback: Use GDB's string representation
        try:
            str_val = str(gdb_value)
            if len(str_val) > GdbValueSerializer.MAX_STRING_LENGTH:
                str_val = str_val[:GdbValueSerializer.MAX_STRING_LENGTH]
                result["meta"] = {"string_truncated": True}
            result["value"] = str_val
        except gdb.error:
            result["value"] = "<unreadable>"

        return result