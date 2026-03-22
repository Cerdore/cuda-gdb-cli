"""gdb.Value to JSON serialization."""

from typing import Any, Optional, Dict, List
import gdb


# Configuration constants
MAX_ARRAY_ELEMENTS = 256
MAX_STRING_LENGTH = 4096
MAX_DEPTH = 5


class GdbValueSerializer:
    """Convert gdb.Value to JSON-compatible Python objects."""

    @staticmethod
    def serialize(gdb_value: gdb.Value, depth: int = 0) -> Dict[str, Any]:
        """
        Recursively serialize gdb.Value to JSON-compatible dict.

        Args:
            gdb_value: The gdb.Value to serialize
            depth: Current recursion depth (for nested structures)

        Returns:
            Dict with value, type, address, and metadata
        """
        if depth > MAX_DEPTH:
            return {
                "value": "<max depth exceeded>",
                "type": str(gdb_value.type) if gdb_value else "unknown",
                "meta": {"truncated": True, "reason": "max_depth_exceeded"},
            }

        try:
            # Handle None/invalid values
            if gdb_value is None:
                return {"value": None, "type": "none", "meta": {}}

            val_type = gdb_value.type

            # Check if optimized out
            try:
                # Try to stringize - this will fail for optimized out
                str(gdb_value)
            except RuntimeError as e:
                if "optimized out" in str(e).lower():
                    return {
                        "value": None,
                        "type": str(val_type),
                        "meta": {"optimized_out": True},
                    }
                raise

            # Handle different types
            type_str = str(val_type)
            if val_type.code == gdb.TYPE_CODE_PTR:
                return GdbValueSerializer._serialize_pointer(gdb_value, depth)
            elif val_type.code == gdb.TYPE_CODE_ARRAY:
                return GdbValueSerializer._serialize_array(gdb_value, depth)
            elif val_type.code == gdb.TYPE_CODE_STRUCT:
                return GdbValueSerializer._serialize_struct(gdb_value, depth)
            elif val_type.code == gdb.TYPE_CODE_UNION:
                return GdbValueSerializer._serialize_union(gdb_value, depth)
            elif val_type.code == gdb.TYPE_CODE_ENUM:
                return {"value": str(gdb_value), "type": type_str, "meta": {}}
            elif val_type.code in (gdb.TYPE_CODE_INT, gdb.TYPE_CODE_UINT):
                return GdbValueSerializer._serialize_integer(gdb_value, type_str)
            elif val_type.code == gdb.TYPE_CODE_FLT:
                return GdbValueSerializer._serialize_float(gdb_value, type_str)
            elif val_type.code == gdb.TYPE_CODE_CHAR:
                return {"value": chr(int(gdb_value)), "type": type_str, "meta": {}}
            elif val_type.code == gdb.TYPE_CODE_BOOL:
                return {"value": bool(gdb_value), "type": type_str, "meta": {}}
            else:
                # Default: string representation
                return {"value": str(gdb_value), "type": type_str, "meta": {}}

        except RuntimeError as e:
            if "optimized out" in str(e).lower():
                return {
                    "value": None,
                    "type": "unknown",
                    "meta": {"optimized_out": True},
                }
            raise
        except Exception as e:
            return {
                "value": None,
                "type": "error",
                "meta": {"error": str(e), "error_type": "serialization_failed"},
            }

    @staticmethod
    def _serialize_pointer(gdb_value: gdb.Value, depth: int) -> Dict[str, Any]:
        """Serialize a pointer type."""
        try:
            address = int(gdb_value)
            target_type = gdb_value.type.target()
            type_str = str(gdb_value.type)

            result = {
                "value": hex(address),
                "type": type_str,
                "address": hex(address),
                "meta": {},
            }

            # Check for CUDA address spaces
            if "@shared" in type_str:
                result["address_space"] = "shared"
            elif "@global" in type_str:
                result["address_space"] = "global"
            elif "@local" in type_str:
                result["address_space"] = "local"
            elif "@generic" in type_str:
                result["address_space"] = "generic"

            # Try to dereference if not too deep
            if depth < MAX_DEPTH - 1:
                try:
                    dereferenced = gdb_value.dereference()
                    result["dereferenced"] = GdbValueSerializer.serialize(dereferenced, depth + 1)
                except:
                    pass

            return result
        except Exception as e:
            return {
                "value": None,
                "type": str(gdb_value.type),
                "meta": {"error": str(e), "error_type": "pointer_serialization_failed"},
            }

    @staticmethod
    def _serialize_array(gdb_value: gdb.Value, depth: int) -> Dict[str, Any]:
        """Serialize an array type."""
        try:
            type_str = str(gdb_value.type)
            arr_type = gdb_value.type
            element_type = arr_type.target()
            element_count = int(arr_type.range()[1]) + 1

            actual_count = min(element_count, MAX_ARRAY_ELEMENTS)
            elements = []

            for i in range(actual_count):
                try:
                    elem = gdb_value[i]
                    elements.append(GdbValueSerializer.serialize(elem, depth + 1))
                except Exception:
                    elements.append({"error": f"Cannot access element {i}"})

            result = {
                "value": elements,
                "type": type_str,
                "meta": {
                    "element_count": element_count,
                    "displayed_count": actual_count,
                    "truncated": element_count > MAX_ARRAY_ELEMENTS,
                },
            }

            # Try to get address
            try:
                if gdb_value.address:
                    result["address"] = hex(int(gdb_value.address))
            except Exception:
                pass

            return result
        except Exception as e:
            return {
                "value": None,
                "type": "array",
                "meta": {"error": str(e), "error_type": "array_serialization_failed"},
            }

    @staticmethod
    def _serialize_struct(gdb_value: gdb.Value, depth: int) -> Dict[str, Any]:
        """Serialize a struct/union type."""
        try:
            type_str = str(gdb_value.type)
            fields = {}

            try:
                for field in gdb_value.type.fields():
                    if not field.is_static:
                        try:
                            field_value = gdb_value[field.name]
                            fields[field.name] = GdbValueSerializer.serialize(
                                field_value, depth + 1
                            )
                        except Exception as e:
                            fields[field.name] = {
                                "error": str(e),
                                "meta": {"optimized_out": True},
                            }
            except RuntimeError:
                # No fields accessible
                pass

            # Get address
            address = None
            try:
                if gdb_value.address:
                    address = hex(int(gdb_value.address))
            except Exception:
                pass

            return {
                "value": fields,
                "type": type_str,
                "address": address,
                "meta": {"field_count": len(fields)},
            }
        except Exception as e:
            return {
                "value": None,
                "type": "struct",
                "meta": {"error": str(e), "error_type": "struct_serialization_failed"},
            }

    @staticmethod
    def _serialize_union(gdb_value: gdb.Value, depth: int) -> Dict[str, Any]:
        """Serialize a union type."""
        # Treat union like struct for serialization
        return GdbValueSerializer._serialize_struct(gdb_value, depth)

    @staticmethod
    def _serialize_integer(gdb_value: gdb.Value, type_str: str) -> Dict[str, Any]:
        """Serialize an integer type."""
        try:
            int_val = int(gdb_value)
            result = {
                "value": int_val,
                "type": type_str,
                "meta": {},
            }

            # Add hex representation for integers
            if "char" not in type_str.lower():
                result["hex"] = hex(int_val)

            return result
        except Exception as e:
            return {
                "value": str(gdb_value),
                "type": type_str,
                "meta": {"error": str(e)},
            }

    @staticmethod
    def _serialize_float(gdb_value: gdb.Value, type_str: str) -> Dict[str, Any]:
        """Serialize a floating point type."""
        try:
            float_val = float(gdb_value)
            return {
                "value": float_val,
                "type": type_str,
                "meta": {},
            }
        except Exception as e:
            return {
                "value": str(gdb_value),
                "type": type_str,
                "meta": {"error": str(e)},
            }


def serialize_gdb_value(gdb_value: gdb.Value, max_depth: int = MAX_DEPTH) -> Dict[str, Any]:
    """Convenience function to serialize gdb.Value."""
    global MAX_DEPTH
    original_depth = MAX_DEPTH
    MAX_DEPTH = max_depth
    try:
        return GdbValueSerializer.serialize(gdb_value)
    finally:
        MAX_DEPTH = original_depth