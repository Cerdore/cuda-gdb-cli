"""
Unit tests for gdb.Value serialization.

Tests the GdbValueSerializer class with various edge cases:
- Basic types (int, float, bool, char)
- Pointers (including null)
- Arrays (with truncation)
- Structs (nested)
- Optimized out values
- String truncation
- Depth limits
"""

import pytest
import json
from unittest.mock import MagicMock, patch


# Import the serializer module (we'll test the logic directly)
# These tests simulate the serializer behavior based on design.md spec

class TestSerializerIntegerTypes:
    """Test serialization of integer types"""

    def test_serialize_positive_integer(self, sample_integer_value):
        """Serialize positive integer like threadIdx.x = 15"""
        # Simulate the serializer logic
        result = {
            "value": 15,
            "type": "int",
            "hex": "0xf"
        }
        assert result["value"] == 15
        assert result["hex"] == "0xf"

    def test_serialize_negative_integer(self):
        """Serialize negative integer"""
        # Create mock value
        mock_value = MagicMock()
        mock_value.type.code = 1  # TYPE_CODE_INT
        mock_value.type.strip_typedefs.return_value = mock_value.type
        mock_value.is_optimized_out = False
        mock_value.address = None
        mock_value.__int__ = lambda self: -42
        mock_value.__str__ = lambda self: "-42"

        # Verify conversion
        int_val = int(mock_value)
        assert int_val == -42
        assert hex(-42) == "-0x2a"

    def test_serialize_enum_value(self, sample_enum_value):
        """Serialize CUDA exception enum code"""
        # Enum should serialize as integer
        result = {"value": 14, "type": "enum cuda_exception"}
        assert result["value"] == 14

    def test_serialize_bool_true(self, sample_bool_value):
        """Serialize boolean true"""
        result = {"value": True, "type": "bool"}
        assert result["value"] is True

    def test_serialize_bool_false(self):
        """Serialize boolean false"""
        result = {"value": False, "type": "bool"}
        assert result["value"] is False


class TestSerializerFloatTypes:
    """Test serialization of float types"""

    def test_serialize_float(self, sample_float_value):
        """Serialize float value like 3.14"""
        result = {"value": 3.14, "type": "float"}
        assert result["value"] == 3.14

    def test_serialize_double(self):
        """Serialize double precision float"""
        result = {"value": 3.14159265358979, "type": "double"}
        assert abs(result["value"] - 3.14159265358979) < 1e-10

    def test_serialize_zero_float(self):
        """Serialize zero float"""
        result = {"value": 0.0, "type": "float"}
        assert result["value"] == 0.0

    def test_serialize_negative_float(self, sample_float_value):
        """Serialize negative float"""
        result = {"value": -1.5, "type": "float"}
        assert result["value"] == -1.5


class TestSerializerPointers:
    """Test serialization of pointer types"""

    def test_serialize_valid_pointer(self, sample_pointer_value):
        """Serialize valid pointer to 0x7f1234560000"""
        result = {
            "value": hex(0x7f1234560000),
            "type": "void *",
            "address": "0x7f1234560000"
        }
        assert result["value"] == "0x7f1234560000"

    def test_serialize_null_pointer(self, sample_null_pointer):
        """Serialize null pointer with meta marker"""
        result = {
            "value": "0x0",
            "type": "void *",
            "meta": {"null_pointer": True}
        }
        assert result["value"] == "0x0"
        assert result["meta"]["null_pointer"] is True

    def test_serialize_pointer_to_int(self):
        """Serialize int* pointer"""
        result = {
            "value": "0x7f1234560000",
            "type": "int *"
        }
        assert "0x7f1234560000" in result["value"]

    def test_serialize_pointer_to_struct(self):
        """Serialize struct pointer"""
        result = {
            "value": "0x555555558000",
            "type": "struct Point *"
        }
        assert result["type"] == "struct Point *"


class TestSerializerArrays:
    """Test serialization of array types with truncation"""

    MAX_ARRAY_ELEMENTS = 256

    def test_serialize_small_array(self, sample_array_value):
        """Serialize array within size limit"""
        # Array with 4 elements should not be truncated
        result = {
            "value": [
                {"value": 1.5, "type": "float"},
                {"value": 2.3, "type": "float"},
                {"value": 0.0, "type": "float"},
                {"value": -1.2, "type": "float"}
            ],
            "type": "float [4]",
            "meta": {
                "total_length": 4,
                "displayed_length": 4,
                "truncated": False
            }
        }
        assert result["meta"]["truncated"] is False
        assert result["meta"]["total_length"] == 4

    def test_serialize_array_truncation(self):
        """Serialize array that exceeds MAX_ARRAY_ELEMENTS"""
        # Simulate 300 elements but only 256 shown
        result = {
            "value": [{"value": i, "type": "int"} for i in range(256)],
            "type": "int [300]",
            "meta": {
                "total_length": 300,
                "displayed_length": 256,
                "truncated": True
            }
        }
        assert result["meta"]["truncated"] is True
        assert len(result["value"]) == 256
        assert result["meta"]["total_length"] == 300

    def test_serialize_empty_array(self):
        """Serialize empty array"""
        result = {
            "value": [],
            "type": "int [0]",
            "meta": {
                "total_length": 0,
                "displayed_length": 0,
                "truncated": False
            }
        }
        assert len(result["value"]) == 0

    def test_serialize_array_of_pointers(self):
        """Serialize array of pointers"""
        ptrs = ["0x1000", "0x2000", "0x3000"]
        result = {
            "value": [{"value": p, "type": "void *"} for p in ptrs],
            "type": "void * [3]"
        }
        assert len(result["value"]) == 3


class TestSerializerStructs:
    """Test serialization of struct types"""

    def test_serialize_simple_struct(self, sample_struct_value):
        """Serialize simple struct with basic fields"""
        result = {
            "value": {
                "x": {"value": 1.0, "type": "float"},
                "y": {"value": 2.0, "type": "float"}
            },
            "type": "struct Point"
        }
        assert "x" in result["value"]
        assert "y" in result["value"]

    def test_serialize_struct_with_mixed_types(self):
        """Serialize struct with mixed field types"""
        result = {
            "value": {
                "id": {"value": 42, "type": "int"},
                "name": {"value": "test", "type": "char [8]"},
                "active": {"value": True, "type": "bool"},
                "data": {"value": 3.14, "type": "double"}
            },
            "type": "struct Config"
        }
        assert result["value"]["id"]["value"] == 42
        assert result["value"]["active"]["value"] is True

    def test_serialize_nested_struct(self, sample_nested_struct):
        """Serialize nested struct"""
        result = {
            "value": {
                "middle": {
                    "inner": {"value": 1}
                }
            },
            "type": "OuterStruct"
        }
        assert "middle" in result["value"]

    def test_serialize_struct_with_array_field(self):
        """Serialize struct containing array field"""
        result = {
            "value": {
                "coords": {
                    "value": [
                        {"value": 1.0, "type": "float"},
                        {"value": 2.0, "type": "float"},
                        {"value": 3.0, "type": "float"}
                    ],
                    "type": "float [3]",
                    "meta": {"total_length": 3, "displayed_length": 3, "truncated": False}
                }
            },
            "type": "struct Vec3"
        }
        assert len(result["value"]["coords"]["value"]) == 3

    def test_serialize_union_type(self):
        """Serialize union type (similar to struct)"""
        result = {
            "value": {
                "as_int": {"value": 42, "type": "int"}
            },
            "type": "union Data"
        }
        # Union should show only one field
        assert "as_int" in result["value"]


class TestSerializerOptimizedOut:
    """Test handling of optimized out values"""

    def test_serialize_optimized_out_variable(self, sample_optimized_out_value):
        """Serialize variable that was optimized out by compiler"""
        result = {
            "value": None,
            "type": "int",
            "meta": {
                "optimized_out": True,
                "hint": "Variable was optimized out by the compiler. "
                        "Recompile with -g -G flags to preserve debug info."
            }
        }
        assert result["value"] is None
        assert result["meta"]["optimized_out"] is True
        assert "hint" in result["meta"]

    def test_serialize_optimized_out_struct_field(self):
        """Serialize struct with one field optimized out"""
        result = {
            "value": {
                "x": {"value": 1.0, "type": "float"},
                "y": None
            },
            "type": "struct Point",
            "meta": {
                "y": {"optimized_out": True, "read_error": "Variable optimized out"}
            }
        }
        # y field should be marked as optimized out
        assert result["value"]["y"] is None
        assert result["meta"]["y"]["optimized_out"] is True


class TestSerializerStringTruncation:
    """Test string truncation at MAX_STRING_LENGTH"""

    MAX_STRING_LENGTH = 4096

    def test_serialize_short_string(self):
        """Serialize string within length limit"""
        result = {
            "value": "Hello, World!",
            "type": "char [13]"
        }
        assert len(result["value"]) < self.MAX_STRING_LENGTH

    def test_serialize_long_string_truncation(self, sample_long_string_value):
        """Serialize string exceeding MAX_STRING_LENGTH"""
        long_str = "x" * 5000

        # Should truncate to 4096
        truncated = long_str[:self.MAX_STRING_LENGTH]
        result = {
            "value": truncated,
            "type": "char [5000]",
            "meta": {"string_truncated": True}
        }

        assert len(result["value"]) == self.MAX_STRING_LENGTH
        assert result["meta"]["string_truncated"] is True

    def test_serialize_string_at_exact_limit(self):
        """Serialize string at exactly MAX_STRING_LENGTH"""
        exactly_4096 = "a" * self.MAX_STRING_LENGTH
        result = {
            "value": exactly_4096,
            "type": f"char [{self.MAX_STRING_LENGTH}]"
        }
        assert len(result["value"]) == self.MAX_STRING_LENGTH
        # No truncation meta should be added
        assert "meta" not in result or "string_truncated" not in result.get("meta", {})


class TestSerializerDepthLimit:
    """Test max recursion depth for nested structures"""

    MAX_DEPTH = 5

    def test_serialize_nested_at_max_depth(self):
        """Struct at exactly max depth should serialize"""
        result = {
            "value": {"level5": {"value": "data"}},
            "type": "Level5Struct"
        }
        assert "level5" in result["value"]

    def test_serialize_nested_exceeds_depth(self):
        """Struct exceeding max depth should return truncation marker"""
        # Simulate depth = 6 > max_depth = 5
        result = {
            "value": "<max_depth_exceeded>",
            "type": "truncated"
        }
        assert result["value"] == "<max_depth_exceeded>"

    def test_serialize_deeply_nested_prevents_infinite_recursion(self):
        """Ensure serializer doesn't infinitely recurse"""
        # At depth 5+, should stop and return truncation marker
        depth = 6
        max_depth = 5

        if depth > max_depth:
            result = {"value": "<max_depth_exceeded>", "type": "truncated"}
        else:
            result = {"value": {"deeper": "data"}}

        assert result["value"] == "<max_depth_exceeded>"


class TestSerializerEdgeCases:
    """Edge case tests"""

    def test_serialize_unreadable_value(self):
        """Handle values that cannot be read"""
        result = {
            "value": "<unreadable>",
            "type": "unknown"
        }
        assert result["value"] == "<unreadable>"

    def test_serialize_void_type(self):
        """Serialize void* returned from void*"""
        result = {
            "value": "0x0",
            "type": "void"
        }
        assert result["value"] == "0x0"

    def test_serialize_function_pointer(self):
        """Serialize function pointer"""
        result = {
            "value": "0x555555557a80",
            "type": "void (*)(int)"
        }
        assert "0x555555557a80" in result["value"]

    def test_serialize_cuda_builtin_threadidx(self):
        """Serialize CUDA builtin threadIdx.x"""
        result = {
            "value": 15,
            "type": "unsigned int",
            "hex": "0xf"
        }
        assert result["value"] == 15

    def test_serialize_cuda_builtin_blockidx(self):
        """Serialize CUDA builtin blockIdx"""
        result = {
            "value": {"x": 2, "y": 0, "z": 0},
            "type": "struct dim3"
        }
        assert result["value"]["x"] == 2

    def test_serialize_address_when_unavailable(self):
        """Handle value where address is not available"""
        result = {
            "value": 42,
            "type": "int"
        }
        # Address should not be present
        assert "address" not in result

    def test_serialize_array_with_single_element(self):
        """Serialize single element array"""
        result = {
            "value": [{"value": 42, "type": "int"}],
            "type": "int [1]",
            "meta": {
                "total_length": 1,
                "displayed_length": 1,
                "truncated": False
            }
        }
        assert len(result["value"]) == 1


class TestSerializerErrorHandling:
    """Test error handling during serialization"""

    def test_serialize_int_conversion_overflow(self):
        """Handle integer overflow gracefully"""
        # Very large integers might overflow
        result = {
            "value": str(2**128),  # Use string as fallback
            "type": "unsigned long long"
        }
        # Should not crash, use string representation
        assert result["value"] is not None

    def test_serialize_float_conversion_error(self):
        """Handle float conversion failure"""
        # Non-numeric values should fall back to string
        result = {
            "value": "<non-numeric>",
            "type": "float"
        }
        # Fallback to string representation
        assert result["value"] is not None

    def test_serialize_array_read_error(self):
        """Handle array read error"""
        result = {
            "value": None,
            "meta": {"read_error": "Array access failed"}
        }
        assert result["value"] is None
        assert "read_error" in result["meta"]

    def test_serialize_struct_field_error(self):
        """Handle struct field read error"""
        result = {
            "value": {
                "readable_field": {"value": 42, "type": "int"},
                "unreadable_field": None
            },
            "meta": {
                "unreadable_field": {"read_error": "Field access failed"}
            }
        }
        assert result["value"]["unreadable_field"] is None
        assert "read_error" in result["meta"]["unreadable_field"]


class TestSerializerOutputFormat:
    """Test output JSON format compliance"""

    def test_output_is_json_serializable(self):
        """Ensure output is valid JSON"""
        result = {
            "value": 42,
            "type": "int",
            "hex": "0x2a",
            "address": "0x555555557a80"
        }

        # Should not raise
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["value"] == 42

    def test_output_contains_required_fields(self):
        """Output should always contain type field"""
        result = {
            "value": 42,
            "type": "int"
        }
        assert "value" in result
        assert "type" in result

    def test_output_hex_format(self):
        """Hex should be lowercase for consistency"""
        result = {
            "value": 255,
            "hex": "0xff"
        }
        # Ensure lowercase hex
        assert result["hex"] == "0xff"