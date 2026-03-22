"""
Pytest configuration and fixtures for cuda-gdb-cli tests.

Provides mock fixtures to simulate the cuda-gdb environment without
requiring actual CUDA hardware or cuda-gdb installation.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from enum import Enum
from typing import Any, Dict, Optional


# =============================================================================
# Mock GDB Types and Values
# =============================================================================

class MockGdbType:
    """Mock for gdb.Type"""

    def __init__(self, type_code: int, type_name: str = "int"):
        self._type_code = type_code
        self._type_name = type_name
        self._fields = []
        self._range = (0, 0)

    @property
    def code(self) -> int:
        return self._type_code

    def strip_typedefs(self):
        return self

    @property
    def fields(self):
        return self._fields

    def range(self):
        return self._range

    def __str__(self):
        return self._type_name


class MockGdbValue:
    """Mock for gdb.Value with various edge cases"""

    def __init__(
        self,
        value: Any,
        type_code: int,
        type_name: str = "int",
        address: Optional[int] = None,
        is_optimized_out: bool = False
    ):
        self._value = value
        self._type = MockGdbType(type_code, type_name)
        self._address = address
        self._is_optimized_out = is_optimized_out

    @property
    def type(self):
        return self._type

    @property
    def address(self):
        return self._address

    @property
    def is_optimized_out(self):
        return self._is_optimized_out

    def __int__(self):
        if self._is_optimized_out:
            raise MockGdbError("Variable is optimized out")
        return int(self._value)

    def __float__(self):
        if self._is_optimized_out:
            raise MockGdbError("Variable is optimized out")
        return float(self._value)

    def __str__(self):
        return str(self._value)

    def __getitem__(self, index):
        if isinstance(self._value, (list, tuple)):
            return self._value[index]
        raise MockGdbError("Value is not subscriptable")


class MockGdbError(Exception):
    """Mock for gdb.error"""
    pass


class MockGdbBreakpointEvent:
    """Mock for gdb.BreakpointEvent"""

    def __init__(self, breakpoint_numbers: list):
        self._breakpoints = [MockBp(n) for n in breakpoint_numbers]

    @property
    def breakpoints(self):
        return self._breakpoints


class MockGdbSignalEvent:
    """Mock for gdb.SignalEvent"""

    def __init__(self, signal_name: str):
        self._signal = signal_name

    @property
    def stop_signal(self):
        return self._signal


class MockBp:
    """Mock for GDB breakpoint"""

    def __init__(self, number: int):
        self.number = number


class MockFrame:
    """Mock for GDB frame"""

    def __init__(self, filename: str = "test.cu", line: int = 42):
        self._filename = filename
        self._line = line

    def find_sal(self):
        mock_sal = MagicMock()
        mock_sal.symtab = MagicMock()
        mock_sal.symtab.filename = self._filename
        mock_sal.line = self._line
        return mock_sal


# =============================================================================
# Type Code Constants (matching gdb.TYPE_CODE_*)
# =============================================================================

class TypeCode:
    TYPE_CODE_INT = 1
    TYPE_CODE_FLT = 2
    TYPE_CODE_STRUCT = 3
    TYPE_CODE_UNION = 4
    TYPE_CODE_ENUM = 5
    TYPE_CODE_ARRAY = 6
    TYPE_CODE_PTR = 7
    TYPE_CODE_CHAR = 8
    TYPE_CODE_BOOL = 9


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def mock_gdb():
    """Complete mock of gdb module"""
    with patch.dict('sys.modules', {'gdb': MagicMock()}):
        import sys
        gdb_mock = MagicMock()

        # Set up type codes
        gdb_mock.TYPE_CODE_INT = TypeCode.TYPE_CODE_INT
        gdb_mock.TYPE_CODE_FLT = TypeCode.TYPE_CODE_FLT
        gdb_mock.TYPE_CODE_STRUCT = TypeCode.TYPE_CODE_STRUCT
        gdb_mock.TYPE_CODE_UNION = TypeCode.TYPE_CODE_UNION
        gdb_mock.TYPE_CODE_ENUM = TypeCode.TYPE_CODE_ENUM
        gdb_mock.TYPE_CODE_ARRAY = TypeCode.TYPE_CODE_ARRAY
        gdb_mock.TYPE_CODE_PTR = TypeCode.TYPE_CODE_PTR
        gdb_mock.TYPE_CODE_CHAR = TypeCode.TYPE_CODE_CHAR
        gdb_mock.TYPE_CODE_BOOL = TypeCode.TYPE_CODE_BOOL

        # Set up error
        gdb_mock.error = MockGdbError

        # Set up event types
        gdb_mock.BreakpointEvent = MockGdbBreakpointEvent
        gdb_mock.SignalEvent = MockGdbSignalEvent

        # Set up Frame
        gdb_mock.selected_frame = MockFrame

        # Set up Thread
        gdb_mock.Thread = MagicMock()

        sys.modules['gdb'] = gdb_mock
        yield gdb_mock


@pytest.fixture
def mock_gdb_value():
    """Factory for creating mock gdb.Value objects"""
    def _create(value, type_code=TypeCode.TYPE_CODE_INT, type_name="int",
                address=None, is_optimized_out=False):
        return MockGdbValue(value, type_code, type_name, address, is_optimized_out)
    return _create


@pytest.fixture
def sample_integer_value(mock_gdb_value):
    """Basic integer value (e.g., threadIdx.x = 15)"""
    return mock_gdb_value(15, TypeCode.TYPE_CODE_INT, "int")


@pytest.fixture
def sample_float_value(mock_gdb_value):
    """Float value (e.g., float variable = 3.14)"""
    return mock_gdb_value(3.14, TypeCode.TYPE_CODE_FLT, "float")


@pytest.fixture
def sample_pointer_value(mock_gdb_value):
    """Pointer value (e.g., void* ptr = 0x7f1234560000)"""
    return mock_gdb_value(0x7f1234560000, TypeCode.TYPE_CODE_PTR, "void *")


@pytest.fixture
def sample_null_pointer(mock_gdb_value):
    """Null pointer"""
    return mock_gdb_value(0, TypeCode.TYPE_CODE_PTR, "void *")


@pytest.fixture
def sample_array_value(mock_gdb_value):
    """Array value (e.g., float arr[4] = {1.5, 2.3, 0.0, -1.2})"""
    arr = [1.5, 2.3, 0.0, -1.2]
    mock_val = mock_gdb_value(arr, TypeCode.TYPE_CODE_ARRAY, "float [4]")
    mock_val._type._range = (0, 3)
    return mock_val


@pytest.fixture
def sample_struct_value(mock_gdb_value):
    """Struct value (e.g., struct Point { float x; float y; })"""
    # Create a mock struct value
    mock_val = mock_gdb_value({}, TypeCode.TYPE_CODE_STRUCT, "struct Point")

    # Add mock fields
    field_x = MagicMock()
    field_x.name = "x"
    field_y = MagicMock()
    field_y.name = "y"

    mock_val._type._fields = [field_x, field_y]
    mock_val._value = {"x": 1.0, "y": 2.0}
    return mock_val


@pytest.fixture
def sample_optimized_out_value(mock_gdb_value):
    """Value that has been optimized out by compiler"""
    return mock_gdb_value(None, TypeCode.TYPE_CODE_INT, "int",
                          is_optimized_out=True)


@pytest.fixture
def sample_enum_value(mock_gdb_value):
    """Enum value (e.g., CUDA_EXCEPTION_14)"""
    return mock_gdb_value(14, TypeCode.TYPE_CODE_ENUM, "enum cuda_exception")


@pytest.fixture
def sample_bool_value(mock_gdb_value):
    """Boolean value"""
    return mock_gdb_value(True, TypeCode.TYPE_CODE_BOOL, "bool")


@pytest.fixture
def sample_long_string_value(mock_gdb_value):
    """String value exceeding MAX_STRING_LENGTH (4096 chars)"""
    long_str = "x" * 5000
    return mock_gdb_value(long_str, TypeCode.TYPE_CODE_CHAR, "char [5000]")


@pytest.fixture
def sample_nested_struct(mock_gdb_value):
    """Nested struct (depth > max_depth)"""
    # Create a deeply nested structure
    inner = {"value": 1}
    middle = {"inner": inner}
    outer = {"middle": middle}

    mock_val = mock_gdb_value(outer, TypeCode.TYPE_CODE_STRUCT, "OuterStruct")

    # Add fields that would cause deep recursion
    field_mid = MagicMock()
    field_mid.name = "middle"

    mock_val._type._fields = [field_mid]
    mock_val._value = outer
    return mock_val


# =============================================================================
# Modality State Fixtures
# =============================================================================

class DebugModality(Enum):
    """Debug modality states matching design.md"""
    INITIALIZING = "INITIALIZING"
    MUTABLE = "MUTABLE"       # Live mode - can read/write/execute
    IMMUTABLE = "IMMUTABLE"   # Coredump mode - read only
    RUNNING = "RUNNING"       # Target running (Live mode sub-state)
    STOPPED = "STOPPED"       # Target stopped (Live mode sub-state)
    CRASHED = "CRASHED"       # cuda-gdb itself crashed


@pytest.fixture
def modality_guard():
    """Create a fresh ModalityGuard instance"""
    import sys
    import os
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    sys.path.insert(0, src_path)

    # Import from actual implementation
    from state.modality_guard import ModalityGuard
    from state.modality_guard import DebugModality

    # Create fresh instance (need to reset singleton)
    ModalityGuard._instance = None
    guard = ModalityGuard()
    return guard


@pytest.fixture
def mutable_mode_guard(modality_guard):
    """ModalityGuard in MUTABLE (Live stopped) mode"""
    modality_guard.current_mode = DebugModality.STOPPED
    return modality_guard


@pytest.fixture
def immutable_mode_guard(modality_guard):
    """ModalityGuard in IMMUTABLE (Coredump) mode"""
    modality_guard.current_mode = DebugModality.IMMUTABLE
    return modality_guard


@pytest.fixture
def running_mode_guard(modality_guard):
    """ModalityGuard in RUNNING mode"""
    modality_guard.current_mode = DebugModality.RUNNING
    return modality_guard


# =============================================================================
# JSON-RPC Request/Response Fixtures
# =============================================================================

@pytest.fixture
def jsonrpc_request():
    """Factory for creating JSON-RPC requests"""
    def _create(method: str, params: Dict = None, request_id: int = 1):
        req = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params:
            req["params"] = params
        return req
    return _create


@pytest.fixture
def valid_cuda_set_focus_request(jsonrpc_request):
    """Valid cuda_set_focus request"""
    return jsonrpc_request("cuda_set_focus", {
        "block": [2, 0, 0],
        "thread": [15, 0, 0]
    })


@pytest.fixture
def valid_cuda_evaluate_var_request(jsonrpc_request):
    """Valid cuda_evaluate_var request"""
    return jsonrpc_request("cuda_evaluate_var", {
        "expression": "threadIdx.x"
    })


@pytest.fixture
def valid_cuda_execution_control_request(jsonrpc_request):
    """Valid cuda_execution_control request"""
    return jsonrpc_request("cuda_execution_control", {
        "action": "step"
    })


@pytest.fixture
def jsonrpc_response():
    """Factory for creating JSON-RPC responses"""
    def _success(request_id: int, result: Dict):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }

    def _error(request_id: int, code: int, message: str, data: Dict = None):
        error = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        if data:
            error["error"]["data"] = data
        return error

    return {"success": _success, "error": _error}


# =============================================================================
# CUDA Context Fixtures
# =============================================================================

@pytest.fixture
def cuda_kernel_info():
    """Sample kernel information"""
    return {
        "kernel_id": 0,
        "function_name": "matmul_kernel",
        "grid_dim": [128, 128, 1],
        "block_dim": [32, 32, 1],
        "shared_memory_bytes": 4096,
        "device": 0,
        "state": "stopped"
    }


@pytest.fixture
def cuda_focus_software_coords():
    """Software coordinates for GPU thread focus"""
    return {
        "block": [2, 0, 0],
        "thread": [15, 0, 0],
        "kernel": None
    }


@pytest.fixture
def cuda_focus_hardware_mapping():
    """Hardware mapping for GPU thread focus"""
    return {
        "device": 0,
        "sm": 7,
        "warp": 3,
        "lane": 15
    }


@pytest.fixture
def cuda_register_snapshot():
    """Sample register snapshot"""
    return {
        "R0": "0x00000042",
        "R1": "0x00000000",
        "R2": "0x7f123456",
        "R3": "0x00000010",
        "R4": "0xdeadbeef",
        "R5": "0x00000001"
    }


@pytest.fixture
def cuda_exception_info():
    """Sample CUDA exception information"""
    return {
        "status": "exception_detected",
        "exception_code": "CUDA_EXCEPTION_14",
        "exception_name": "Warp Illegal Address",
        "severity": "critical",
        "description": "Any lane within the warp accessed an illegal memory address.",
        "common_causes": [
            "Global memory buffer overflow",
            "Accessing device memory after cudaFree()"
        ],
        "errorpc": "0x555555557a80",
        "pc": "0x555555557a84",
        "faulting_instruction": "ST.E.U8 [R2], R0"
    }