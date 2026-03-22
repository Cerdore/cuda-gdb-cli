"""Tests for CUDA handlers."""

import pytest
import sys
from unittest.mock import patch, MagicMock


class MockGdb:
    """Mock gdb module for testing without actual GDB."""
    TYPE_CODE_PTR = 1
    TYPE_CODE_ARRAY = 2
    TYPE_CODE_STRUCT = 3
    TYPE_CODE_UNION = 4
    TYPE_CODE_ENUM = 5
    TYPE_CODE_INT = 6
    TYPE_CODE_UINT = 7
    TYPE_CODE_FLT = 8
    TYPE_CODE_CHAR = 9
    TYPE_CODE_BOOL = 10

    class error(Exception):
        pass

    class MemoryError(Exception):
        pass

    _output_map = {}
    _selected_frame = None
    _should_error = False

    @classmethod
    def reset(cls):
        cls._output_map = {}
        cls._selected_frame = None
        cls._should_error = False

    @classmethod
    def set_output(cls, command, output):
        cls._output_map[command] = output

    @classmethod
    def set_error(cls, should_error=True):
        cls._should_error = should_error

    @classmethod
    def execute(cls, command, to_string=False):
        if cls._should_error:
            raise cls.error("GDB error")
        if command in cls._output_map:
            return cls._output_map[command]
        return ""

    @classmethod
    def parse_and_eval(cls, expr):
        class MockValue:
            def __init__(self, val):
                self._val = val
                self.type = MockType("int")
                self.address = None

            def __int__(self):
                return int(self._val)

            def __getitem__(self, idx):
                return MockValue(self._val)

            def dereference(self):
                return MockValue(self._val)

        return MockValue(42)

    @classmethod
    def selected_frame(cls):
        class MockFrame:
            def name(self):
                return "test_kernel"
            def pc(self):
                return 0x7fff00001000
            def sal(self):
                class SAL:
                    symtab = None
                    line = 28
                return SAL()
        return MockFrame()


class MockType:
    """Mock GDB type."""
    def __init__(self, name):
        self.name = name
        self.code = 6  # TYPE_CODE_INT


@pytest.fixture(autouse=True)
def setup_gdb_mock():
    """Setup and teardown gdb mock for each test."""
    # Save original gdb module if exists
    original_gdb = sys.modules.get('gdb')

    # Reset mock state
    MockGdb.reset()

    # Install mock
    sys.modules['gdb'] = MockGdb

    yield MockGdb

    # Restore original
    if original_gdb:
        sys.modules['gdb'] = original_gdb
    elif 'gdb' in sys.modules:
        del sys.modules['gdb']


class TestCudaThreadsHandler:
    """Tests for handle_cuda_threads."""

    def test_parse_basic_output(self, setup_gdb_mock, sample_cuda_threads_output):
        """Test parsing basic cuda threads output."""
        setup_gdb_mock.set_output("info cuda threads", sample_cuda_threads_output)

        # Clear module cache to pick up mock
        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_threads

        result = handle_cuda_threads()

        assert "cuda_threads" in result
        assert len(result["cuda_threads"]) == 4
        assert result["total_count"] == 4

        # Check first thread (current)
        first = result["cuda_threads"][0]
        assert first["is_current"] == True
        assert first["block_idx"] == [0, 0, 0]
        assert first["thread_idx"] == [0, 0, 0]
        assert first["name"] == "matmul_kernel"
        assert first["file"] == "matmul.cu"
        assert first["line"] == 28

    def test_empty_output(self, setup_gdb_mock):
        """Test handling empty output."""
        setup_gdb_mock.set_output("info cuda threads", "")

        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_threads

        result = handle_cuda_threads()

        assert result["cuda_threads"] == []
        assert result["total_count"] == 0

    def test_gdb_error(self, setup_gdb_mock):
        """Test handling GDB error."""
        setup_gdb_mock.set_error(True)

        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_threads

        result = handle_cuda_threads()

        assert "error" in result or result["total_count"] == 0


class TestCudaKernelsHandler:
    """Tests for handle_cuda_kernels."""

    def test_parse_kernels_output(self, setup_gdb_mock, sample_cuda_kernels_output):
        """Test parsing cuda kernels output."""
        setup_gdb_mock.set_output("info cuda kernels", sample_cuda_kernels_output)

        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_kernels

        result = handle_cuda_kernels()

        assert "cuda_kernels" in result
        assert len(result["cuda_kernels"]) == 2

        # Check first kernel
        kernel = result["cuda_kernels"][0]
        assert kernel["id"] == 0
        assert kernel["function"] == "matmul_kernel"
        assert kernel["grid_dim"] == [32, 32, 1]
        assert kernel["block_dim"] == [16, 16, 1]
        assert kernel["device"] == 0
        assert kernel["status"] == "running"


class TestCudaDevicesHandler:
    """Tests for handle_cuda_devices."""

    def test_parse_devices_output(self, setup_gdb_mock, sample_cuda_devices_output):
        """Test parsing cuda devices output."""
        setup_gdb_mock.set_output("info cuda devices", sample_cuda_devices_output)

        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_devices

        result = handle_cuda_devices()

        assert "cuda_devices" in result
        assert len(result["cuda_devices"]) == 1

        device = result["cuda_devices"][0]
        assert device["id"] == 0
        assert "A100" in device["name"]
        assert device["sms"] == 108
        assert device["capability"] == "8.0"


class TestCudaExceptionsHandler:
    """Tests for handle_cuda_exceptions."""

    def test_parse_exceptions_output(self, setup_gdb_mock, sample_cuda_exceptions_output):
        """Test parsing cuda exceptions output."""
        setup_gdb_mock.set_output("info cuda exceptions", sample_cuda_exceptions_output)

        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_exceptions

        result = handle_cuda_exceptions()

        assert "cuda_exceptions" in result
        assert len(result["cuda_exceptions"]) == 1

        exc = result["cuda_exceptions"][0]
        assert exc["kernel"] == 0
        assert exc["block"] == [0, 0, 0]
        assert exc["thread"] == [1, 0, 0]
        assert exc["type"] == "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS"
        assert "illegal memory" in exc["description"].lower()

    def test_no_exceptions(self, setup_gdb_mock):
        """Test when no exceptions exist."""
        setup_gdb_mock.set_output("info cuda exceptions", "No CUDA exceptions.\n")

        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_exceptions

        result = handle_cuda_exceptions()

        assert result["cuda_exceptions"] == []
        assert result["total_count"] == 0


class TestCudaMemoryHandler:
    """Tests for handle_cuda_memory."""

    def test_missing_expr(self, setup_gdb_mock):
        """Test error when expr is missing."""
        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_memory

        result = handle_cuda_memory()

        assert "error" in result
        assert "expr" in result["error"].lower()

    def test_invalid_space(self, setup_gdb_mock):
        """Test error for invalid address space."""
        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_memory

        result = handle_cuda_memory(expr="test_var", space="invalid")

        assert "error" in result
        assert "invalid" in result["error"].lower()


class TestCudaFocusHandler:
    """Tests for handle_cuda_focus."""

    def test_get_current_focus(self, setup_gdb_mock):
        """Test getting current focus without changes."""
        setup_gdb_mock.set_output("info cuda focus",
            "Current CUDA focus: kernel 0, block (0,0,0), thread (1,0,0)")
        setup_gdb_mock.set_output("cuda kernel", "Current kernel: 0")
        setup_gdb_mock.set_output("cuda block", "(0,0,0)")
        setup_gdb_mock.set_output("cuda thread", "(1,0,0)")

        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import handle_cuda_focus

        result = handle_cuda_focus()

        assert "software_coords" in result or "focus" in result


class TestHandlerRegistry:
    """Tests for handler registry."""

    def test_all_handlers_registered(self):
        """Test that all 8 CUDA handlers are registered."""
        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import CUDA_HANDLERS, list_cuda_handlers

        handlers = list_cuda_handlers()
        assert len(handlers) == 8
        assert "cuda_threads" in handlers
        assert "cuda_kernels" in handlers
        assert "cuda_focus" in handlers
        assert "cuda_devices" in handlers
        assert "cuda_exceptions" in handlers
        assert "cuda_memory" in handlers
        assert "cuda_warps" in handlers
        assert "cuda_lanes" in handlers

    def test_get_handler(self):
        """Test getting a handler by name."""
        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import get_cuda_handler

        handler = get_cuda_handler("cuda_threads")
        assert handler is not None
        assert callable(handler)

    def test_get_nonexistent_handler(self):
        """Test getting a nonexistent handler."""
        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import get_cuda_handler

        handler = get_cuda_handler("nonexistent")
        assert handler is None


class TestCudaExceptionMap:
    """Tests for CUDA exception mapping."""

    def test_exception_descriptions_exist(self):
        """Test that exception descriptions are defined."""
        if 'src.gdb_server.cuda_handlers' in sys.modules:
            del sys.modules['src.gdb_server.cuda_handlers']

        from src.gdb_server.cuda_handlers import CUDA_EXCEPTION_MAP

        assert "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS" in CUDA_EXCEPTION_MAP
        assert "CUDA_EXCEPTION_WARP_ASSERT" in CUDA_EXCEPTION_MAP

        # All descriptions should be non-empty strings
        for exc_type, desc in CUDA_EXCEPTION_MAP.items():
            assert isinstance(desc, str)
            assert len(desc) > 0