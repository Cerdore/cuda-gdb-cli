"""End-to-End Integration Tests for CUDA-GDB-CLI.

These tests verify the complete debugging workflow from session creation
to cleanup. They test the integration between:
- CLI commands
- RPC client
- RPC server
- CUDA handlers
- Value formatter

Note: These tests mock the GDB layer since actual CUDA hardware
may not be available in the test environment.
"""

import pytest
import json
import sys
import os
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# =============================================================================
# Mock Setup
# =============================================================================

class MockGdbModule:
    """Complete mock of the gdb module for testing without CUDA hardware."""

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

    # Simulated outputs
    _outputs = {}
    _current_kernel = 0
    _current_block = [0, 0, 0]
    _current_thread = [0, 0, 0]

    @classmethod
    def reset(cls):
        cls._outputs = {
            "info cuda threads": """BlockIdx  ThreadIdx  To  Name           Filename   Line
* (0,0,0)  (0,0,0)    -   matmul_kernel  matmul.cu  28
  (0,0,0)  (1,0,0)    -   matmul_kernel  matmul.cu  28
""",
            "info cuda kernels": """Kernel  Function       GridDim      BlockDim     Device  Status
0       matmul_kernel  (32,32,1)    (16,16,1)    0       running
""",
            "info cuda devices": """Device  Name                    SMs  Cap  Threads/SM  Regs/SM  Mem
0       NVIDIA A100-SXM4-40GB   108  8.0  2048        65536    40GB
""",
            "info cuda exceptions": """Kernel  Block      Thread     Device  SM  Warp  Lane  Exception
0       (0,0,0)    (1,0,0)    0       0   0     1     CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS
""",
            "info cuda warps": """Warp  Device  SM    Active  Status
0     0       0     32      active
""",
            "info cuda lanes": """Lane  ThreadIdx   Active  Status
0     (0,0,0)     yes     active
1     (1,0,0)     yes     active
""",
            "cuda kernel": f"Current kernel: {cls._current_kernel}",
            "cuda block": f"({cls._current_block[0]},{cls._current_block[1]},{cls._current_block[2]})",
            "cuda thread": f"({cls._current_thread[0]},{cls._current_thread[1]},{cls._current_thread[2]})",
            "info cuda focus": f"Current CUDA focus: kernel {cls._current_kernel}, block ({cls._current_block[0]},{cls._current_block[1]},{cls._current_block[2]}), thread ({cls._current_thread[0]},{cls._current_thread[1]},{cls._current_thread[2]})",
            "backtrace": """#0  matmul_kernel (A=0x7fff00000000, B=0x7fff00010000, C=0x7fff00020000, M=32, N=32, K=32) at matmul.cu:28
#1  __device_stub__matmul_kernel at matmul.cu:15
""",
            "info threads": """  Id   Target Id         Frame
* 1    CUDA thread 0.0.0   matmul_kernel at matmul.cu:28
""",
        }
        cls._current_kernel = 0
        cls._current_block = [0, 0, 0]
        cls._current_thread = [0, 0, 0]

    @classmethod
    def set_output(cls, command, output):
        cls._outputs[command] = output

    @classmethod
    def execute(cls, command, to_string=False):
        if command in cls._outputs:
            return cls._outputs[command]
        # Handle focus switching commands
        if command.startswith("cuda kernel "):
            cls._current_kernel = int(command.split()[-1])
            return f"Switched to kernel {cls._current_kernel}"
        if command.startswith("cuda block "):
            coords = command.split()[-1].strip("()")
            x, y, z = map(int, coords.split(","))
            cls._current_block = [x, y, z]
            return f"Switched to block ({x},{y},{z})"
        if command.startswith("cuda thread "):
            coords = command.split()[-1].strip("()")
            x, y, z = map(int, coords.split(","))
            cls._current_thread = [x, y, z]
            return f"Switched to thread ({x},{y},{z})"
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
                return "matmul_kernel"
            def pc(self):
                return 0x7fff00001000
            def block(self):
                class MockBlock:
                    def __iter__(self):
                        return iter([])
                return MockBlock()
            def sal(self):
                class SAL:
                    symtab = type('obj', (), {'filename': 'matmul.cu'})()
                    line = 28
                return SAL()
        return MockFrame()

    @classmethod
    def post_event(cls, func):
        """Simulate gdb.post_event by executing immediately."""
        func()


class MockType:
    """Mock GDB type."""
    def __init__(self, name):
        self.name = name
        self.code = 6  # TYPE_CODE_INT

    def __str__(self):
        return self.name


@pytest.fixture(autouse=True)
def setup_gdb_mock():
    """Setup and teardown gdb mock for each test."""
    original_gdb = sys.modules.get('gdb')
    MockGdbModule.reset()
    sys.modules['gdb'] = MockGdbModule

    yield MockGdbModule

    if original_gdb:
        sys.modules['gdb'] = original_gdb
    elif 'gdb' in sys.modules:
        del sys.modules['gdb']


# =============================================================================
# E2E Tests: Session Workflow
# =============================================================================

class TestE2ESessionWorkflow:
    """Test the complete session lifecycle."""

    def test_session_lifecycle(self, setup_gdb_mock):
        """Test: load → commands → stop workflow."""
        # Clear module cache
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server' in mod or 'src.client' in mod:
                del sys.modules[mod]

        from src.gdb_server.cuda_handlers import (
            handle_cuda_threads,
            handle_cuda_kernels,
            handle_cuda_focus,
            handle_cuda_exceptions,
        )

        # Simulate session start
        session_id = "test-session-001"

        # 1. Check devices
        result = handle_cuda_devices()
        assert "cuda_devices" in result
        assert len(result["cuda_devices"]) == 1

        # 2. Check kernels
        result = handle_cuda_kernels()
        assert "cuda_kernels" in result
        assert len(result["cuda_kernels"]) == 1

        # 3. Check threads
        result = handle_cuda_threads()
        assert "cuda_threads" in result
        assert len(result["cuda_threads"]) == 2

        # 4. Check exceptions
        result = handle_cuda_exceptions()
        assert "cuda_exceptions" in result
        assert len(result["cuda_exceptions"]) == 1

        # 5. Focus on exception thread
        result = handle_cuda_focus(block=[0, 0, 0], thread=[1, 0, 0])
        assert "software_coords" in result

        # Session cleanup (implicit)


class TestE2ECudaHandlersIntegration:
    """Test integration between CUDA handlers."""

    def test_exception_to_focus_workflow(self, setup_gdb_mock):
        """Test: find exception → switch focus to exception thread."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.cuda_handlers import (
            handle_cuda_exceptions,
            handle_cuda_focus,
        )

        # 1. Find exception
        exc_result = handle_cuda_exceptions()
        assert len(exc_result["cuda_exceptions"]) == 1

        exc = exc_result["cuda_exceptions"][0]

        # 2. Switch focus to exception thread
        focus_result = handle_cuda_focus(
            kernel=exc["kernel"],
            block=exc["block"],
            thread=exc["thread"]
        )

        assert "software_coords" in focus_result

    def test_kernel_to_threads_workflow(self, setup_gdb_mock):
        """Test: list kernels → list threads for specific kernel."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.cuda_handlers import (
            handle_cuda_kernels,
            handle_cuda_threads,
        )

        # 1. List kernels
        kernels = handle_cuda_kernels()
        assert len(kernels["cuda_kernels"]) == 1

        kernel = kernels["cuda_kernels"][0]

        # 2. List threads (kernel filter would apply in real usage)
        threads = handle_cuda_threads(kernel=kernel["id"])
        assert len(threads["cuda_threads"]) == 2


class TestE2EMemoryInspection:
    """Test CUDA memory inspection workflow."""

    def test_memory_space_validation(self, setup_gdb_mock):
        """Test memory handler validates address spaces."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.cuda_handlers import handle_cuda_memory

        # Valid spaces
        valid_spaces = ["shared", "global", "local", "generic", "const", "param"]

        for space in valid_spaces:
            result = handle_cuda_memory(expr="test_var", space=space)
            # Should not have invalid space error
            if "error" in result:
                assert "invalid" not in result["error"].lower() or space not in valid_spaces

        # Invalid space
        result = handle_cuda_memory(expr="test_var", space="invalid_space")
        assert "error" in result

    def test_memory_missing_expr(self, setup_gdb_mock):
        """Test memory handler requires expr parameter."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.cuda_handlers import handle_cuda_memory

        result = handle_cuda_memory(space="shared")
        assert "error" in result


class TestE2EHandlerRegistry:
    """Test handler registration and retrieval."""

    def test_all_handlers_callable(self, setup_gdb_mock):
        """Test all registered handlers are callable."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.cuda_handlers import CUDA_HANDLERS

        for name, handler in CUDA_HANDLERS.items():
            assert callable(handler), f"Handler {name} is not callable"

    def test_handler_return_types(self, setup_gdb_mock):
        """Test all handlers return dict."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.cuda_handlers import CUDA_HANDLERS

        for name, handler in CUDA_HANDLERS.items():
            result = handler()
            assert isinstance(result, dict), f"Handler {name} did not return dict"


class TestE2EErrorHandling:
    """Test error handling across the system."""

    def test_gdb_error_propagation(self, setup_gdb_mock):
        """Test that GDB errors are properly propagated."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.cuda_handlers import handle_cuda_threads

        # Setup error condition
        def error_execute(cmd, to_string=False):
            raise MockGdbModule.error("CUDA context not available")

        setup_gdb_mock.execute = error_execute

        result = handle_cuda_threads()
        assert "error" in result or result["total_count"] == 0


# =============================================================================
# CPU Handler Tests
# =============================================================================

def handle_cuda_devices(**kwargs):
    """Helper to import and call handle_cuda_devices."""
    from src.gdb_server.cuda_handlers import handle_cuda_devices
    return handle_cuda_devices(**kwargs)


class TestE2ECPUHandlers:
    """Test CPU-side handlers (inherited from gdb-cli)."""

    def test_backtrace_handler(self, setup_gdb_mock):
        """Test backtrace handler."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server.gdb_rpc_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.gdb_rpc_server import handle_backtrace

        result = handle_backtrace()
        assert "output" in result or "error" in result

    def test_threads_handler(self, setup_gdb_mock):
        """Test threads handler."""
        for mod in list(sys.modules.keys()):
            if 'src.gdb_server.gdb_rpc_server' in mod:
                del sys.modules[mod]

        from src.gdb_server.gdb_rpc_server import handle_threads

        result = handle_threads()
        assert "output" in result or "error" in result


# =============================================================================
# Integration Test: JSON-RPC Round-trip
# =============================================================================

class TestE2EJsonRpcRoundTrip:
    """Test JSON-RPC encoding/decoding."""

    def test_request_decode_response_encode(self):
        """Test JSON-RPC request decoding and response encoding."""
        from src.gdb_server.json_rpc import (
            decode_request,
            create_success_response,
            create_error_response,
        )
        import json

        # Request
        request_str = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_threads",
            "params": {"limit": 10}
        })

        request = decode_request(request_str)
        assert request["method"] == "cuda_threads"
        assert request["params"]["limit"] == 10

        # Success response
        response = create_success_response(1, {"cuda_threads": []})
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response

        # Error response
        error_response = create_error_response(1, -32000, "GDB error", {"source": "test"})
        assert "error" in error_response
        assert error_response["error"]["code"] == -32000


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])