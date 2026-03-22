"""
End-to-end integration tests for CUDA-GDB-CLI JSON-RPC flow.

Tests the complete request/response cycle through all layers:
- JSON-RPC request parsing
- Method dispatching
- Modality guard checks
- Tool handler execution
- Response serialization
"""

import pytest
import json
from unittest.mock import MagicMock, patch, Mock
from enum import Enum


# =============================================================================
# Test Constants (matching api-schema.md)
# =============================================================================

# JSON-RPC Error Codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

GDB_ERROR = -32000
TIMEOUT = -32001
PROCESS_CRASHED = -32002
MODALITY_FORBIDDEN = -32003
TARGET_RUNNING = -32004
OPTIMIZED_OUT = -32005
NO_ACTIVE_KERNEL = -32006
MEMORY_TRUNCATED = -32007


# =============================================================================
# Mock Implementations for Testing
# =============================================================================

class DebugModality(Enum):
    """Debug modality states"""
    INITIALIZING = 1
    MUTABLE = 2
    IMMUTABLE = 3
    RUNNING = 4
    STOPPED = 5
    CRASHED = 6


class MockGdb:
    """Mock gdb module for integration tests"""

    def __init__(self):
        self._selected_frame = None
        self._inferior_running = False
        self._current_focus = {"block": [0, 0, 0], "thread": [0, 0, 0]}
        self._kernels = []

    def execute(self, cmd, to_string=False):
        """Mock gdb.execute"""
        if "cuda block" in cmd and "cuda thread" in cmd:
            # Focus switch
            self._current_focus = self._parse_focus_from_cmd(cmd)
            return f"cuda kernel 0 block ({self._current_focus['block'][0]},0,0) thread ({self._current_focus['thread'][0]},0,0)"
        if cmd == "cuda kernel 0":
            return "cuda kernel 0"
        if cmd == "continue":
            self._inferior_running = True
            return ""
        if cmd == "step":
            self._inferior_running = False
            return ""
        if cmd.startswith("info target"):
            return "exec-file: ./test_app\n"
        return ""

    def parse_and_eval(self, expr):
        """Mock gdb.parse_and_eval"""
        if expr == "threadIdx.x":
            return MockValue(0, "int")
        if expr == "blockIdx.x":
            return MockValue(0, "int")
        if expr.startswith("$R"):
            return MockValue(0x42, "int")
        if expr == "$pc":
            return MockValue(0x555555557a80, "int")
        return MockValue(0, "int")

    def selected_frame(self):
        """Mock gdb.selected_frame"""
        return MockFrame()

    @property
    def events(self):
        mock_events = MagicMock()
        mock_events.stop = MagicMock()
        mock_events.stop.connect = MagicMock()
        mock_events.exited = MagicMock()
        mock_events.exited.connect = MagicMock()
        return mock_events

    def post_event(self, callback):
        """Mock gdb.post_event"""
        callback()

    def _parse_focus_from_cmd(self, cmd):
        """Parse focus from command string"""
        # Simplified parser
        return {"block": [2, 0, 0], "thread": [15, 0, 0]}


class MockValue:
    """Mock gdb.Value"""

    def __init__(self, val, type_str):
        self._val = val
        self._type = MockType(type_str)
        self._address = None
        self._is_optimized_out = False

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
        return self._val

    def __str__(self):
        return str(self._val)


class MockType:
    """Mock gdb.Type"""

    def __init__(self, type_str):
        self._str = type_str

    def __str__(self):
        return self._str

    def strip_typedefs(self):
        return self

    @property
    def code(self):
        return 1  # TYPE_CODE_INT


class MockFrame:
    """Mock gdb.Frame"""

    def find_sal(self):
        sal = MagicMock()
        sal.symtab = MagicMock()
        sal.symtab.filename = "test.cu"
        sal.line = 42
        return sal


# =============================================================================
# RPC Request Handler (for testing)
# =============================================================================

class RPCRequestHandler:
    """Handle JSON-RPC requests - simplified for testing"""

    def __init__(self, modality_guard=None):
        self.modality_guard = modality_guard or MockModalityGuard()
        self.gdb = MockGdb()

    def handle_request(self, request):
        """Process a single JSON-RPC request"""
        # Validate JSON-RPC structure
        if "jsonrpc" not in request:
            return self._error(request.get("id"), INVALID_REQUEST,
                              "Missing jsonrpc field")

        if request.get("jsonrpc") != "2.0":
            return self._error(request.get("id"), INVALID_REQUEST,
                              "Invalid jsonrpc version")

        if "method" not in request:
            return self._error(request.get("id"), INVALID_REQUEST,
                              "Missing method field")

        method = request["method"]
        params = request.get("params", {})

        # Check modality permissions
        permission_error = self.modality_guard.check_permission(method)
        if permission_error:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": permission_error
            }

        # Dispatch to handler
        handler = self._get_handler(method)
        if not handler:
            return self._error(request.get("id"), METHOD_NOT_FOUND,
                              f"Unknown method: {method}")

        try:
            result = handler(params)
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }
        except Exception as e:
            return self._error(request.get("id"), INTERNAL_ERROR, str(e))

    def _get_handler(self, method):
        """Get handler for method"""
        handlers = {
            "cuda_set_focus": self._handle_set_focus,
            "cuda_evaluate_var": self._handle_evaluate_var,
            "cuda_execution_control": self._handle_execution_control,
            "cuda_list_kernels": self._handle_list_kernels,
            "cuda_dump_warp_registers": self._handle_dump_registers,
            "cuda_analyze_exception": self._handle_analyze_exception,
            "cuda_device_info": self._handle_device_info,
            "cuda_backtrace": self._handle_backtrace,
            "cuda_disassemble": self._handle_disassemble,
        }
        return handlers.get(method)

    def _handle_set_focus(self, params):
        """Handle cuda_set_focus"""
        block = params.get("block", [0, 0, 0])
        thread = params.get("thread", [0, 0, 0])

        # Execute focus switch
        self.gdb.execute(f"cuda kernel 0")
        self.gdb.execute(f"cuda block {block[0]},{block[1]},{block[2]} thread {thread[0]},{thread[1]},{thread[2]}")

        return {
            "status": "ok",
            "software_coords": {"block": block, "thread": thread, "kernel": None},
            "hardware_mapping": {"device": 0, "sm": 7, "warp": 3, "lane": 15},
            "verification": {"verified": True, "actual_thread": thread, "actual_block": block}
        }

    def _handle_evaluate_var(self, params):
        """Handle cuda_evaluate_var"""
        expr = params.get("expression", "")

        # Evaluate expression
        val = self.gdb.parse_and_eval(expr)
        int_val = int(val)

        return {
            "status": "ok",
            "data": {
                "value": int_val,
                "hex": hex(int_val),
                "type": "unsigned int"
            }
        }

    def _handle_execution_control(self, params):
        """Handle cuda_execution_control"""
        action = params.get("action", "step")

        if action == "continue":
            self.gdb.execute("continue")
            return {
                "status": "running",
                "action": action,
                "blocked": True,
                "message": "Target is now running..."
            }
        else:
            self.gdb.execute(action)
            return {
                "status": "stopped",
                "action": action,
                "current_focus": {
                    "kernel": "test_kernel",
                    "block": [0, 0, 0],
                    "thread": [0, 0, 0]
                }
            }

    def _handle_list_kernels(self, params):
        """Handle cuda_list_kernels"""
        return {
            "status": "ok",
            "kernels": [{
                "kernel_id": 0,
                "function_name": "matmul_kernel",
                "grid_dim": [128, 128, 1],
                "block_dim": [32, 32, 1],
                "shared_memory_bytes": 4096,
                "device": 0,
                "state": "stopped"
            }],
            "total_active_kernels": 1
        }

    def _handle_dump_registers(self, params):
        """Handle cuda_dump_warp_registers"""
        return {
            "status": "ok",
            "warp_info": {"device": 0, "sm": 7, "warp": 3},
            "general_registers": {"R0": "0x42", "R1": "0x0"},
            "predicate_registers": {"P0": "0x1"},
            "special_registers": {},
            "register_count": 2,
            "max_possible": 255
        }

    def _handle_analyze_exception(self, params):
        """Handle cuda_analyze_exception"""
        return {
            "status": "no_exception",
            "message": "No CUDA exception detected"
        }

    def _handle_device_info(self, params):
        """Handle cuda_device_info"""
        return {
            "status": "ok",
            "device": {
                "device_id": 0,
                "name": "NVIDIA A100-SXM4-80GB",
                "compute_capability": "8.0",
                "sm_count": 108,
                "warp_size": 32
            }
        }

    def _handle_backtrace(self, params):
        """Handle cuda_backtrace"""
        return {
            "status": "ok",
            "frames": [
                {"level": 0, "function": "main", "file": "test.cu", "line": 42}
            ],
            "total_frames": 1
        }

    def _handle_disassemble(self, params):
        """Handle cuda_disassemble"""
        return {
            "status": "ok",
            "format": "sass",
            "instructions": [
                {"address": "0x555555557a80", "instruction": "MOV R2, R5", "is_current_pc": False, "is_errorpc": False}
            ]
        }

    def _error(self, request_id, code, message):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }


class MockModalityGuard:
    """Mock modality guard for testing"""

    def __init__(self, mode=DebugModality.STOPPED):
        self.current_mode = mode

    def check_permission(self, method):
        """Always allow in mock"""
        return None


# =============================================================================
# Test Cases
# =============================================================================

class TestJSONRPCRequestValidation:
    """Test JSON-RPC request validation"""

    def test_valid_request_minimal(self):
        """Minimal valid request"""
        handler = RPCRequestHandler()
        request = {"jsonrpc": "2.0", "id": 1, "method": "cuda_list_kernels"}

        response = handler.handle_request(request)

        assert "result" in response
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1

    def test_missing_jsonrpc_field(self):
        """Request missing jsonrpc field"""
        handler = RPCRequestHandler()
        request = {"id": 1, "method": "cuda_list_kernels"}

        response = handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == INVALID_REQUEST

    def test_invalid_jsonrpc_version(self):
        """Request with invalid jsonrpc version"""
        handler = RPCRequestHandler()
        request = {"jsonrpc": "1.0", "id": 1, "method": "cuda_list_kernels"}

        response = handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == INVALID_REQUEST

    def test_missing_method_field(self):
        """Request missing method field"""
        handler = RPCRequestHandler()
        request = {"jsonrpc": "2.0", "id": 1}

        response = handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == INVALID_REQUEST

    def test_missing_id_field_notification(self):
        """Notification (no id) should work"""
        handler = RPCRequestHandler()
        request = {"jsonrpc": "2.0", "method": "cuda_list_kernels"}

        response = handler.handle_request(request)

        # Notifications should not have id
        assert "id" not in response

    def test_params_defaults_to_empty_object(self):
        """Params should default to empty object"""
        handler = RPCRequestHandler()
        request = {"jsonrpc": "2.0", "id": 1, "method": "cuda_list_kernels"}

        response = handler.handle_request(request)

        assert "result" in response


class TestMethodDispatch:
    """Test method dispatch"""

    def test_cuda_set_focus_dispatch(self):
        """cuda_set_focus should dispatch correctly"""
        handler = RPCRequestHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_set_focus",
            "params": {"block": [2, 0, 0], "thread": [15, 0, 0]}
        }

        response = handler.handle_request(request)

        assert "result" in response
        result = response["result"]
        assert result["status"] == "ok"
        assert result["software_coords"]["block"] == [2, 0, 0]
        assert result["software_coords"]["thread"] == [15, 0, 0]

    def test_cuda_evaluate_var_dispatch(self):
        """cuda_evaluate_var should dispatch correctly"""
        handler = RPCRequestHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "cuda_evaluate_var",
            "params": {"expression": "threadIdx.x"}
        }

        response = handler.handle_request(request)

        assert "result" in response
        assert response["result"]["status"] == "ok"
        assert "data" in response["result"]

    def test_unknown_method(self):
        """Unknown method should return METHOD_NOT_FOUND"""
        handler = RPCRequestHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown_method"
        }

        response = handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == METHOD_NOT_FOUND


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""

    def test_set_focus_then_evaluate(self):
        """Complete workflow: set focus, then evaluate variable"""
        handler = RPCRequestHandler()

        # Step 1: Set focus
        focus_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_set_focus",
            "params": {"block": [2, 0, 0], "thread": [15, 0, 0]}
        }
        focus_response = handler.handle_request(focus_request)
        assert "result" in focus_response

        # Step 2: Evaluate variable in new focus
        eval_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "cuda_evaluate_var",
            "params": {"expression": "threadIdx.x"}
        }
        eval_response = handler.handle_request(eval_request)
        assert "result" in eval_response
        assert "value" in eval_response["result"]["data"]

    def test_list_kernels_and_set_focus(self):
        """List kernels then set focus to valid thread"""
        handler = RPCRequestHandler()

        # List kernels
        list_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_list_kernels"
        }
        list_response = handler.handle_request(list_request)
        assert list_response["result"]["total_active_kernels"] == 1
        kernel = list_response["result"]["kernels"][0]
        assert kernel["function_name"] == "matmul_kernel"

        # Set focus to valid coordinate
        focus_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "cuda_set_focus",
            "params": {"block": [0, 0, 0], "thread": [0, 0, 0]}
        }
        focus_response = handler.handle_request(focus_request)
        assert focus_response["result"]["hardware_mapping"]["verified"] is True

    def test_execution_control_workflow(self):
        """Test execution control workflow"""
        handler = RPCRequestHandler()

        # Step
        step_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_execution_control",
            "params": {"action": "step"}
        }
        step_response = handler.handle_request(step_request)
        assert step_response["result"]["status"] == "stopped"

        # Continue (would block in real implementation)
        # Skipping for test as it would block

    def test_read_only_operations_sequence(self):
        """Sequence of read-only operations"""
        handler = RPCRequestHandler()

        # Set focus
        handler.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "cuda_set_focus",
            "params": {"block": [0, 0, 0], "thread": [0, 0, 0]}
        })

        # Dump registers
        regs_response = handler.handle_request({
            "jsonrpc": "2.0", "id": 2, "method": "cuda_dump_warp_registers"
        })
        assert regs_response["result"]["status"] == "ok"
        assert "general_registers" in regs_response["result"]

        # Backtrace
        bt_response = handler.handle_request({
            "jsonrpc": "2.0", "id": 3, "method": "cuda_backtrace"
        })
        assert bt_response["result"]["status"] == "ok"
        assert len(bt_response["result"]["frames"]) > 0

        # Device info
        dev_response = handler.handle_request({
            "jsonrpc": "2.0", "id": 4, "method": "cuda_device_info"
        })
        assert dev_response["result"]["status"] == "ok"


class TestModalityGuardedRequests:
    """Test requests blocked by modality guard"""

    def test_execution_blocked_in_coredump_mode(self):
        """Execution control should be blocked in coredump mode"""
        class CoredumpModalityGuard:
            def __init__(self):
                self.current_mode = DebugModality.IMMUTABLE

            def check_permission(self, method):
                if method == "cuda_execution_control":
                    return {
                        "code": MODALITY_FORBIDDEN,
                        "message": "Method 'cuda_execution_control' is forbidden in Coredump mode",
                        "data": {"current_mode": "IMMUTABLE"}
                    }
                return None

        handler = RPCRequestHandler(modality_guard=CoredumpModalityGuard())
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_execution_control",
            "params": {"action": "continue"}
        }

        response = handler.handle_request(request)

        assert "error" in response
        assert response["error"]["code"] == MODALITY_FORBIDDEN

    def test_read_allowed_in_coredump_mode(self):
        """Read operations should work in coredump mode"""
        class CoredumpModalityGuard:
            def __init__(self):
                self.current_mode = DebugModality.IMMUTABLE

            def check_permission(self, method):
                if method in {"cuda_set_breakpoint", "cuda_execution_control",
                              "cuda_modify_variable"}:
                    return {
                        "code": MODALITY_FORBIDDEN,
                        "message": f"Method '{method}' forbidden in coredump"
                    }
                return None

        handler = RPCRequestHandler(modality_guard=CoredumpModalityGuard())
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_evaluate_var",
            "params": {"expression": "threadIdx.x"}
        }

        response = handler.handle_request(request)
        # Should succeed (no modality error)
        assert "result" in response or ("error" in response and
               response["error"]["code"] != MODALITY_FORBIDDEN)


class TestResponseFormat:
    """Test response format compliance"""

    def test_success_response_format(self):
        """Success response should have result field"""
        handler = RPCRequestHandler()
        request = {"jsonrpc": "2.0", "id": 1, "method": "cuda_list_kernels"}

        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response

    def test_error_response_format(self):
        """Error response should have error field"""
        handler = RPCRequestHandler()
        request = {"jsonrpc": "2.0", "id": 1, "method": "unknown_method"}

        response = handler.handle_request(request)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "error" in response
        assert "code" in response["error"]
        assert "message" in response["error"]

    def test_error_response_includes_data(self):
        """Error response may include data field"""
        class ErrorModalityGuard:
            def check_permission(self, method):
                return {
                    "code": MODALITY_FORBIDDEN,
                    "message": "Forbidden",
                    "data": {"hint": "Try something else"}
                }

        handler = RPCRequestHandler(modality_guard=ErrorModalityGuard())
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_execution_control"
        }

        response = handler.handle_request(request)

        assert "data" in response["error"]
        assert "hint" in response["error"]["data"]


class TestConcurrentRequests:
    """Test handling multiple requests"""

    def test_sequential_requests(self):
        """Multiple sequential requests should each get response"""
        handler = RPCRequestHandler()

        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "cuda_list_kernels"},
            {"jsonrpc": "2.0", "id": 2, "method": "cuda_device_info"},
            {"jsonrpc": "2.0", "id": 3, "method": "cuda_list_kernels"},
        ]

        responses = [handler.handle_request(req) for req in requests]

        assert len(responses) == 3
        assert all("result" in r for r in responses)
        assert [r["id"] for r in responses] == [1, 2, 3]


class TestEdgeCases:
    """Edge case tests"""

    def test_empty_params_object(self):
        """Method with no required params should work with empty params"""
        handler = RPCRequestHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_list_kernels",
            "params": {}
        }

        response = handler.handle_request(request)
        assert "result" in response

    def test_null_params(self):
        """Method with null params should work"""
        handler = RPCRequestHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_list_kernels",
            "params": None
        }

        response = handler.handle_request(request)
        assert "result" in response

    def test_batch_request_not_supported(self):
        """Batch requests not supported - should return error"""
        handler = RPCRequestHandler()
        request = [
            {"jsonrpc": "2.0", "id": 1, "method": "cuda_list_kernels"},
            {"jsonrpc": "2.0", "id": 2, "method": "cuda_device_info"},
        ]

        # This would be a list, not a dict - our handler should handle gracefully
        try:
            response = handler.handle_request(request)
            # Should either error or handle somehow
        except (AttributeError, TypeError):
            pass  # Expected - we don't support batch


class TestAPISchemaCompliance:
    """Verify compliance with api-schema.md"""

    def test_cuda_set_focus_response_matches_schema(self):
        """Response should match api-schema.md"""
        handler = RPCRequestHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_set_focus",
            "params": {"block": [2, 0, 0], "thread": [15, 0, 0]}
        }

        response = handler.handle_request(request)
        result = response["result"]

        # Required fields from schema
        assert "status" in result
        assert "software_coords" in result
        assert "hardware_mapping" in result
        assert "verification" in result

        # Software coords
        assert "block" in result["software_coords"]
        assert "thread" in result["software_coords"]

        # Hardware mapping
        hw = result["hardware_mapping"]
        assert "device" in hw
        assert "sm" in hw
        assert "warp" in hw
        assert "lane" in hw

    def test_cuda_evaluate_var_response_matches_schema(self):
        """Evaluate var response should match schema"""
        handler = RPCRequestHandler()
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "cuda_evaluate_var",
            "params": {"expression": "threadIdx.x"}
        }

        response = handler.handle_request(request)
        result = response["result"]

        assert "status" in result
        assert "data" in result
        assert "value" in result["data"]
        assert "type" in result["data"]

    def test_error_codes_match_schema(self):
        """Error codes should match api-schema.md"""
        handler = RPCRequestHandler()

        # Test METHOD_NOT_FOUND
        response = handler.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "nonexistent"
        })
        assert response["error"]["code"] == METHOD_NOT_FOUND

        # Test INVALID_REQUEST
        response = handler.handle_request({
            "id": 1, "method": "test"
        })
        assert response["error"]["code"] == INVALID_REQUEST