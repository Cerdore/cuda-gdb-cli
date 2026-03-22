"""Pytest configuration for CUDA-GDB-CLI tests."""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_gdb():
    """Mock gdb module for testing without actual GDB."""
    class MockFrame:
        def __init__(self):
            self.name = "test_kernel"
            self.pc_value = 0x7fff00001000

        def name(self):
            return self.name

        def pc(self):
            return self.pc_value

        def sal(self):
            class SAL:
                symtab = None
                line = 28
            return SAL()

    class MockGdb:
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

        @classmethod
        def set_output(cls, command, output):
            cls._output_map[command] = output

        @classmethod
        def execute(cls, command, to_string=False):
            if command in cls._output_map:
                return cls._output_map[command]
            return ""

        @classmethod
        def parse_and_eval(cls, expr):
            # Return mock value
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
            if cls._selected_frame:
                return cls._selected_frame
            return MockFrame()

        @classmethod
        def set_selected_frame(cls, frame):
            cls._selected_frame = frame

    return MockGdb


@pytest.fixture
def sample_cuda_threads_output():
    """Sample output from 'info cuda threads' command."""
    return """BlockIdx  ThreadIdx  To  Name           Filename   Line
* (0,0,0)  (0,0,0)    -   matmul_kernel  matmul.cu  28
  (0,0,0)  (1,0,0)    -   matmul_kernel  matmul.cu  28
  (0,0,0)  (2,0,0)    -   matmul_kernel  matmul.cu  29
  (0,0,1)  (0,0,0)    -   matmul_kernel  matmul.cu  28
"""


@pytest.fixture
def sample_cuda_kernels_output():
    """Sample output from 'info cuda kernels' command."""
    return """Kernel  Function       GridDim      BlockDim     Device  Status
0       matmul_kernel  (32,32,1)    (16,16,1)    0       running
1       reduce_kernel  (64,1,1)     (256,1,1)    0       stopped
"""


@pytest.fixture
def sample_cuda_devices_output():
    """Sample output from 'info cuda devices' command."""
    return """Device  Name                    SMs  Cap  Threads/SM  Regs/SM  Mem
0       NVIDIA A100-SXM4-40GB   108  8.0  2048        65536    40GB
"""


@pytest.fixture
def sample_cuda_exceptions_output():
    """Sample output from 'info cuda exceptions' command."""
    return """Kernel  Block      Thread     Device  SM  Warp  Lane  Exception
0       (0,0,0)    (1,0,0)    0       0   0     1     CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS
"""