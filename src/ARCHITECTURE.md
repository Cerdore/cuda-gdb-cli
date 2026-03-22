# CUDA-GDB-CLI Architecture Document

> **Version**: v1.0
> **Date**: 2026-03-23
> **Status**: Architecture Design

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Source Directory Structure](#2-source-directory-structure)
- [3. Module Architecture](#3-module-architecture)
- [4. Thread Isolation Model](#4-thread-isolation-model)
- [5. Data Flow](#5-data-flow)
- [6. Module Interfaces](#6-module-interfaces)
- [7. Error Handling Strategy](#7-error-handling-strategy)
- [8. Configuration](#8-configuration)

---

## 1. Overview

### 1.1 Design Principles

| Principle | Description |
|-----------|-------------|
| **Structured I/O** | All Agent interactions via JSON-RPC 2.0 with strict schema validation |
| **Non-blocking** | Execution control never blocks RPC communication channel |
| **High-fidelity** | Direct GDB Python API access, no CLI text parsing |
| **Modality-aware** | Automatic Live/Coredump mode detection with operation guards |
| **Token-efficient** | Minimal response payloads, truncated arrays with metadata |

### 1.2 Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 1: Agent Client Layer                   │
│         (External: Cursor, Claude Desktop, Custom Agents)        │
│                     JSON-RPC 2.0 / MCP Protocol                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              Layer 2: Transport Proxy (transport_proxy.py)       │
│  - Process lifecycle management                                   │
│  - stdio/UDS bridge                                               │
│  - Timeout guards, crash recovery                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │ IPC (Unix Domain Socket)
┌──────────────────────────────▼──────────────────────────────────┐
│          Layer 3: Embedded RPC Engine (agent_rpc_server.py)      │
│           Running inside cuda-gdb Python interpreter              │
│  - RPC Listener Thread (gdb.Thread)                              │
│  - GDB Main Event Loop integration                                │
│  - Tool handlers, modality guard, serializers                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Source Directory Structure

```
src/
├── __init__.py
├── ARCHITECTURE.md                    # This document
│
├── agent_rpc_server.py                # Entry point - injected into cuda-gdb
│   # - Environment parsing
│   # - Modality detection
│   # - RPC listener startup
│   # - GDB event callback registration
│
├── transport_proxy.py                 # Layer 2: External process manager
│   # - cuda-gdb subprocess lifecycle
│   # - stdio/UDS bridge
│   # - Timeout handling, crash recovery
│
├── core/                              # Core infrastructure
│   ├── __init__.py
│   ├── rpc_listener.py               # RPC Listener Thread (gdb.Thread)
│   ├── command_queue.py              # Thread-safe command queue
│   ├── gdb_executor.py               # Safe GDB command execution
│   └── notification_channel.py       # Async event push channel
│
├── state/                             # State management
│   ├── __init__.py
│   ├── modality_guard.py             # FSM: INITIALIZING/MUTABLE/IMMUTABLE/RUNNING/STOPPED
│   ├── focus_tracker.py              # GPU thread focus state
│   └── session_state.py              # Session-wide state container
│
├── handlers/                          # Tool handlers (MCP endpoints)
│   ├── __init__.py
│   ├── base_handler.py               # Abstract base class for handlers
│   ├── focus.py                      # cuda_set_focus
│   ├── evaluate.py                   # cuda_evaluate_var
│   ├── registers.py                  # cuda_dump_warp_registers
│   ├── exception_analyzer.py         # cuda_analyze_exception
│   ├── execution.py                  # cuda_execution_control
│   ├── memory.py                     # cuda_read_shared_memory
│   ├── kernels.py                    # cuda_list_kernels
│   ├── backtrace.py                  # cuda_backtrace
│   ├── disassemble.py                # cuda_disassemble
│   ├── breakpoints.py                # cuda_set_breakpoint, cuda_remove_breakpoint
│   └── device.py                     # cuda_device_info
│
├── cuda/                              # CUDA-specific utilities
│   ├── __init__.py
│   ├── shared_memory.py              # @shared address space accessor
│   ├── register_probe.py             # Safe register enumeration
│   ├── exception_map.py              # CUDA_EXCEPTION enum mappings
│   ├── focus_manager.py              # Block/Thread/Kernel coordinate management
│   └── hardware_coords.py            # Device/SM/Warp/Lane mapping
│
├── serialization/                     # Data serialization
│   ├── __init__.py
│   ├── gdb_value.py                  # gdb.Value → JSON serializer
│   ├── json_rpc.py                   # JSON-RPC 2.0 encoder/decoder
│   └── truncation.py                 # Array/string truncation utilities
│
├── events/                            # Async event handling
│   ├── __init__.py
│   ├── stop_handler.py               # gdb.events.stop callback
│   ├── exit_handler.py               # gdb.events.exited callback
│   └── notification_builder.py       # Build JSON-RPC notifications
│
├── errors/                            # Error handling
│   ├── __init__.py
│   ├── codes.py                      # JSON-RPC error code constants
│   ├── gdb_errors.py                 # gdb.error → JSON-RPC error mapping
│   └── optimized_out.py              # Special handling for optimized variables
│
├── config/                            # Configuration
│   ├── __init__.py
│   ├── schema.py                     # Configuration schema definition
│   └── defaults.py                   # Default configuration values
│
└── utils/                             # Utilities
    ├── __init__.py
    ├── logger.py                     # Logging utilities
    └── validation.py                 # Input validation helpers
```

---

## 3. Module Architecture

### 3.1 Core Module

```
┌─────────────────────────────────────────────────────────────────┐
│                         core/ Module                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  RPCListener     │    │  CommandQueue    │                   │
│  │  (gdb.Thread)    │───►│  (queue.Queue)   │                   │
│  │                  │    │                  │                   │
│  │  - Socket I/O    │    │  - put(task)     │                   │
│  │  - JSON decode   │    │  - get()         │                   │
│  │  - Response send │    │  - Condition var │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           │                       ▼                              │
│           │              ┌──────────────────┐                    │
│           │              │  GdbExecutor     │                    │
│           │              │                  │                    │
│           │              │  - post_event()  │                    │
│           │              │  - execute()     │                    │
│           │              │  - wait_result() │                    │
│           │              └──────────────────┘                    │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                            │
│  │ NotificationChan │                                            │
│  │                  │                                            │
│  │  - send(notif)   │                                            │
│  │  - UDS write     │                                            │
│  └──────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 State Module

```
┌─────────────────────────────────────────────────────────────────┐
│                         state/ Module                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ModalityGuard (FSM)                    │   │
│  │                                                           │   │
│  │   INITIALIZING ──detect──► MUTABLE (Live)                │   │
│  │        │                    │                             │   │
│  │        └──detect coredump──►│                             │   │
│  │                             IMMUTABLE (Coredump)          │   │
│  │                                                           │   │
│  │   MUTABLE ──continue──► RUNNING ──stop──► STOPPED        │   │
│  │                                                           │   │
│  │   Methods:                                                │   │
│  │   - detect_modality() → mode info                        │   │
│  │   - check_permission(method) → None | error dict         │   │
│  │   - on_target_stopped()                                  │   │
│  │   - on_target_running()                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │  FocusTracker    │    │  SessionState    │                   │
│  │                  │    │                  │                   │
│  │  - block [x,y,z] │    │  - mode          │                   │
│  │  - thread [x,y,z]│    │  - target_type   │                   │
│  │  - kernel_id     │    │  - cuda_version  │                   │
│  │  - hardware map  │    │  - gdb_version   │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Handlers Module

```
┌─────────────────────────────────────────────────────────────────┐
│                        handlers/ Module                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │                  BaseHandler (ABC)                      │     │
│  │                                                         │     │
│  │  + method_name: str                                     │     │
│  │  + input_schema: dict                                   │     │
│  │  + execute(params: dict) → Result                       │     │
│  │  + validate(params: dict) → None | ValidationError     │     │
│  │  + check_modality() → None | Error                      │     │
│  └────────────────────────────────────────────────────────┘     │
│           △                                                      │
│           │ implements                                           │
│  ┌────────┴─────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  SetFocusHandler     EvaluateHandler    DumpRegsHandler  │   │
│  │  AnalyzeExcHandler   ExecControlHandler ReadMemHandler   │   │
│  │  ListKernelsHandler  BacktraceHandler   DisasmHandler    │   │
│  │  SetBpHandler        RemoveBpHandler    DeviceInfoHandler│   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 CUDA Module

```
┌─────────────────────────────────────────────────────────────────┐
│                         cuda/ Module                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ SharedMemAccessor│    │  RegisterProbe   │                   │
│  │                  │    │                  │                   │
│  │  + read_by_var() │    │  + dump_warp()   │                   │
│  │  + read_by_addr()│    │  + detect_limit()│                   │
│  │  - @shared inject│    │  - binary search │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ ExceptionAnalyzer│    │  FocusManager    │                   │
│  │                  │    │                  │                   │
│  │  + analyze()     │    │  + set_focus()   │                   │
│  │  + get_errorpc() │    │  + get_mapping() │                   │
│  │  + disasm_fault()│    │  + verify()      │                   │
│  │  - EXCEPTION_MAP │    │                  │                   │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │ HardwareCoords   │                                           │
│  │                  │                                           │
│  │  device: int     │                                           │
│  │  sm: int         │                                           │
│  │  warp: int       │                                           │
│  │  lane: int       │                                           │
│  └──────────────────┘                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Thread Isolation Model

### 4.1 Thread Architecture

The thread isolation model is the **critical design element** that prevents RPC channel deadlocks during blocking GDB commands.

```
┌─────────────────────────────────────────────────────────────────┐
│                    cuda-gdb Process                              │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  GDB Main Thread                                          │  │
│  │  (The only thread that can call gdb.execute/parse_and_eval)│  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  GDB Event Loop                                     │  │  │
│  │  │                                                     │  │  │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────────────┐ │  │  │
│  │  │  │ Breakpoint│ │ post_event│ │ Signal/Exception  │ │  │  │
│  │  │  │ Callbacks │ │ Tasks     │ │ Handlers          │ │  │  │
│  │  │  └───────────┘ └───────────┘ └───────────────────┘ │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  │  BLOCKING: gdb.execute("continue") runs here              │  │
│  │  BLOCKING: All tool handlers execute here                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  RPC Listener Thread (gdb.Thread subclass)                │  │
│  │  - Automatically calls gdb.block_signals()                │  │
│  │  - Independent socket I/O loop                            │  │
│  │  - NEVER calls gdb.execute/parse_and_eval directly        │  │
│  │  - Communicates via CommandQueue + gdb.post_event()       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Command Execution Flow

```
Request Lifecycle (Non-blocking for continue/step):

  1. RPC Thread receives JSON-RPC request
     │
     2. Decode JSON, validate schema
     │
     3. Create CommandTask with Event semaphore
     │
     4. command_queue.put(task)
     │
     5. gdb.post_event(process_task_callback)
     │
     6. Wait on task.completed.wait(timeout=25s)
     │
     ├─────────────────────────────────────────────────────────┐
     │                                                         │
     │  7. GDB Main Thread executes process_task_callback()    │
     │     │                                                   │
     │     8. Pop task from queue                              │
     │     │                                                   │
     │     9. ModalityGuard.check_permission(method)           │
     │     │                                                   │
     │     10. If blocking command (continue):                 │
     │         a. Return {"status": "running"} immediately     │
     │         b. gdb.post_event(async_execute)               │
     │         c. task.completed.set()                         │
     │         │                                               │
     │         └── async_execute runs later, blocks main thread│
     │             When done, gdb.events.stop fires            │
     │             → notification sent to Agent                │
     │                                                         │
     │     11. If non-blocking command:                        │
     │         a. Execute handler                              │
     │         b. Store result in task.result                  │
     │         c. task.completed.set()                         │
     │                                                         │
     └─────────────────────────────────────────────────────────┘
     │
     12. RPC Thread wakes up
     │
     13. Serialize task.result or task.error to JSON
     │
     14. Send JSON-RPC response via socket
```

### 4.3 Blocking Command Special Handling

```python
# Pseudocode for blocking command handling

class ExecutionController:
    """Handles continue/step/next with async execution"""

    def handle(self, params):
        # 1. Modality check (Coredump mode forbidden)
        if ModalityGuard.current_mode == DebugModality.IMMUTABLE:
            return error_response(-32003, "Coredump mode forbids execution control")

        # 2. State check (already running?)
        if self.is_running:
            if params["action"] == "interrupt":
                gdb.execute("interrupt", to_string=True)
                return {"status": "interrupt_sent"}
            return error_response(-32004, "Target already running")

        # 3. Mark as running
        self.is_running = True

        # 4. Schedule async execution (will block main thread)
        def async_execute():
            try:
                gdb.execute(params["action"], to_string=True)
            finally:
                self.is_running = False
                # gdb.events.stop will fire and send notification

        gdb.post_event(async_execute)

        # 5. Return immediately (don't wait for execution)
        return {
            "status": "running",
            "action": params["action"],
            "blocked": True,
            "message": "Target running. Stop event will be pushed as notification."
        }
```

---

## 5. Data Flow

### 5.1 Request Flow (Agent → GDB)

```
┌─────────┐    JSON-RPC     ┌──────────────┐    UDS     ┌─────────────────┐
│  Agent  │ ──────────────► │ Transport    │ ────────► │ RPC Listener    │
│         │                 │ Proxy        │            │ Thread          │
└─────────┘                 └──────────────┘            └────────┬────────┘
                                                                 │
                              JSON decode, validate               │
                                                                 ▼
                                                        ┌─────────────────┐
                                                        │ CommandQueue    │
                                                        │ .put(task)      │
                                                        └────────┬────────┘
                                                                 │
                                                        gdb.post_event()
                                                                 │
                                                                 ▼
┌─────────┐    Result      ┌──────────────┐            ┌─────────────────┐
│  Agent  │ ◄────────────── │ Transport    │ ◄──────────│ GDB Main Thread │
│         │    Response    │ Proxy        │            │ Tool Handler    │
└─────────┘                 └──────────────┘            └─────────────────┘
```

### 5.2 Async Notification Flow (GDB → Agent)

```
┌─────────────────────────────────────────────────────────────────┐
│  GDB Main Thread                                                │
│                                                                 │
│  GPU hits breakpoint / exception / exit                         │
│         │                                                       │
│         ▼                                                       │
│  gdb.events.stop.connect(callback) fires                        │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │ Stop Handler Callback                       │                │
│  │                                             │                │
│  │  1. Capture focus snapshot                  │                │
│  │  2. Detect CUDA exception ($errorpc)       │                │
│  │  3. Build JSON-RPC notification             │                │
│  │  4. notification_channel.send(json)        │                │
│  └─────────────────────────────────────────────┘                │
│                                                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ NotificationChannel │
                    │ (UDS write)         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐      ┌─────────┐
                    │ Transport Proxy     │ ────►│  Agent  │
                    │ (stdio forward)     │      │         │
                    └─────────────────────┘      └─────────┘
```

### 5.3 Memory Access Flow

```
cuda_read_shared_memory flow:

  Request: {"address": "0x20", "data_type": "float", "count": 8}

  1. ModalityGuard.check_permission("cuda_read_shared_memory")
     │
     2. SharedMemoryAccessor.read_by_address("0x20", "float", 8)
     │
     3. Construct expression: "*((@shared float*)0x20)@8"
        │                                    ↑
        │                    @shared modifier injected
        │
     4. gdb.parse_and_eval(expression)
        │
        ├─ Success ─► GdbValueSerializer.serialize(result)
        │              │
        │              └─► Return {"status": "ok", "data": [...]}
        │
        └─ Error ─► Error classification:
                    │
                    ├─ "IPC access denied" ─► error_type: "ipc_access_denied"
                    │
                    ├─ "out of bounds" ─► error_type: "address_out_of_bounds"
                    │
                    └─ Coredump mode ─► Append coredump_context metadata
```

---

## 6. Module Interfaces

### 6.1 Core Interfaces

```python
# core/rpc_listener.py

class RPCListenerThread(gdb.Thread):
    """
    RPC listener running in separate gdb.Thread.
    Never calls GDB APIs directly - uses command queue.
    """

    def __init__(self, socket_path: str, command_queue: CommandQueue):
        """
        Args:
            socket_path: Unix Domain Socket path for IPC
            command_queue: Thread-safe queue for command tasks
        """

    def run(self) -> None:
        """Main loop: accept connections, receive requests, send responses."""

    def send_notification(self, notification: dict) -> None:
        """Send async notification (called from main thread via callback)."""


# core/command_queue.py

class CommandTask:
    """Encapsulates a pending RPC command."""

    request_id: Optional[int | str]
    method: str
    params: dict
    result: Optional[dict]
    error: Optional[dict]
    completed: threading.Event  # Signals completion


class CommandQueue:
    """Thread-safe queue for RPC commands."""

    def put(self, task: CommandTask) -> None:
        """Enqueue a command task."""

    def get(self, timeout: Optional[float] = None) -> Optional[CommandTask]:
        """Dequeue a command task."""

    def notify_result(self, task: CommandTask) -> None:
        """Signal that task has result ready."""


# core/gdb_executor.py

class GdbExecutor:
    """
    Safe GDB command execution with post_event dispatch.
    All GDB API calls must go through this class.
    """

    @staticmethod
    def execute_sync(handler: Callable, params: dict, timeout: float = 25.0) -> dict:
        """
        Execute handler in GDB main thread, wait for result.

        Returns:
            {"result": ...} or {"error": ...}
        """

    @staticmethod
    def execute_async(handler: Callable, params: dict) -> dict:
        """
        Schedule handler in GDB main thread, return immediately.

        Returns:
            {"status": "running", "blocked": True}
        """
```

### 6.2 State Interfaces

```python
# state/modality_guard.py

from enum import Enum, auto

class DebugModality(Enum):
    INITIALIZING = auto()  # Startup
    MUTABLE = auto()       # Live mode - full access
    IMMUTABLE = auto()     # Coredump mode - read only
    RUNNING = auto()       # Live mode substate - target executing
    STOPPED = auto()       # Live mode substate - target paused
    CRASHED = auto()       # cuda-gdb crashed


class ModalityGuard:
    """
    FSM for debugging mode and permission checking.
    Singleton - shared across all handlers.
    """

    current_mode: DebugModality
    last_focus_snapshot: Optional[dict]

    def detect_modality(self) -> dict:
        """
        Detect Live vs Coredump mode from target info.

        Returns:
            {"mode": "MUTABLE"|"IMMUTABLE", "capabilities": {...}}
        """

    def check_permission(self, method_name: str) -> Optional[dict]:
        """
        Check if method is allowed in current mode.

        Returns:
            None if allowed, error dict if forbidden.
        """

    def on_target_stopped(self) -> None:
        """Transition to STOPPED, capture focus snapshot."""

    def on_target_running(self) -> None:
        """Transition to RUNNING."""


# state/focus_tracker.py

class FocusTracker:
    """Tracks current GPU thread focus coordinates."""

    software_coords: SoftwareCoords  # block, thread, kernel
    hardware_coords: HardwareCoords  # device, sm, warp, lane

    def update(self, block: List[int], thread: List[int], kernel: Optional[int]) -> dict:
        """Update focus and return hardware mapping."""

    def get_snapshot(self) -> dict:
        """Return current focus for notifications."""


@dataclass
class SoftwareCoords:
    block: Tuple[int, int, int]
    thread: Tuple[int, int, int]
    kernel: Optional[int]


@dataclass
class HardwareCoords:
    device: int
    sm: int
    warp: int
    lane: int
```

### 6.3 Handler Interfaces

```python
# handlers/base_handler.py

from abc import ABC, abstractmethod

class BaseHandler(ABC):
    """Abstract base for all tool handlers."""

    @property
    @abstractmethod
    def method_name(self) -> str:
        """JSON-RPC method name."""

    @property
    @abstractmethod
    def input_schema(self) -> dict:
        """JSON Schema for input validation."""

    @abstractmethod
    def execute(self, params: dict) -> dict:
        """
        Execute the tool logic. Runs in GDB main thread.

        Returns:
            {"status": "ok", ...} or {"status": "error", ...}
        """

    def validate(self, params: dict) -> None:
        """Validate params against schema. Raises ValidationError."""

    def check_modality(self) -> Optional[dict]:
        """Check modality permission. Returns error dict if forbidden."""


# handlers/focus.py

class SetFocusHandler(BaseHandler):
    method_name = "cuda_set_focus"
    input_schema = {
        "type": "object",
        "properties": {
            "block": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
            "thread": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3},
            "kernel": {"type": "integer"}
        },
        "required": ["block", "thread"]
    }

    def execute(self, params: dict) -> dict:
        """
        Switch GPU thread focus.

        Returns:
            {
                "status": "ok",
                "software_coords": {"block": [...], "thread": [...], "kernel": ...},
                "hardware_mapping": {"device": 0, "sm": 7, "warp": 3, "lane": 15},
                "verification": {"verified": True, "actual_thread": [...], "actual_block": [...]}
            }
        """


# handlers/execution.py

class ExecutionControlHandler(BaseHandler):
    method_name = "cuda_execution_control"
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["step", "next", "stepi", "continue", "finish", "interrupt"]}
        },
        "required": ["action"]
    }

    def execute(self, params: dict) -> dict:
        """
        Control program execution.

        For step/next: returns immediately with result
        For continue: returns {"status": "running"}, sends notification on stop
        For interrupt: sends SIGINT to running target
        """
```

### 6.4 CUDA Module Interfaces

```python
# cuda/shared_memory.py

class SharedMemoryAccessor:
    """Safe shared memory reader with @shared modifier injection."""

    @staticmethod
    def read_by_variable(variable_name: str, array_length: Optional[int] = None) -> dict:
        """
        Read shared memory by variable name.

        Args:
            variable_name: e.g., "s_data"
            array_length: elements to read if array

        Returns:
            {"status": "ok", "memory_space": "shared", "data": ...}
            or {"status": "error", "error_type": "ipc_access_denied", ...}
        """

    @staticmethod
    def read_by_address(address: str, data_type: str, count: int = 1) -> dict:
        """
        Read shared memory by physical address.
        Injects @shared modifier: *((@shared TYPE*)ADDR)@COUNT

        Args:
            address: Hex string like "0x20"
            data_type: C type like "float", "int"
            count: Number of elements
        """


# cuda/register_probe.py

class RegisterProbe:
    """Safe CUDA hardware register enumeration."""

    PREDICATE_REGISTER_COUNT = 7  # P0-P6 fixed

    @staticmethod
    def dump_warp_registers() -> dict:
        """
        Dump all allocated registers for current warp.
        Auto-detects register allocation limit to avoid segfault.

        Returns:
            {
                "status": "ok",
                "warp_info": {"sm": 7, "warp": 3},
                "general_registers": {"R0": "0x42", "R1": "0x0", ...},
                "predicate_registers": {"P0": "0x1", ...},
                "special_registers": {"CC": "0x0"},
                "register_count": 64,
                "max_possible": 255
            }
        """

    @staticmethod
    def _detect_register_limit() -> int:
        """
        Detect actual register allocation for current warp.

        Strategy 1: Parse 'info registers system' output
        Strategy 2: Binary search probe R0-R255
        """


# cuda/exception_analyzer.py

class CudaExceptionAnalyzer:
    """Analyze CUDA hardware exceptions."""

    EXCEPTION_MAP = {
        "CUDA_EXCEPTION_1": {
            "name": "Lane Illegal Address",
            "severity": "critical",
            "description": "...",
            "common_causes": [...]
        },
        # ... full mapping
    }

    @staticmethod
    def analyze() -> dict:
        """
        Analyze current CUDA exception context.

        Returns:
            {
                "status": "exception_detected",
                "exception_code": "CUDA_EXCEPTION_14",
                "exception_name": "Warp Illegal Address",
                "severity": "critical",
                "description": "...",
                "common_causes": [...],
                "errorpc": "0x555555557a80",
                "pc": "0x555555557a84",
                "faulting_instruction": "ST.E.U8 [R2], R0",
                "focus_at_exception": "kernel 0, block (2,0,0), thread (15,0,0)",
                "key_registers_snapshot": {"R0": "0xff", ...}
            }
            or {"status": "no_exception", "message": "..."}
        """
```

### 6.5 Serialization Interfaces

```python
# serialization/gdb_value.py

class GdbValueSerializer:
    """Convert gdb.Value to JSON-compatible Python objects."""

    MAX_ARRAY_ELEMENTS = 256
    MAX_STRING_LENGTH = 4096
    MAX_DEPTH = 5

    @staticmethod
    def serialize(gdb_value: gdb.Value, depth: int = 0) -> dict:
        """
        Recursively serialize gdb.Value.

        Returns:
            {
                "value": <actual value>,
                "type": <type string>,
                "address": <hex string, optional>,
                "hex": <hex string for integers, optional>,
                "meta": {
                    "optimized_out": True,  # if applicable
                    "truncated": True,      # if array truncated
                    "total_length": 1000,   # original array length
                    "displayed_length": 256
                }
            }
        """
```

---

## 7. Error Handling Strategy

### 7.1 Error Code Mapping

| Code | Constant | Description |
|------|----------|-------------|
| -32700 | `PARSE_ERROR` | JSON parse failure |
| -32600 | `INVALID_REQUEST` | Missing required fields |
| -32601 | `METHOD_NOT_FOUND` | Unknown method name |
| -32602 | `INVALID_PARAMS` | Parameter validation failure |
| -32603 | `INTERNAL_ERROR` | RPC engine exception |
| -32000 | `GDB_ERROR` | gdb.error exception |
| -32001 | `TIMEOUT` | Command timeout |
| -32002 | `PROCESS_CRASHED` | cuda-gdb crashed |
| -32003 | `MODALITY_FORBIDDEN` | Mode doesn't allow operation |
| -32004 | `TARGET_RUNNING` | Target running, need interrupt |
| -32005 | `OPTIMIZED_OUT` | Variable optimized away |
| -32006 | `NO_ACTIVE_KERNEL` | No GPU kernel active |
| -32007 | `MEMORY_TRUNCATED` | Coredump memory excluded |

### 7.2 Error Response Format

```python
# errors/gdb_errors.py

def map_gdb_error(error: gdb.error, method: str) -> dict:
    """
    Map gdb.error to structured JSON-RPC error.

    Returns:
        {
            "code": -32000,
            "message": "Human readable error",
            "data": {
                "source": "gdb_internal | rpc_engine | modality_guard",
                "error_type": "invalid_coordinates | optimized_out | ...",
                "hint": "Actionable suggestion for Agent",
                "details": {...}
            }
        }
    """
    error_msg = str(error)

    if "not within" in error_msg:
        return error_invalid_coordinates(error_msg)
    if "optimized out" in error_msg:
        return error_optimized_out(error_msg)
    if "no active kernel" in error_msg.lower():
        return error_no_active_kernel(error_msg)
    if "Cannot access memory" in error_msg:
        return error_memory_access(error_msg)

    return error_generic_gdb(error_msg)
```

---

## 8. Configuration

### 8.1 Configuration Schema

```yaml
# config.yaml

transport:
  ipc_type: "uds"
  socket_path_template: "/tmp/cuda-gdb-agent-{pid}.sock"

rpc:
  internal_timeout: 25
  max_response_size: 1048576
  max_serialization_depth: 5
  max_array_elements: 256

debugger:
  cuda_gdb_path: "cuda-gdb"
  extra_args: ""
  enable_tui: false

resilience:
  external_timeout: 30
  interrupt_grace_period: 5
  auto_restart_on_crash: false

logging:
  level: "INFO"
  output: "stderr"
```

### 8.2 Handler Registry

```python
# handlers/__init__.py

HANDLER_REGISTRY: Dict[str, Type[BaseHandler]] = {
    "cuda_set_focus": SetFocusHandler,
    "cuda_evaluate_var": EvaluateHandler,
    "cuda_dump_warp_registers": DumpRegistersHandler,
    "cuda_analyze_exception": AnalyzeExceptionHandler,
    "cuda_execution_control": ExecutionControlHandler,
    "cuda_read_shared_memory": ReadSharedMemoryHandler,
    "cuda_list_kernels": ListKernelsHandler,
    "cuda_backtrace": BacktraceHandler,
    "cuda_disassemble": DisassembleHandler,
    "cuda_set_breakpoint": SetBreakpointHandler,
    "cuda_remove_breakpoint": RemoveBreakpointHandler,
    "cuda_device_info": DeviceInfoHandler,
}
```

---

## References

- `design.md:57-130` - Three-layer architecture overview
- `design.md:375-520` - Thread isolation mechanism
- `design.md:989-1119` - Register probe implementation
- `design.md:1126-1357` - Exception analyzer implementation
- `design.md:1364-1489` - Focus manager implementation
- `design.md:1543-1698` - Modality guard FSM
- `api-schema.md:94-207` - cuda_set_focus specification
- `api-schema.md:211-362` - cuda_evaluate_var specification
- `api-schema.md:366-433` - cuda_dump_warp_registers specification
- `api-schema.md:436-509` - cuda_analyze_exception specification
- `api-schema.md:514-624` - cuda_execution_control specification
- `api-schema.md:1111-1128` - Error code definitions