# CUDA-GDB-CLI Architecture

## Overview

CUDA-GDB-CLI is a fork of gdb-cli with CUDA GPU debugging capabilities. It allows AI Agents (Claude Code, Codex CLI, Cursor, etc.) to debug CUDA programs via bash commands.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Agent (Claude Code / Codex CLI)                    │
│                            Tool: bash shell                                  │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │  bash commands
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        cuda-gdb-cli (Python CLI, Click)                      │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  cli.py                                                             │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │    │
│  │  │ CPU Commands │  │ CUDA Commands│  │ Session Management       │  │    │
│  │  │ bt, threads  │  │ cuda-threads │  │ load, attach, stop       │  │    │
│  │  │ eval, memory │  │ cuda-kernels │  └──────────────────────────┘  │    │
│  │  │ disasm, exec │  │ cuda-focus   │                               │    │
│  │  │              │  │ cuda-devices │                               │    │
│  │  │              │  │ cuda-excepts │                               │    │
│  │  │              │  │ cuda-memory  │                               │    │
│  │  │              │  │ cuda-warps   │                               │    │
│  │  └──────────────┘  └──────────────┘                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     │  GDBClient.call(method, params)       │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  client.py                                                          │    │
│  │  • Unix Domain Socket connection                                    │    │
│  │  • JSON-RPC 2.0 request encoding                                    │    │
│  │  • Socket path: /tmp/cuda-gdb-{session_id}.sock                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │  Unix Domain Socket (JSON)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     cuda-gdb Process (Background Daemon)                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    GDB Python RPC Server (Embedded)                  │    │
│  │                                                                      │    │
│  │  ┌─────────────────────┐    ┌─────────────────────┐                 │    │
│  │  │  RPC Listener       │    │  Command Queue      │                 │    │
│  │  │  (Background Thread)│    │  gdb_executor.py    │                 │    │
│  │  │  • socket accept    │───▶│                     │                 │    │
│  │  │  • recv JSON-RPC    │    │  gdb.post_event()   │                 │    │
│  │  │  • send response    │    │  Main thread safe   │                 │    │
│  │  └─────────────────────┘    └──────────┬──────────┘                 │    │
│  │                                        │                             │    │
│  │                                        ▼                             │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                     Handler Registry                         │    │    │
│  │  │                                                             │    │    │
│  │  │  ┌───────────────────┐    ┌────────────────────────────┐   │    │    │
│  │  │  │ CPU Handlers      │    │ CUDA Handlers              │   │    │    │
│  │  │  │ (from gdb-cli)    │    │ (to be implemented)        │   │    │    │
│  │  │  │ • backtrace       │    │ • cuda_threads             │   │    │    │
│  │  │  │ • threads         │    │ • cuda_kernels             │   │    │    │
│  │  │  │ • evaluate        │    │ • cuda_focus               │   │    │    │
│  │  │  │ • locals          │    │ • cuda_devices             │   │    │    │
│  │  │  │ • memory          │    │ • cuda_exceptions          │   │    │    │
│  │  │  │ • disassemble     │    │ • cuda_memory              │   │    │    │
│  │  │  │ • exec            │    │ • cuda_warps               │   │    │    │
│  │  │  └───────────────────┘    └────────────────────────────┘   │    │    │
│  │  │                                          │                  │    │    │
│  │  │                                          ▼                  │    │    │
│  │  │  ┌─────────────────────────────────────────────────────┐   │    │    │
│  │  │  │  Modality Guard (FSM)                               │   │    │    │
│  │  │  │  IMMUTABLE (coredump) / MUTABLE (live)              │   │    │    │
│  │  │  │  RUNNING / STOPPED / CRASHED                        │   │    │    │
│  │  │  └─────────────────────────────────────────────────────┘   │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Value Formatter (gdb.Value → JSON)                         │    │    │
│  │  │  Supports CUDA address spaces: @shared, @global, @local     │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    NVIDIA cuda-gdb (GDB Superset)                    │    │
│  │                                                                      │    │
│  │  • GDB Python API: gdb.execute(), gdb.parse_and_eval()              │    │
│  │  • CUDA Extensions: info cuda threads/kernels/devices/exceptions    │    │
│  │  • GPU Thread Model: Grid → Block → Thread                          │    │
│  │  • Address Spaces: @shared, @global, @local, @generic               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Thread Isolation Model

**Critical Design Constraint**: GDB Python API is NOT thread-safe. All GDB API calls must execute on the main GDB thread.

```
┌─────────────────────────────────────────────────────────────────┐
│                     cuda-gdb Process                            │
│                                                                 │
│  ┌─────────────────────┐      ┌─────────────────────────────┐  │
│  │  Main Thread        │      │  RPC Listener Thread        │  │
│  │  (GDB Main Thread)  │      │  (Background Thread)        │  │
│  │                     │      │                             │  │
│  │  • gdb.execute()    │◀─────│  • socket.accept()          │  │
│  │  • gdb.parse_eval() │      │  • recv JSON-RPC request    │  │
│  │  • Handler execution│      │  • send JSON-RPC response   │  │
│  │                     │      │                             │  │
│  │  ┌───────────────┐  │      │  gdb.post_event(handler)    │  │
│  │  │ Event Loop    │  │◀─────│  ─────────────────────────▶ │  │
│  │  │ (GDB Built-in)│  │      │                             │  │
│  │  └───────────────┘  │      │  ⚠️ NEVER call GDB API      │  │
│  │                     │      │     directly from RPC thread│  │
│  └─────────────────────┘      └─────────────────────────────┘  │
│                                                                 │
│  Key: gdb.post_event() is the ONLY safe cross-thread call      │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation in gdb_executor.py

```python
class GdbExecutor:
    @staticmethod
    def execute_sync(handler, params, timeout=25.0):
        """Execute handler in GDB main thread, wait for result."""
        result_container = {}
        completed_event = threading.Event()

        def execute_on_main_thread():
            try:
                result_container["result"] = handler(params)
            except Exception as e:
                result_container["error"] = {"code": -32000, "message": str(e)}
            finally:
                completed_event.set()

        # Schedule on main thread
        gdb.post_event(execute_on_main_thread)

        # Wait for completion
        if not completed_event.wait(timeout=timeout):
            return {"error": {"code": -32001, "message": "Command timeout"}}
        return result_container
```

## Data Flow

```
AI Agent calls:
cuda-gdb-cli cuda-threads -s f465d650 --kernel 0
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│ cli.py: cuda_threads_cmd()                                    │
│   client.call("cuda_threads", kernel=0, limit=50)             │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│ client.py: GDBClient.call()                                   │
│   → JSON-RPC request:                                         │
│   {"jsonrpc":"2.0","id":1,"method":"cuda_threads",            │
│    "params":{"kernel":0,"limit":50}}                          │
│   → Send to Unix Socket: /tmp/cuda-gdb-f465d650.sock          │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│ gdb_rpc_server.py (RPC Listener Thread)                       │
│   → Receive request, parse method="cuda_threads"              │
│   → Lookup handler: handlers["cuda_threads"]                  │
│   → gdb.post_event(execute_on_main_thread)                    │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│ gdb_executor.py (Main Thread Execution)                       │
│   → handler(params) scheduled on main thread                  │
│   → modality_guard.check_permission("cuda_threads")           │
│   → Returns None (read-only operation, allowed)               │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│ cuda_handlers.py: handle_cuda_threads()                       │
│   → output = gdb.execute("info cuda threads", to_string=True) │
│   → threads = _parse_cuda_threads_output(output)              │
│   → Filter kernel=0, truncate to limit=50                     │
│   → Return {"cuda_threads": [...], "total_count": N}          │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│ value_formatter.py: Serialize result                          │
│   → gdb.Value → JSON (CUDA address spaces supported)          │
└───────────────────────────────────────────────────────────────┘
                │
                ▼
Response to AI Agent:
{
  "cuda_threads": [
    {"kernel": 0, "block_idx": [0,0,0], "thread_idx": [0,0,0],
     "name": "matmul_kernel", "file": "matmul.cu", "line": 28}
  ],
  "total_count": 16384,
  "truncated": true
}
```

## Implementation Status

| Module | Status | Description |
|--------|--------|-------------|
| `cli.py` | ✅ Complete | CUDA subcommands defined |
| `client.py` | ✅ Complete | Unix Socket client |
| `launcher.py` | ✅ Complete | cuda-gdb startup logic |
| `safety.py` | ✅ Complete | CUDA command whitelist |
| `modality_guard.py` | ✅ Complete | FSM state machine |
| `value_formatter.py` | ✅ Complete | Address space support |
| `gdb_executor.py` | ✅ Complete | Thread isolation scheduling |
| **`cuda_handlers.py`** | ❌ TODO | **Core: 8 CUDA handlers** |
| **`gdb_rpc_server.py`** | ⚠️ Modify | Register CUDA handlers |

## CUDA Handlers to Implement

1. **handle_cuda_threads** - Parse `info cuda threads` output
2. **handle_cuda_kernels** - Parse `info cuda kernels` output
3. **handle_cuda_focus** - View/switch GPU focus
4. **handle_cuda_devices** - Parse `info cuda devices` output
5. **handle_cuda_exceptions** - Parse `info cuda exceptions` output
6. **handle_cuda_memory** - Read `@shared/@global/@local` memory
7. **handle_cuda_warps** - Parse `info cuda warps` output
8. **handle_cuda_lanes** - Parse `info cuda lanes` output

## GPU Thread Model

CUDA introduces a hierarchical thread model:

```
Grid (三维坐标)
  │
  ├── Block (0,0,0)          Block (1,0,0)          ...
  │     │                       │
  │     ├── Thread (0,0,0)      ├── Thread (0,0,0)
  │     ├── Thread (1,0,0)      ├── Thread (1,0,0)
  │     ├── ...                 ├── ...
  │     └── Thread (15,15,0)    └── Thread (15,15,0)
  │
  └── Block (31,31,0)
        └── ...

Hardware Mapping:
- Grid → Device (GPU)
- Block → SM (Streaming Multiprocessor)
- Thread → Lane (within Warp)
- Warp → 32 threads executed in lockstep
```

## CUDA Address Spaces

| Space | Syntax | Description |
|-------|--------|-------------|
| Global | `@global` | Device memory, visible to all threads |
| Shared | `@shared` | Per-block shared memory, fast |
| Local | `@local` | Per-thread private memory |
| Generic | `@generic` | Default address space |

## Modality Guard States

```
                    ┌─────────────────┐
                    │ INITIALIZING    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ IMMUTABLE       │           │ MUTABLE         │
    │ (Coredump Mode) │           │ (Live Mode)     │
    │ Read-Only       │           │ Full Access     │
    └─────────────────┘           └────────┬────────┘
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                              ▼                       ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │ STOPPED         │◀───▶│ RUNNING         │
                    │ Target Paused   │     │ Target Running  │
                    └─────────────────┘     └─────────────────┘
```

## Session Management

Sessions are identified by UUID and persisted in `/tmp/cuda-gdb-{session_id}.sock`.

```
# Load coredump
cuda-gdb-cli load --binary ./app --core ./core.12345
→ {"session_id": "f465d650", "mode": "core", "status": "started"}

# Attach to live process
cuda-gdb-cli attach --pid 9876 --binary ./app
→ {"session_id": "a1b2c3d4", "mode": "attach", "status": "started"}

# Stop session
cuda-gdb-cli stop -s f465d650
→ {"session_id": "f465d650", "status": "stopped"}
```

## Error Codes

Following JSON-RPC 2.0 specification:

| Code | Description |
|------|-------------|
| -32000 | GDB internal error |
| -32001 | Command timeout |
| -32002 | Modality forbidden |
| -32003 | Safety violation |
| -32600 | Invalid request |
| -32601 | Method not found |
| -32602 | Invalid params |