# CUDA-GDB-CLI

> AI Agent interface for CUDA GPU debugging via bash commands

A fork of [gdb-cli](https://github.com/Cerdore/gdb-cli) extended with CUDA debugging capabilities. Enables AI Agents (Claude Code, Codex CLI, Cursor, etc.) to debug CUDA GPU programs through structured JSON output.

**[中文文档](#中文文档)**

---

## Overview

```
┌─────────────────┐     bash      ┌─────────────────┐     JSON-RPC     ┌─────────────────┐
│   AI Agent      │ ───────────▶ │  cuda-gdb-cli   │ ───────────────▶ │   cuda-gdb      │
│ (Claude Code)   │              │    (Python)     │   Unix Socket    │  (RPC Server)   │
└─────────────────┘ ◀─────────── └─────────────────┘ ◀─────────────── └─────────────────┘
                         JSON                     structured output
```

## Installation

```bash
pip install cuda-gdb-cli

# Verify dependencies
cuda-gdb --version  # Requires CUDA Toolkit
nvidia-smi          # Requires NVIDIA driver
```

## Commands

### Session Management

| Command | Description | Example |
|---------|-------------|---------|
| `load` | Load coredump for analysis | `cuda-gdb-cli load --binary ./app --core ./core.12345` |
| `attach` | Attach to running process | `cuda-gdb-cli attach --pid 9876 --binary ./app` |
| `stop` | End debugging session | `cuda-gdb-cli stop -s $SESSION` |

### CPU Debugging (inherited from gdb-cli)

| Command | Description | Example |
|---------|-------------|---------|
| `bt` | Backtrace | `cuda-gdb-cli bt -s $SESSION` |
| `threads` | List CPU threads | `cuda-gdb-cli threads -s $SESSION` |
| `eval-cmd` | Evaluate expression | `cuda-gdb-cli eval-cmd -s $SESSION "my_var"` |
| `locals-cmd` | Show local variables | `cuda-gdb-cli locals-cmd -s $SESSION` |
| `memory` | Read memory | `cuda-gdb-cli memory -s $SESSION -a 0x7fff0000 -c 32` |
| `disasm` | Disassemble | `cuda-gdb-cli disasm -s $SESSION --count 10` |
| `exec` | Execute GDB command | `cuda-gdb-cli exec -s $SESSION "info registers"` |

### CUDA GPU Debugging

| Command | Description | Example |
|---------|-------------|---------|
| `cuda-kernels` | List active kernels | `cuda-gdb-cli cuda-kernels -s $SESSION` |
| `cuda-threads` | List GPU threads | `cuda-gdb-cli cuda-threads -s $SESSION --kernel 0` |
| `cuda-focus` | View/switch GPU focus | `cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"` |
| `cuda-devices` | Show GPU topology | `cuda-gdb-cli cuda-devices -s $SESSION` |
| `cuda-exceptions` | Show CUDA exceptions | `cuda-gdb-cli cuda-exceptions -s $SESSION` |
| `cuda-memory` | Read GPU memory space | `cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "smem" --type float --count 32` |
| `cuda-warps` | Show warp status | `cuda-gdb-cli cuda-warps -s $SESSION` |
| `cuda-lanes` | Show lane status | `cuda-gdb-cli cuda-lanes -s $SESSION` |

## Output Format

All commands return **structured JSON** for reliable Agent parsing:

```json
{
  "cuda_threads": [
    {
      "kernel": 0,
      "block_idx": [0, 0, 0],
      "thread_idx": [0, 0, 0],
      "name": "matmul_kernel",
      "file": "matmul.cu",
      "line": 28,
      "is_current": true,
      "exception": null
    }
  ],
  "total_count": 16384,
  "truncated": true
}
```

## Quick Start

```bash
# 1. Load CUDA coredump
cuda-gdb-cli load --binary ./app --core ./core.12345
# Output: {"session_id": "f465d650", "mode": "core", "status": "started"}

SESSION="f465d650"

# 2. Get GPU overview
cuda-gdb-cli cuda-devices -s $SESSION
cuda-gdb-cli cuda-kernels -s $SESSION
cuda-gdb-cli cuda-exceptions -s $SESSION

# 3. Focus on crash point
cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"

# 4. Inspect state
cuda-gdb-cli bt -s $SESSION
cuda-gdb-cli locals-cmd -s $SESSION
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "tile_A" --type float --count 16

# 5. End session
cuda-gdb-cli stop -s $SESSION
```

## CUDA Address Spaces

| Space | Flag | Description |
|-------|------|-------------|
| Global | `--space global` | Device memory, visible to all threads |
| Shared | `--space shared` | Per-block shared memory, fast |
| Local | `--space local` | Per-thread private memory |
| Generic | `--space generic` | Default address space |

## CUDA Exception Types

| Exception | Description |
|-----------|-------------|
| `CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS` | Thread accessed invalid memory |
| `CUDA_EXCEPTION_WARP_ASSERT` | GPU `assert()` triggered |
| `CUDA_EXCEPTION_LANE_INVALID_PC` | Invalid program counter |
| `CUDA_EXCEPTION_WARP_MISALIGNED` | Misaligned memory access |

## Architecture

```
src/gdb_server/
├── cuda_handlers.py      # 8 CUDA command handlers
├── gdb_rpc_server.py     # JSON-RPC server (embedded in cuda-gdb)
├── gdb_executor.py       # Thread-safe GDB execution
├── modality_guard.py     # FSM for mode detection (coredump/live)
├── value_formatter.py    # gdb.Value → JSON serialization
└── focus_tracker.py      # GPU thread focus tracking
```

**Key Design:**
- RPC listener runs in background thread
- All GDB API calls dispatched to main thread via `gdb.post_event()`
- Thread safety enforced through `GdbExecutor`

## Error Codes (JSON-RPC 2.0)

| Code | Name | Description |
|------|------|-------------|
| -32000 | `GDB_ERROR` | GDB internal error |
| -32001 | `TIMEOUT` | Command timeout |
| -32003 | `MODALITY_FORBIDDEN` | Operation not allowed in current mode |
| -32006 | `NO_ACTIVE_KERNEL` | No GPU kernel active |
| -32601 | `METHOD_NOT_FOUND` | Unknown RPC method |
| -32602 | `INVALID_PARAMS` | Missing or invalid parameters |

## Testing

```bash
# Run tests
python3 -m pytest tests/ -v

# Build CUDA crash test (requires CUDA Toolkit)
cd tests/crash_test && make

# Run specific crash scenario
./cuda_crash_test 1  # Illegal memory access
./cuda_crash_test 4  # Null pointer dereference
```

## Documentation

| File | Content |
|------|---------|
| [`design.md`](design.md) | Complete design specification |
| [`docs/architecture.md`](docs/architecture.md) | Architecture documentation |
| [`skills/cuda-gdb-cli/SKILL.md`](skills/cuda-gdb-cli/SKILL.md) | Claude Code Skill definition |

## Requirements

- Python 3.8+
- CUDA Toolkit 11.0+ (includes `cuda-gdb`)
- NVIDIA GPU with CUDA support
- GDB 9.0+ with Python support

## License

MIT

---

# 中文文档

## 概述

基于 [gdb-cli](https://github.com/Cerdore/gdb-cli) fork 扩展的 CUDA GPU 调试工具，让 AI Agent 通过 bash 命令调试 CUDA 程序。

## 安装

```bash
pip install cuda-gdb-cli
```

## 命令列表

### 会话管理

| 命令 | 说明 | 示例 |
|------|------|------|
| `load` | 加载 coredump | `cuda-gdb-cli load --binary ./app --core ./core.12345` |
| `attach` | 附加到进程 | `cuda-gdb-cli attach --pid 9876` |
| `stop` | 结束会话 | `cuda-gdb-cli stop -s $SESSION` |

### CUDA 调试命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `cuda-kernels` | 列出活跃 kernel | `cuda-gdb-cli cuda-kernels -s $SESSION` |
| `cuda-threads` | 列出 GPU 线程 | `cuda-gdb-cli cuda-threads -s $SESSION --kernel 0` |
| `cuda-focus` | 查看/切换焦点 | `cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"` |
| `cuda-devices` | GPU 设备信息 | `cuda-gdb-cli cuda-devices -s $SESSION` |
| `cuda-exceptions` | CUDA 异常信息 | `cuda-gdb-cli cuda-exceptions -s $SESSION` |
| `cuda-memory` | 读取 GPU 内存 | `cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "smem" --type float --count 32` |

## 快速开始

```bash
# 加载 coredump
cuda-gdb-cli load --binary ./app --core ./core.12345
SESSION="f465d650"

# 查看异常
cuda-gdb-cli cuda-exceptions -s $SESSION

# 切换到异常线程
cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"

# 查看调用栈和局部变量
cuda-gdb-cli bt -s $SESSION
cuda-gdb-cli locals-cmd -s $SESSION

# 检查共享内存
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "tile_A" --type float --count 16

# 结束
cuda-gdb-cli stop -s $SESSION
```

## 输出格式

所有命令返回结构化 JSON：

```json
{
  "cuda_exceptions": [
    {
      "kernel": 0,
      "block": [0, 0, 0],
      "thread": [1, 0, 0],
      "type": "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS",
      "description": "GPU thread accessed illegal memory"
    }
  ]
}
```

## 文档

| 文件 | 内容 |
|------|------|
| [`design.md`](design.md) | 完整设计方案 |
| [`docs/architecture.md`](docs/architecture.md) | 架构文档 |

## 系统要求

- Python 3.8+
- CUDA Toolkit 11.0+
- NVIDIA GPU
- GDB 9.0+ (支持 Python)