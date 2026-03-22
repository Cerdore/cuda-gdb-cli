# CUDA-GDB CLI for AI

[![PyPI version](https://badge.fury.io/py/cuda-gdb-cli.svg)](https://badge.fury.io/py/cuda-gdb-cli)
[![Python versions](https://img.shields.io/pypi/pyversions/cuda-gdb-cli.svg)](https://pypi.org/project/cuda-gdb-cli/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Languages**: [English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

A CUDA GPU debugging tool designed for AI Agents. Forked from [gdb-cli](https://github.com/Cerdore/gdb-cli) with CUDA-specific extensions.

Uses thin client CLI + GDB built-in Python RPC Server architecture. Enables stateful CUDA-GDB debugging through Bash commands with structured JSON output.

### Features

- **CUDA Core Dump Analysis**: Load CUDA core dumps with GPU thread state resident
- **Live Attach Debugging**: Attach to running CUDA processes with non-stop mode
- **GPU Thread Inspection**: List kernels, threads, warps, lanes with coordinates
- **CUDA Memory Spaces**: Read from @shared, @global, @local, @generic address spaces
- **Exception Analysis**: Identify CUDA_EXCEPTION_* types with thread coordinates
- **Structured JSON Output**: All commands output JSON for reliable Agent parsing
- **Security Mechanisms**: Command whitelist, heartbeat timeout, modality guard

### Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.8+ |
| CUDA Toolkit | 11.0+ (includes `cuda-gdb`) |
| GDB | 9.0+ with Python support |
| OS | Linux |

**Check CUDA-GDB Python Support:**
```bash
cuda-gdb -nx -q -batch -ex "python print('OK')"
```

**Check GPU Availability:**
```bash
nvidia-smi
```

### Installation

```bash
pip install cuda-gdb-cli
```

Or install from GitHub:
```bash
pip install git+https://github.com/Cerdore/cuda-gdb-cli.git
```

Run environment check:
```bash
cuda-gdb-cli env-check
```

### Quick Start

#### 1. Load CUDA Core Dump

```bash
cuda-gdb-cli load --binary ./my_program --core ./core.12345
```

Output:
```json
{
  "session_id": "f465d650",
  "mode": "core",
  "binary": "./my_program",
  "core": "./core.12345",
  "status": "started"
}
```

```bash
SESSION="f465d650"
```

#### 2. CUDA Debugging Operations

All operations use `--session` / `-s` to specify the session ID.

```bash
# GPU overview
cuda-gdb-cli cuda-devices -s $SESSION
cuda-gdb-cli cuda-kernels -s $SESSION
cuda-gdb-cli cuda-threads -s $SESSION --limit 20

# Find exceptions
cuda-gdb-cli cuda-exceptions -s $SESSION

# Switch to specific GPU thread
cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"

# CPU-side operations (inherited from gdb-cli)
cuda-gdb-cli bt -s $SESSION
cuda-gdb-cli threads -s $SESSION
cuda-gdb-cli eval-cmd -s $SESSION "my_var"
cuda-gdb-cli locals-cmd -s $SESSION

# Read GPU memory spaces
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "tile_A" --type float --count 16
cuda-gdb-cli cuda-memory -s $SESSION --space global --expr "d_output" --type int --count 10
```

#### 3. Session Management

```bash
# List active sessions
cuda-gdb-cli sessions

# End session
cuda-gdb-cli stop -s $SESSION
```

#### 4. Live Attach Debugging

```bash
cuda-gdb-cli attach --pid 9876 --binary ./my_program
```

### Full Command Reference

#### load — Load CUDA Core Dump

```bash
cuda-gdb-cli load --binary <path> --core <path> [options]
```

| Option | Description |
|--------|-------------|
| `--binary` / `-b` | Path to binary file (required) |
| `--core` / `-c` | Path to core dump file (required) |
| `--gdb-path` | Path to cuda-gdb executable (default: `cuda-gdb`) |
| `--sysroot` | Sysroot for library paths |
| `--solib-prefix` | Shared library prefix |
| `--source-dir` | Source directory for debugging |
| `--timeout` | Session timeout in seconds (default: 600) |

#### attach — Attach to CUDA Process

```bash
cuda-gdb-cli attach --pid <pid> [options]
```

| Option | Description |
|--------|-------------|
| `--pid` / `-p` | Process ID to attach (required) |
| `--binary` / `-b` | Binary file for symbols |
| `--scheduler-locking` | Enable scheduler locking |
| `--non-stop` | Use non-stop mode |
| `--allow-write` | Allow memory writes |
| `--timeout` | Session timeout |

#### cuda-kernels — List Active Kernels

```bash
cuda-gdb-cli cuda-kernels -s <session>
```

Output:
```json
{
  "cuda_kernels": [
    {
      "id": 0,
      "function": "matmul_kernel",
      "grid_dim": [32, 32, 1],
      "block_dim": [16, 16, 1],
      "device": 0,
      "status": "running"
    }
  ],
  "total_count": 1
}
```

#### cuda-threads — List GPU Threads

```bash
cuda-gdb-cli cuda-threads -s <session> [options]
```

| Option | Description |
|--------|-------------|
| `--kernel` / `-k` | Filter by kernel ID |
| `--block` | Filter by block coordinates (e.g., "0,0,0") |
| `--limit` | Max results (default: 50) |

#### cuda-focus — View/Switch GPU Focus

```bash
cuda-gdb-cli cuda-focus -s <session> [options]
```

| Option | Description |
|--------|-------------|
| `--kernel` / `-k` | Kernel ID |
| `--block` | Block coordinates (e.g., "0,0,0") |
| `--thread` | Thread coordinates (e.g., "1,0,0") |

#### cuda-devices — Show GPU Topology

```bash
cuda-gdb-cli cuda-devices -s <session>
```

#### cuda-exceptions — Show CUDA Exceptions

```bash
cuda-gdb-cli cuda-exceptions -s <session>
```

Output:
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
  ],
  "total_count": 1
}
```

#### cuda-memory — Read GPU Memory Space

```bash
cuda-gdb-cli cuda-memory -s <session> --space <space> --expr <expr> [options]
```

| Option | Description |
|--------|-------------|
| `--space` | Address space: shared, global, local, generic (required) |
| `--expr` | Variable or expression (required) |
| `--type` | Element type: int, float, double, etc. (default: int) |
| `--count` | Number of elements (default: 10) |

#### cuda-warps — Show Warp Status

```bash
cuda-gdb-cli cuda-warps -s <session> [options]
```

| Option | Description |
|--------|-------------|
| `--sm` | Filter by SM ID |

#### cuda-lanes — Show Lane Status

```bash
cuda-gdb-cli cuda-lanes -s <session> [options]
```

| Option | Description |
|--------|-------------|
| `--warp` | Filter by warp ID |

### CPU Commands (inherited from gdb-cli)

| Command | Description |
|---------|-------------|
| `bt` | Backtrace |
| `threads` | List CPU threads |
| `eval-cmd` | Evaluate expression |
| `locals-cmd` | Show local variables |
| `memory` | Read memory |
| `disasm` | Disassemble |
| `exec` | Execute raw GDB command |

### Output Format

All commands return structured JSON for reliable Agent parsing:

**Example - cuda-threads:**
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
  "truncated": true,
  "hint": "Use '--kernel K --block x,y,z' to narrow down"
}
```

### CUDA Address Spaces

| Space | Description |
|-------|-------------|
| `shared` | Per-block shared memory, fast, limited size |
| `global` | Device memory, visible to all threads |
| `local` | Per-thread private memory |
| `generic` | Default address space |

### CUDA Exception Types

| Exception | Description |
|-----------|-------------|
| `CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS` | Thread accessed invalid memory |
| `CUDA_EXCEPTION_WARP_ASSERT` | GPU `assert()` triggered |
| `CUDA_EXCEPTION_LANE_INVALID_PC` | Invalid program counter |
| `CUDA_EXCEPTION_WARP_MISALIGNED` | Misaligned memory access |
| `CUDA_EXCEPTION_HARDWARE_STACK_OVERFLOW` | Thread stack overflow |

### Security Mechanisms

#### Modality Guard (FSM)

| State | Mode | Permissions |
|-------|------|-------------|
| `IMMUTABLE` | Core dump | Read-only |
| `MUTABLE` | Live attach | Full access |
| `RUNNING` | Target executing | Limited |
| `STOPPED` | Target paused | Full access |

#### Command Whitelist (Attach Mode)

| Level | Allowed Commands |
|-------|------------------|
| `readonly` | bt, info, print, threads, locals, cuda-* |
| `readwrite` | + set variable, cuda focus |
| `full` | + continue, step, next |

Blocked: `quit`, `kill`, `shell`, `signal`

#### Heartbeat Timeout

Automatically detaches after 10 minutes of inactivity. Configurable via `--timeout`.

#### Idempotency

One session per PID / Core file allowed. Repeated load/attach returns existing session_id.

### Development

#### Project Structure

```
src/
├── cli.py              # Click CLI commands
├── client.py           # Unix Socket client
├── launcher.py         # cuda-gdb process launcher
├── session.py          # Session management
├── safety.py           # Command whitelist
└── gdb_server/
    ├── cuda_handlers.py    # 8 CUDA handlers
    ├── gdb_rpc_server.py   # JSON-RPC server
    ├── gdb_executor.py     # Thread-safe execution
    ├── modality_guard.py   # FSM state machine
    └── value_formatter.py  # gdb.Value → JSON
```

#### Run Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

#### CUDA Crash Test

```bash
cd tests/crash_test
make                    # Build
./cuda_crash_test 1     # Illegal memory access
./cuda_crash_test 4     # Null pointer
```

### Known Limitations

- No `target remote` support
- No multi-GPU debugging (single device per session)
- Requires CUDA Toolkit with cuda-gdb
- GPU must be available for live attach

### Documentation

| File | Content |
|------|---------|
| [`design.md`](design.md) | Complete design specification |
| [`docs/architecture.md`](docs/architecture.md) | Architecture documentation |
| [`skills/cuda-gdb-cli/SKILL.md`](skills/cuda-gdb-cli/SKILL.md) | Claude Code Skill |

### License

Apache License 2.0

---

<a name="中文"></a>
## 中文

面向 AI Agent 的 CUDA GPU 调试工具。基于 [gdb-cli](https://github.com/Cerdore/gdb-cli) fork 扩展，增加 CUDA 调试能力。

采用 thin CLI + GDB 内嵌 Python RPC Server 架构，通过 Bash 命令实现有状态的 CUDA-GDB 调试，输出结构化 JSON。

### 功能特性

- **CUDA Coredump 分析**：加载 CUDA core dump，GPU 线程状态驻留
- **Live Attach 调试**：附加到运行中的 CUDA 进程，支持 non-stop 模式
- **GPU 线程检查**：列出 kernel、thread、warp、lane 及坐标
- **CUDA 内存空间**：读取 @shared、@global、@local、@generic 地址空间
- **异常分析**：识别 CUDA_EXCEPTION_* 类型及线程坐标
- **结构化 JSON 输出**：所有命令输出 JSON，Agent 解析无歧义
- **安全机制**：命令白名单、心跳超时、状态机权限

### 系统要求

| 组件 | 版本要求 |
|------|----------|
| Python | 3.8+ |
| CUDA Toolkit | 11.0+ (包含 `cuda-gdb`) |
| GDB | 9.0+ 且支持 Python |
| 操作系统 | Linux |

**检查 CUDA-GDB Python 支持：**
```bash
cuda-gdb -nx -q -batch -ex "python print('OK')"
```

### 安装

```bash
pip install cuda-gdb-cli
```

运行环境检查：
```bash
cuda-gdb-cli env-check
```

### 快速开始

#### 1. 加载 CUDA Coredump

```bash
cuda-gdb-cli load --binary ./my_program --core ./core.12345
```

输出：
```json
{"session_id": "f465d650", "mode": "core", "status": "started"}
```

```bash
SESSION="f465d650"
```

#### 2. CUDA 调试操作

```bash
# GPU 概览
cuda-gdb-cli cuda-devices -s $SESSION
cuda-gdb-cli cuda-kernels -s $SESSION
cuda-gdb-cli cuda-threads -s $SESSION --limit 20

# 查找异常
cuda-gdb-cli cuda-exceptions -s $SESSION

# 切换到指定 GPU 线程
cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"

# CPU 端操作（继承自 gdb-cli）
cuda-gdb-cli bt -s $SESSION
cuda-gdb-cli eval-cmd -s $SESSION "my_var"
cuda-gdb-cli locals-cmd -s $SESSION

# 读取 GPU 内存空间
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "tile_A" --type float --count 16
```

#### 3. 结束会话

```bash
cuda-gdb-cli stop -s $SESSION
```

### CUDA 命令列表

| 命令 | 说明 |
|------|------|
| `cuda-kernels` | 列出活跃的 kernel |
| `cuda-threads` | 列出 GPU 线程（支持过滤） |
| `cuda-focus` | 查看/切换 GPU 焦点 |
| `cuda-devices` | 显示 GPU 设备拓扑 |
| `cuda-exceptions` | 显示 CUDA 异常 |
| `cuda-memory` | 读取 GPU 内存空间 |
| `cuda-warps` | 显示 warp 状态 |
| `cuda-lanes` | 显示 lane 状态 |

### CUDA 地址空间

| 空间 | 说明 |
|------|------|
| `shared` | Block 内共享内存，快速，容量有限 |
| `global` | 设备内存，所有线程可见 |
| `local` | 线程私有内存 |
| `generic` | 默认地址空间 |

### CUDA 异常类型

| 异常 | 说明 |
|------|------|
| `CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS` | 线程访问非法内存 |
| `CUDA_EXCEPTION_WARP_ASSERT` | GPU assert() 触发 |
| `CUDA_EXCEPTION_LANE_INVALID_PC` | 无效程序计数器 |
| `CUDA_EXCEPTION_WARP_MISALIGNED` | 内存访问未对齐 |

### 项目结构

```
src/gdb_server/
├── cuda_handlers.py    # 8 个 CUDA handler
├── gdb_rpc_server.py   # JSON-RPC 服务器
├── gdb_executor.py     # 线程安全执行
├── modality_guard.py   # 状态机
└── value_formatter.py  # gdb.Value → JSON 序列化
```

### 文档

| 文件 | 内容 |
|------|------|
| [`design.md`](design.md) | 完整设计方案 |
| [`docs/architecture.md`](docs/architecture.md) | 架构文档 |

### 许可证

Apache License 2.0