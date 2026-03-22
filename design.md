# CUDA-GDB CLI — 基于 gdb-cli 扩展的 CUDA 调试工具

## 0. 设计策略：Fork gdb-cli，扩展 CUDA 支持

### 0.1 为什么 fork 而不是从头写

[gdb-cli](https://github.com/Cerdore/gdb-cli) 是一个已经成熟的、面向 AI Agent 的 GDB 调试 CLI 工具。它的架构经过验证，核心能力完备：

| gdb-cli 已有能力                         | 说明                                                          |
| ---------------------------------------- | ------------------------------------------------------------- |
| **thin CLI + GDB 内嵌 Python RPC** | 比 pexpect 可靠 10 倍，直接用 GDB Python API                  |
| **结构化 JSON 输出**               | `gdb.Value` → JSON 递归序列化，Agent 解析零歧义            |
| **会话管理**                       | Unix Socket + named FIFO 保活，幂等性，心跳超时自动清理       |
| **安全白名单**                     | readonly / readwrite / full 三级，attach 生产进程时的安全保障 |
| **16 个语义化 handler**            | bt, threads, eval, locals, memory, disasm, ptype 等           |
| **Claude Code Skill**              | 自带 SKILL.md，`bunx skills add` 一键安装                   |

**我们需要做的只是 CUDA 差异化的 20%**，而不是重新造 80% 的轮子。

### 0.2 CUDA 扩展的核心差异

`cuda-gdb` 是 `gdb` 的超集。所有 GDB 命令在 `cuda-gdb` 中都可用，但 CUDA 额外引入了以下维度：

| 维度               | GDB (CPU)          | CUDA-GDB (GPU)                                        |
| ------------------ | ------------------ | ----------------------------------------------------- |
| **线程模型** | OS 线程（pthread） | GPU 线程：Grid → Block → Thread（三维坐标）         |
| **硬件坐标** | 无                 | Device → SM → Warp → Lane                          |
| **焦点切换** | `thread N`       | `cuda kernel K block (x,y,z) thread (x,y,z)`        |
| **内存空间** | 统一地址空间       | `@global` / `@shared` / `@local` / `@generic` |
| **异常类型** | SIGSEGV, SIGFPE 等 | CUDA_EXCEPTION_WARP_ASSERT, LANE_ILLEGAL_ADDRESS 等   |
| **执行单元** | 进程               | Kernel（<<<grid, block>>>）                           |
| **寄存器**   | rax, rbx, rip 等   | GPU 寄存器（R0-R255），数量因架构而异                 |

### 0.3 改动范围总览

```
gdb-cli (upstream)
├── src/gdb_cli/
│   ├── cli.py              ← 新增 CUDA 子命令
│   ├── client.py           ← 无需改动（通用 Unix Socket 客户端）
│   ├── launcher.py         ← gdb → cuda-gdb，新增 CUDA 启动参数
│   ├── session.py          ← 新增 CUDA 元数据字段
│   ├── safety.py           ← 新增 CUDA 命令白名单
│   ├── formatters.py       ← 无需改动
│   ├── errors.py           ← 新增 CUDA 错误类型
│   ├── env_check.py        ← 新增 CUDA 环境检查
│   └── gdb_server/
│       ├── gdb_rpc_server.py   ← 新增 CUDA handler 注册
│       ├── handlers.py         ← 新增 8 个 CUDA handler
│       ├── value_formatter.py  ← 扩展 @shared/@global 地址空间
│       └── heartbeat.py        ← 无需改动
└── skills/
    └── cuda-gdb-cli/           ← 新增 CUDA 调试 Skill
        └── SKILL.md
```

---

## 1. 架构：继承 gdb-cli 的 thin CLI + 内嵌 RPC

### 1.1 整体架构（与 gdb-cli 一致）

```
┌─────────────────────────────────────────────────────────┐
│  AI Agent (Claude Code / Codex CLI / Cursor / ...)      │
│  工具：bash shell                                        │
└──────────────────────┬──────────────────────────────────┘
                       │  bash 命令
                       ▼
┌─────────────────────────────────────────────────────────┐
│  cuda-gdb-cli (Python CLI, Click)                       │
│                                                         │
│  • 解析命令行参数                                        │
│  • 通过 Unix Socket 连接 RPC Server                      │
│  • 输出结构化 JSON                                       │
└──────────────────────┬──────────────────────────────────┘
                       │  Unix Domain Socket (JSON)
                       ▼
┌─────────────────────────────────────────────────────────┐
│  cuda-gdb 进程 (常驻后台)                                │
│                                                         │
│  ┌───────────────────────────────────────────────┐      │
│  │  GDB Python RPC Server (内嵌)                  │      │
│  │                                               │      │
│  │  后台线程: Unix Socket accept/recv/send        │      │
│  │  主线程:   gdb.execute() / gdb.parse_and_eval()│      │
│  │                                               │      │
│  │  Handlers:                                    │      │
│  │    ├── CPU: bt, threads, eval, locals, ...    │      │
│  │    └── CUDA: cuda_threads, cuda_kernels,      │      │
│  │             cuda_focus, cuda_memory, ...       │      │
│  └───────────────────────────────────────────────┘      │
│                                                         │
│  NVIDIA cuda-gdb (GDB 超集)                              │
│  支持 GPU 线程、共享内存、Warp 异常等                     │
└─────────────────────────────────────────────────────────┘
```

### 1.2 关键技术决策（继承自 gdb-cli）

| 决策                   | 方案                                                          | 原因                                |
| ---------------------- | ------------------------------------------------------------- | ----------------------------------- |
| **GDB 交互方式** | GDB Python API（`gdb.execute()`, `gdb.parse_and_eval()`） | 比 pexpect 可靠，无提示符误匹配问题 |
| **进程间通信**   | Unix Domain Socket                                            | 简单可靠，无需 HTTP/TCP 开销        |
| **进程保活**     | named FIFO 作为 stdin                                         | GDB 阻塞在读端，不退出              |
| **输出格式**     | 始终 JSON                                                     | Agent 解析准确率高                  |
| **会话管理**     | 文件系统持久化 + 幂等性                                       | 同 PID/Core 复用 session            |
| **安全机制**     | 命令白名单 + 心跳超时                                         | 防止误操作和进程泄漏                |

---

## 2. 新增 CUDA 子命令设计

### 2.1 命令总览

在 gdb-cli 原有命令基础上，新增以下 CUDA 特有子命令：

| 命令                | 用途                                            | 对应 cuda-gdb 命令            |
| ------------------- | ----------------------------------------------- | ----------------------------- |
| `cuda-threads`    | 列出 GPU 线程（按 Block/Thread 坐标）           | `info cuda threads`         |
| `cuda-kernels`    | 列出活跃 Kernel                                 | `info cuda kernels`         |
| `cuda-focus`      | 查看/切换 GPU 焦点（kernel/block/thread）       | `cuda kernel/block/thread`  |
| `cuda-devices`    | 查看 GPU 设备拓扑                               | `info cuda devices`         |
| `cuda-exceptions` | 查看 CUDA 异常                                  | `info cuda exceptions`      |
| `cuda-memory`     | 读取 GPU 地址空间内存（@shared/@global/@local） | `print @space type[N] expr` |
| `cuda-warps`      | 查看 Warp 级状态                                | `info cuda warps`           |
| `cuda-lanes`      | 查看 Lane 级状态                                | `info cuda lanes`           |

**原有 gdb-cli 命令全部保留**（`load`, `attach`, `bt`, `threads`, `eval-cmd`, `locals-cmd`, `exec`, `memory`, `disasm` 等），它们在 `cuda-gdb` 中同样可用，用于 CPU 端调试。

### 2.2 `cuda-threads` — GPU 线程列表

```bash
cuda-gdb-cli cuda-threads -s $SESSION [OPTIONS]

OPTIONS:
  --kernel K           按 Kernel 过滤
  --block "x,y,z"      按 Block 过滤
  --limit N            最大返回数量（默认 50）
  --range "START-END"  线程范围
```

**输出示例：**

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
    },
    {
      "kernel": 0,
      "block_idx": [0, 0, 0],
      "thread_idx": [1, 0, 0],
      "name": "matmul_kernel",
      "file": "matmul.cu",
      "line": 28,
      "is_current": false,
      "exception": "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS"
    }
  ],
  "total_count": 16384,
  "truncated": true,
  "hint": "use '--kernel 0 --block 0,0,0' to narrow down"
}
```

### 2.3 `cuda-kernels` — 活跃 Kernel 列表

```bash
cuda-gdb-cli cuda-kernels -s $SESSION
```

**输出示例：**

```json
{
  "kernels": [
    {
      "kernel_id": 0,
      "function": "matmul_kernel",
      "grid_dim": [32, 32, 1],
      "block_dim": [16, 16, 1],
      "device": 0,
      "status": "running",
      "invocation": "matmul_kernel<<<(32,32,1),(16,16,1)>>>"
    }
  ],
  "count": 1
}
```

### 2.4 `cuda-focus` — GPU 焦点切换

```bash
# 查看当前焦点
cuda-gdb-cli cuda-focus -s $SESSION

# 切换到指定 GPU 线程
cuda-gdb-cli cuda-focus -s $SESSION --kernel 0 --block "0,0,0" --thread "1,0,0"

# 只切换 Block
cuda-gdb-cli cuda-focus -s $SESSION --block "2,3,0"

# 只切换 Thread
cuda-gdb-cli cuda-focus -s $SESSION --thread "15,0,0"
```

**输出示例：**

```json
{
  "focus": {
    "kernel": 0,
    "block_idx": [0, 0, 0],
    "thread_idx": [1, 0, 0],
    "device": 0,
    "sm": 7,
    "warp": 0,
    "lane": 1
  },
  "frame": {
    "function": "matmul_kernel",
    "file": "matmul.cu",
    "line": 28,
    "address": "0x7f4a3c001234"
  }
}
```

### 2.5 `cuda-devices` — GPU 设备拓扑

```bash
cuda-gdb-cli cuda-devices -s $SESSION
```

**输出示例：**

```json
{
  "devices": [
    {
      "device_id": 0,
      "name": "NVIDIA A100-SXM4-80GB",
      "sm_count": 108,
      "compute_capability": "8.0",
      "is_current": true
    }
  ],
  "count": 1
}
```

### 2.6 `cuda-exceptions` — CUDA 异常信息

```bash
cuda-gdb-cli cuda-exceptions -s $SESSION
```

**输出示例：**

```json
{
  "exceptions": [
    {
      "kernel": 0,
      "block_idx": [0, 0, 0],
      "thread_idx": [1, 0, 0],
      "exception": "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS",
      "description": "Illegal address accessed by GPU thread",
      "error_pc": "0x7f4a3c001234"
    }
  ],
  "count": 1
}
```

### 2.7 `cuda-memory` — GPU 地址空间内存读取

```bash
# 读取共享内存
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "shared_data" --type "float" --count 32

# 读取全局内存
cuda-gdb-cli cuda-memory -s $SESSION --space global --expr "d_array" --type "int" --count 10

# 读取局部内存
cuda-gdb-cli cuda-memory -s $SESSION --space local --expr "local_buf" --type "double" --count 4
```

**输出示例：**

```json
{
  "space": "shared",
  "expression": "shared_data",
  "element_type": "float",
  "count": 32,
  "elements": [1.0, 2.0, 3.0, 0.0, 0.0, "..."],
  "truncated": false,
  "address": "0x7fff00001000"
}
```

### 2.8 `cuda-warps` / `cuda-lanes` — Warp/Lane 级状态

```bash
cuda-gdb-cli cuda-warps -s $SESSION [--sm N]
cuda-gdb-cli cuda-lanes -s $SESSION [--warp N]
```

**输出示例（warps）：**

```json
{
  "warps": [
    {
      "warp_id": 0,
      "sm": 7,
      "block_idx": [0, 0, 0],
      "status": "breakpoint",
      "active_mask": "0xffffffff",
      "divergent": false
    }
  ],
  "count": 4
}
```

---

## 3. Handler 实现设计

### 3.1 CUDA Handler 注册

在 `gdb_rpc_server.py` 的 `_register_builtin_handlers()` 中新增 CUDA handler：

```python
def _register_builtin_handlers(self) -> None:
    # ... 原有 handler 注册 ...

    # CUDA 特有 handler
    self._handlers.update({
        "cuda_threads":    cuda_handlers.handle_cuda_threads,
        "cuda_kernels":    cuda_handlers.handle_cuda_kernels,
        "cuda_focus":      cuda_handlers.handle_cuda_focus,
        "cuda_devices":    cuda_handlers.handle_cuda_devices,
        "cuda_exceptions": cuda_handlers.handle_cuda_exceptions,
        "cuda_memory":     cuda_handlers.handle_cuda_memory,
        "cuda_warps":      cuda_handlers.handle_cuda_warps,
        "cuda_lanes":      cuda_handlers.handle_cuda_lanes,
    })
```

### 3.2 核心 Handler 实现

#### `handle_cuda_threads` — 解析 `info cuda threads` 输出

`cuda-gdb` 的 Python API 没有直接暴露 CUDA 线程信息的结构化接口，需要通过 `gdb.execute("info cuda threads", to_string=True)` 获取文本输出后解析：

```python
def handle_cuda_threads(
    kernel: int = None,
    block: str = None,
    limit: int = 50,
    range_str: str = None,
    **kwargs
) -> dict:
    """
    列出 CUDA GPU 线程

    实现策略：
    1. 执行 `info cuda threads` 获取原始文本
    2. 正则解析每行的 BlockIdx, ThreadIdx, Kernel, File, Line 等字段
    3. 按 kernel/block 过滤
    4. 截断并返回 JSON
    """
    output = gdb.execute("info cuda threads", to_string=True)
    threads = _parse_cuda_threads_output(output)

    # 过滤
    if kernel is not None:
        threads = [t for t in threads if t["kernel"] == kernel]
    if block is not None:
        block_idx = _parse_coord(block)
        threads = [t for t in threads if t["block_idx"] == block_idx]

    total_count = len(threads)
    truncated = total_count > limit
    display_threads = threads[:limit]

    result = {
        "cuda_threads": display_threads,
        "total_count": total_count,
        "truncated": truncated
    }
    if truncated:
        result["hint"] = "use '--kernel K --block x,y,z' to narrow down"
    return result


# info cuda threads 输出格式示例：
#   BlockIdx  ThreadIdx  To  Name           Filename   Line
# * (0,0,0)  (0,0,0)    -   matmul_kernel  matmul.cu  28
#   (0,0,0)  (1,0,0)    -   matmul_kernel  matmul.cu  28

CUDA_THREAD_PATTERN = re.compile(
    r'(?P<current>\*?)\s*'
    r'\((?P<bx>\d+),(?P<by>\d+),(?P<bz>\d+)\)\s+'
    r'\((?P<tx>\d+),(?P<ty>\d+),(?P<tz>\d+)\)\s+'
    r'(?P<to>\S+)\s+'
    r'(?P<name>\S+)\s+'
    r'(?P<file>\S+)\s+'
    r'(?P<line>\d+)'
)

def _parse_cuda_threads_output(output: str) -> list:
    threads = []
    for line in output.strip().split('\n'):
        match = CUDA_THREAD_PATTERN.search(line)
        if match:
            threads.append({
                "kernel": 0,  # 从上下文推断
                "block_idx": [int(match.group("bx")), int(match.group("by")), int(match.group("bz"))],
                "thread_idx": [int(match.group("tx")), int(match.group("ty")), int(match.group("tz"))],
                "name": match.group("name"),
                "file": match.group("file"),
                "line": int(match.group("line")),
                "is_current": match.group("current") == "*",
            })
    return threads
```

#### `handle_cuda_focus` — GPU 焦点切换

```python
def handle_cuda_focus(
    kernel: int = None,
    block: str = None,
    thread: str = None,
    **kwargs
) -> dict:
    """
    查看或切换 CUDA GPU 焦点

    实现策略：
    1. 如果提供了 kernel/block/thread 参数，构造 `cuda` 命令执行切换
    2. 执行 `cuda kernel`、`cuda block`、`cuda thread` 获取当前焦点
    3. 返回结构化的焦点信息 + 当前帧信息
    """
    # 执行切换
    if kernel is not None:
        gdb.execute(f"cuda kernel {kernel}", to_string=True)
    if block is not None:
        coord = _parse_coord(block)
        gdb.execute(f"cuda block ({coord[0]},{coord[1]},{coord[2]})", to_string=True)
    if thread is not None:
        coord = _parse_coord(thread)
        gdb.execute(f"cuda thread ({coord[0]},{coord[1]},{coord[2]})", to_string=True)

    # 获取当前焦点
    focus = _get_current_cuda_focus()

    # 获取当前帧
    frame = gdb.selected_frame()
    frame_info = {
        "function": frame.name() or "??",
        "address": hex(frame.pc())
    }
    try:
        sal = frame.sal()
        if sal and sal.symtab:
            frame_info["file"] = sal.symtab.filename
            frame_info["line"] = sal.line
    except Exception:
        pass

    return {"focus": focus, "frame": frame_info}


def _get_current_cuda_focus() -> dict:
    """解析当前 CUDA 焦点信息"""
    focus = {}

    # 获取软件坐标
    for dimension in ["kernel", "block", "thread"]:
        try:
            output = gdb.execute(f"cuda {dimension}", to_string=True)
            focus[dimension] = _parse_focus_output(dimension, output)
        except gdb.error:
            pass

    # 获取硬件坐标
    for dimension in ["device", "sm", "warp", "lane"]:
        try:
            output = gdb.execute(f"cuda {dimension}", to_string=True)
            focus[dimension] = _parse_focus_output(dimension, output)
        except gdb.error:
            pass

    return focus
```

#### `handle_cuda_memory` — GPU 地址空间内存读取

```python
def handle_cuda_memory(
    space: str,
    expr: str,
    element_type: str = "int",
    count: int = 10,
    max_elements: int = 100,
    **kwargs
) -> dict:
    """
    读取 GPU 特定地址空间的内存

    实现策略：
    使用 cuda-gdb 的 @space 修饰符：
      print @shared float[32] shared_data
      print @global int[10] d_array

    Args:
        space: "shared" | "global" | "local" | "generic"
        expr: 变量名或地址表达式
        element_type: C 类型名（int, float, double 等）
        count: 元素数量
    """
    valid_spaces = {"shared", "global", "local", "generic"}
    if space not in valid_spaces:
        return {"error": f"Invalid space '{space}', must be one of {valid_spaces}"}

    actual_count = min(count, max_elements)
    gdb_expr = f"@{space} {element_type}[{actual_count}] {expr}"

    try:
        val = gdb.parse_and_eval(gdb_expr)
        elements = []
        for i in range(actual_count):
            try:
                elem = val[i]
                elements.append(format_gdb_value(elem, max_depth=1))
            except gdb.MemoryError:
                elements.append({"error": f"Cannot access element [{i}]"})
                break

        result = {
            "space": space,
            "expression": expr,
            "element_type": element_type,
            "count": len(elements),
            "elements": elements,
            "truncated": count > max_elements,
        }

        try:
            if val.address:
                result["address"] = hex(int(val.address))
        except Exception:
            pass

        return result

    except gdb.error as e:
        return {"error": f"Cannot read @{space} memory: {e}"}
```

#### `handle_cuda_exceptions` — CUDA 异常解析

```python
def handle_cuda_exceptions(**kwargs) -> dict:
    """
    获取 CUDA 异常信息

    实现策略：
    1. 执行 `info cuda exceptions` 获取文本
    2. 正则解析异常类型、位置、线程坐标
    3. 对于每个异常，尝试获取 $errorpc 反汇编
    """
    output = gdb.execute("info cuda exceptions", to_string=True)
    exceptions = _parse_cuda_exceptions_output(output)

    # 对每个异常尝试获取更多上下文
    for exc in exceptions:
        if exc.get("error_pc"):
            try:
                disasm = gdb.execute(
                    f"x/3i {exc['error_pc']}", to_string=True
                )
                exc["disassembly"] = disasm.strip()
            except Exception:
                pass

    return {"exceptions": exceptions, "count": len(exceptions)}


# CUDA 异常类型映射
CUDA_EXCEPTION_MAP = {
    "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS": "GPU 线程访问了非法内存地址",
    "CUDA_EXCEPTION_LANE_USER_STACK_OVERFLOW": "GPU 线程栈溢出",
    "CUDA_EXCEPTION_DEVICE_HARDWARE_STACK_OVERFLOW": "GPU 硬件栈溢出",
    "CUDA_EXCEPTION_WARP_ILLEGAL_INSTRUCTION": "Warp 执行了非法指令",
    "CUDA_EXCEPTION_WARP_OUT_OF_RANGE_ADDRESS": "Warp 访问了越界地址",
    "CUDA_EXCEPTION_WARP_MISALIGNED_ADDRESS": "Warp 访问了未对齐地址",
    "CUDA_EXCEPTION_WARP_INVALID_ADDRESS_SPACE": "Warp 访问了无效地址空间",
    "CUDA_EXCEPTION_WARP_INVALID_PC": "Warp 跳转到无效 PC",
    "CUDA_EXCEPTION_WARP_HARDWARE_STACK_OVERFLOW": "Warp 硬件栈溢出",
    "CUDA_EXCEPTION_DEVICE_ILLEGAL_ADDRESS": "Device 级非法地址访问",
    "CUDA_EXCEPTION_WARP_ASSERT": "GPU 端 assert() 触发",
}
```

### 3.3 GPU 寄存器处理

GPU 寄存器与 CPU 不同，需要特殊处理：

```python
# 扩展 handlers.py 中的 handle_registers

# GPU 寄存器没有固定名称列表，需要先探测
# 策略：先执行 `info registers` 获取可用寄存器列表，再逐个读取

def _get_cuda_registers(frame) -> list:
    """
    获取 CUDA GPU 寄存器

    cuda-gdb 的 GPU 寄存器命名为 R0, R1, ..., R255
    数量因 GPU 架构和 kernel 编译选项而异
    安全策略：先 `info registers` 获取列表，再逐个读取
    """
    try:
        output = gdb.execute("info registers", to_string=True)
        regs = []
        for line in output.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                regs.append({
                    "name": parts[0],
                    "value": parts[1],
                })
        return regs
    except gdb.error as e:
        return [{"error": f"Cannot read GPU registers: {e}"}]
```

---

## 4. Launcher 改动

### 4.1 `gdb` → `cuda-gdb` 替换

```python
# launcher.py 改动

def launch_core(
    binary: str,
    core: str,
    sysroot: str = None,
    solib_prefix: str = None,
    source_dir: str = None,
    timeout: int = 600,
    gdb_path: str = "cuda-gdb",      # ← 默认改为 cuda-gdb
    cuda_memcheck: bool = False,      # ← 新增：是否启用 memcheck 集成
) -> GDBProcess:
    """启动 Core 模式 cuda-gdb 进程"""

    gdb_commands = [
        "set pagination off",
        "set print elements 0",
        "set confirm off",
    ]

    # CUDA 特有设置
    if cuda_memcheck:
        gdb_commands.append("set cuda memcheck on")

    # sysroot / solib-prefix（与 gdb-cli 一致）
    if sysroot:
        gdb_commands.append(f"set sysroot {sysroot}")
    if solib_prefix:
        gdb_commands.append(f"set solib-absolute-prefix {solib_prefix}")
    if source_dir:
        gdb_commands.append(f"directory {source_dir}")

    # 加载 binary 和 core
    gdb_commands.append(f"file {binary}")
    gdb_commands.append(f"core-file {core}")

    # 启动 RPC Server（与 gdb-cli 一致）
    gdb_commands.extend(_build_server_commands(session))

    # 构建 cuda-gdb 参数
    gdb_args = [gdb_path, "-nx", "-q"]
    for cmd in gdb_commands:
        gdb_args.extend(["-ex", cmd])

    _start_gdb_process(gdb_args, session)
    # ...


def launch_attach(
    pid: int,
    binary: str = None,
    scheduler_locking: bool = True,
    non_stop: bool = True,
    timeout: int = 600,
    allow_write: bool = False,
    allow_call: bool = False,
    gdb_path: str = "cuda-gdb",      # ← 默认改为 cuda-gdb
    cuda_software_preemption: bool = False,  # ← 新增
) -> GDBProcess:
    """启动 Attach 模式 cuda-gdb 进程"""

    gdb_commands = [
        "set pagination off",
        "set print elements 0",
        "set confirm off",
    ]

    # CUDA 特有设置
    if cuda_software_preemption:
        gdb_commands.append("set cuda software_preemption on")

    # ... 其余与 gdb-cli 一致
```

### 4.2 环境检查扩展

```python
# env_check.py 扩展

def env_check() -> dict:
    results = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
    }

    # 检查 cuda-gdb
    cuda_gdb_path = shutil.which("cuda-gdb")
    if cuda_gdb_path:
        results["cuda_gdb_path"] = cuda_gdb_path
        # 检查 Python 支持
        try:
            output = subprocess.check_output(
                [cuda_gdb_path, "-nx", "-q", "-batch", "-ex", "python print('OK')"],
                text=True, timeout=10
            )
            results["cuda_gdb_python"] = "OK" in output
        except Exception:
            results["cuda_gdb_python"] = False
    else:
        results["cuda_gdb_path"] = None
        results["cuda_gdb_error"] = "cuda-gdb not found. Install CUDA Toolkit."

    # 检查 CUDA 版本
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            output = subprocess.check_output([nvcc_path, "--version"], text=True)
            match = re.search(r"release (\d+\.\d+)", output)
            if match:
                results["cuda_version"] = match.group(1)
        except Exception:
            pass

    # 检查 GPU 驱动
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.check_output(
                [nvidia_smi, "--query-gpu=driver_version,name", "--format=csv,noheader"],
                text=True
            )
            lines = output.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                results["gpu_driver"] = parts[0].strip()
                results["gpu_name"] = parts[1].strip() if len(parts) > 1 else "unknown"
        except Exception:
            pass

    return results
```

---

## 5. Value Formatter 扩展

### 5.1 地址空间修饰符处理

`cuda-gdb` 中的指针可能带有地址空间修饰符（`@shared`, `@global`, `@local`）。需要在 `value_formatter.py` 中扩展指针格式化逻辑：

```python
# value_formatter.py 扩展

# CUDA 地址空间映射
CUDA_ADDRESS_SPACES = {
    "shared":  "@shared",
    "global":  "@global",
    "local":   "@local",
    "generic": "@generic",
}

def _format_pointer(val, depth, max_depth, max_elements, max_string_len, max_fields):
    """格式化指针（扩展 CUDA 地址空间支持）"""
    try:
        addr = int(val)
        target_type = val.type.target()

        result = {
            "type": "pointer",
            "address": hex(addr),
            "target_type": str(target_type)
        }

        # 检测 CUDA 地址空间
        type_str = str(val.type)
        for space_name, space_prefix in CUDA_ADDRESS_SPACES.items():
            if space_prefix in type_str:
                result["address_space"] = space_name
                break

        # ... 其余逻辑与 gdb-cli 一致（NULL 检查、char* 字符串、解引用）

        return result
    except Exception as e:
        return {"type": "pointer", "error": str(e)}
```

---

## 6. Safety 扩展

### 6.1 CUDA 命令白名单

在 `safety.py` 中扩展 CUDA 特有命令的安全分类：

```python
# safety.py 扩展

# CUDA 只读命令（所有安全级别都允许）
CUDA_READONLY_COMMANDS = {
    "info cuda threads",
    "info cuda kernels",
    "info cuda devices",
    "info cuda exceptions",
    "info cuda warps",
    "info cuda lanes",
    "info cuda managed",
    "cuda kernel",        # 查看/切换焦点（只读操作）
    "cuda block",
    "cuda thread",
    "cuda device",
    "cuda sm",
    "cuda warp",
    "cuda lane",
}

# CUDA 写操作命令（需要 readwrite 或 full 级别）
CUDA_WRITE_COMMANDS = {
    "set cuda memcheck",
    "set cuda software_preemption",
    "set cuda break_on_launch",
}

# CUDA 始终禁止的命令
CUDA_BLOCKED_COMMANDS = {
    # 无额外禁止项，继承 gdb-cli 的 quit/kill/shell 禁止
}
```

---

## 7. CLI 命令注册

### 7.1 新增 CUDA 子命令（Click）

```python
# cli.py 扩展

@main.command("cuda-threads")
@click.option("--session", "-s", required=True, help="会话 ID")
@click.option("--kernel", "-k", type=int, help="按 Kernel 过滤")
@click.option("--block", help="按 Block 过滤 (如 '0,0,0')")
@click.option("--limit", default=50, help="最大返回数量")
def cuda_threads_cmd(session, kernel, block, limit):
    """列出 CUDA GPU 线程"""
    with get_client(session) as client:
        params = {"limit": limit}
        if kernel is not None:
            params["kernel"] = kernel
        if block:
            params["block"] = block
        result = client.call("cuda_threads", **params)
        print_json(result)


@main.command("cuda-kernels")
@click.option("--session", "-s", required=True, help="会话 ID")
def cuda_kernels_cmd(session):
    """列出活跃 CUDA Kernel"""
    with get_client(session) as client:
        result = client.call("cuda_kernels")
        print_json(result)


@main.command("cuda-focus")
@click.option("--session", "-s", required=True, help="会话 ID")
@click.option("--kernel", "-k", type=int, help="切换到指定 Kernel")
@click.option("--block", help="切换到指定 Block (如 '2,3,0')")
@click.option("--thread", help="切换到指定 Thread (如 '1,0,0')")
def cuda_focus_cmd(session, kernel, block, thread):
    """查看/切换 CUDA GPU 焦点"""
    with get_client(session) as client:
        params = {}
        if kernel is not None:
            params["kernel"] = kernel
        if block:
            params["block"] = block
        if thread:
            params["thread"] = thread
        result = client.call("cuda_focus", **params)
        print_json(result)


@main.command("cuda-devices")
@click.option("--session", "-s", required=True, help="会话 ID")
def cuda_devices_cmd(session):
    """查看 GPU 设备拓扑"""
    with get_client(session) as client:
        result = client.call("cuda_devices")
        print_json(result)


@main.command("cuda-exceptions")
@click.option("--session", "-s", required=True, help="会话 ID")
def cuda_exceptions_cmd(session):
    """查看 CUDA 异常信息"""
    with get_client(session) as client:
        result = client.call("cuda_exceptions")
        print_json(result)


@main.command("cuda-memory")
@click.option("--session", "-s", required=True, help="会话 ID")
@click.option("--space", required=True,
              type=click.Choice(["shared", "global", "local", "generic"]),
              help="GPU 地址空间")
@click.option("--expr", required=True, help="变量名或地址表达式")
@click.option("--type", "element_type", default="int", help="元素类型 (int/float/double)")
@click.option("--count", default=10, help="元素数量")
def cuda_memory_cmd(session, space, expr, element_type, count):
    """读取 GPU 地址空间内存"""
    with get_client(session) as client:
        result = client.call("cuda_memory",
                             space=space, expr=expr,
                             element_type=element_type, count=count)
        print_json(result)
```

---

## 8. Claude Code Skill 定义

### 8.1 SKILL.md

```markdown
---
name: cuda-gdb-cli
description: |
  CUDA GPU debugging assistant that combines source code analysis with GPU runtime state.
  Use this skill when the user wants to:
  - Analyze CUDA core dumps or GPU crash dumps
  - Debug running CUDA processes
  - Investigate GPU kernel crashes, illegal memory access, warp exceptions
  - Debug shared memory issues, thread divergence, or race conditions
  Requires: cuda-gdb-cli (pip install cuda-gdb-cli) and cuda-gdb (CUDA Toolkit).
---

# CUDA GDB Debug Skill

You are an expert CUDA debugger. Your job is to help users debug CUDA GPU programs
by combining **source code analysis** with **GPU runtime state inspection**.

## Core Principle

CUDA debugging requires THREE kinds of information:
1. **Static Code**: Kernel source, launch parameters, memory allocation
2. **CPU State**: Host-side call stacks, variables, thread states
3. **GPU State**: Kernel threads, shared memory, warp exceptions, device topology

## Workflow

### Step 1: Initialize Debug Session

**For CUDA core dump:**
```bash
cuda-gdb-cli load --binary <binary> --core <core> [--gdb-path cuda-gdb]
```

**For live CUDA process:**

```bash
cuda-gdb-cli attach --pid <pid> [--binary <binary>]
```

### Step 2: Gather GPU Overview

```bash
SESSION="<session_id>"

# GPU 设备信息
cuda-gdb-cli cuda-devices -s $SESSION

# 活跃 Kernel 列表
cuda-gdb-cli cuda-kernels -s $SESSION

# GPU 线程概览
cuda-gdb-cli cuda-threads -s $SESSION --limit 20

# CUDA 异常
cuda-gdb-cli cuda-exceptions -s $SESSION
```

### Step 3: Focus on Crash Point

```bash
# CPU 端调用栈
cuda-gdb-cli bt -s $SESSION

# 切换到异常 GPU 线程
cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"

# 查看该线程的调用栈和局部变量
cuda-gdb-cli bt -s $SESSION
cuda-gdb-cli locals-cmd -s $SESSION
```

### Step 4: Correlate Source Code (CRITICAL)

For each frame in the backtrace:

1. **Read kernel source**: Use `Read` tool to get ±20 lines around crash point
2. **Check launch parameters**: grid_dim, block_dim, shared memory size
3. **Verify index calculations**: threadIdx, blockIdx, blockDim arithmetic
4. **Check memory bounds**: array sizes vs computed indices

### Step 5: Deep GPU Investigation

```bash
# 检查共享内存内容
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "smem" --type float --count 32

# 检查全局内存
cuda-gdb-cli cuda-memory -s $SESSION --space global --expr "d_output" --type int --count 10

# 查看 GPU 寄存器
cuda-gdb-cli registers -s $SESSION

# 反汇编崩溃点
cuda-gdb-cli disasm -s $SESSION --count 10

# 检查多个 GPU 线程的状态
cuda-gdb-cli cuda-threads -s $SESSION --kernel 0 --block "0,0,0"
```

### Step 6: Generate Analysis Report

Structure findings as:

```markdown
## CUDA Debug Session Summary

**Kernel**: `matmul_kernel<<<(32,32,1),(16,16,1)>>>`
**Exception**: `CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS`
**Crash Thread**: Block (0,0,0), Thread (1,0,0)

### Crash Point
`matmul.cu:28` — `C[row * N + col] = sum;`

### Root Cause
Thread (1,0,0) computed `row * N + col = 1024` which exceeds
the allocated size of array C (1024 elements, valid indices 0-1023).

### Evidence
- `row = 0, col = 0, N = 1024, k = 1024`
- The loop `for (k = 0; k <= N; k++)` should be `k < N`
- Off-by-one error causes `k` to reach 1024

### Fix
Change `k <= N` to `k < N` at matmul.cu:25
```

## Common CUDA Debugging Patterns

### Pattern: Illegal Memory Access

**Indicators:** `CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS`
**Investigation:**

1. Check thread index calculations
2. Verify array bounds
3. Check shared memory bank conflicts

### Pattern: Warp Divergence Issues

**Indicators:** Performance problems, unexpected results
**Investigation:**

1. `cuda-gdb-cli cuda-warps` to check warp status
2. Check conditional branches in kernel code

### Pattern: Shared Memory Race Condition

**Indicators:** Non-deterministic results
**Investigation:**

1. Check `__syncthreads()` placement
2. Read shared memory from multiple threads
3. Verify write-before-read ordering

### Pattern: Kernel Launch Failure

**Indicators:** No GPU threads visible
**Investigation:**

1. Check launch parameters (grid/block dimensions)
2. Verify CUDA API return codes on host side
3. Check GPU memory allocation

```

---

## 9. 项目结构（fork 后）

```

cuda-gdb-cli/                          # fork from Cerdore/gdb-cli
├── pyproject.toml                     # 改名 gdb-cli → cuda-gdb-cli
├── README.md
├── src/
│   └── cuda_gdb_cli/                  # 改名 gdb_cli → cuda_gdb_cli
│       ├── __init__.py
│       ├── cli.py                     # ← 新增 cuda-* 子命令
│       ├── client.py                  # 无改动
│       ├── launcher.py                # ← gdb → cuda-gdb
│       ├── session.py                 # ← 新增 CUDA 元数据
│       ├── safety.py                  # ← 新增 CUDA 命令白名单
│       ├── formatters.py              # 无改动
│       ├── errors.py                  # ← 新增 CUDA 错误类型
│       ├── env_check.py               # ← 新增 CUDA 环境检查
│       └── gdb_server/
│           ├── gdb_rpc_server.py      # ← 注册 CUDA handler
│           ├── handlers.py            # 无改动（CPU handler）
│           ├── cuda_handlers.py       # ← 新增：8 个 CUDA handler
│           ├── value_formatter.py     # ← 扩展地址空间
│           └── heartbeat.py           # 无改动
├── skills/
│   └── cuda-gdb-cli/                  # ← 新增 CUDA Skill
│       ├── SKILL.md
│       └── evals/
├── tests/
│   ├── test_cli.py
│   ├── test_cuda_handlers.py          # ← 新增
│   └── crash_test/
│       ├── cuda_crash_test.cu         # ← 新增 CUDA 测试程序
│       └── Makefile
└── docs/
    └── cuda-gdb-commands.md           # ← CUDA 命令速查

```

---

## 10. Agent 使用示例（完整 Workflow）

### 10.1 Coredump 分析

```bash
# 1. 加载 CUDA coredump
$ cuda-gdb-cli load --binary ./matmul --core ./core.99421
{"session_id": "f465d650", "mode": "core", "status": "started"}

$ SESSION="f465d650"

# 2. GPU 概览
$ cuda-gdb-cli cuda-devices -s $SESSION
{"devices": [{"device_id": 0, "name": "NVIDIA A100", "sm_count": 108}]}

$ cuda-gdb-cli cuda-kernels -s $SESSION
{"kernels": [{"kernel_id": 0, "function": "matmul_kernel",
              "grid_dim": [32,32,1], "block_dim": [16,16,1]}]}

# 3. 查看异常
$ cuda-gdb-cli cuda-exceptions -s $SESSION
{"exceptions": [{"kernel": 0, "block_idx": [0,0,0], "thread_idx": [1,0,0],
                 "exception": "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS"}]}

# 4. 切换到异常线程
$ cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"
{"focus": {"kernel": 0, "block_idx": [0,0,0], "thread_idx": [1,0,0]},
 "frame": {"function": "matmul_kernel", "file": "matmul.cu", "line": 28}}

# 5. 查看调用栈和局部变量
$ cuda-gdb-cli bt -s $SESSION --full
{"frames": [{"number": 0, "function": "matmul_kernel", "file": "matmul.cu",
             "line": 28, "locals": [{"name": "k", "value": 1024}]}]}

# 6. 检查变量
$ cuda-gdb-cli eval-cmd -s $SESSION "row * N + k"
{"expression": "row * N + k", "value": 1024, "type": "int"}

# 7. 检查共享内存
$ cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "tile_A" --type float --count 16
{"space": "shared", "elements": [1.0, 2.0, 3.0, ...]}

# 8. 结束
$ cuda-gdb-cli stop -s $SESSION
{"session_id": "f465d650", "status": "stopped"}
```

### 10.2 Live Attach 调试

```bash
# 1. Attach 到运行中的 CUDA 进程
$ cuda-gdb-cli attach --pid 9876 --binary ./training_app
{"session_id": "a1b2c3d4", "mode": "attach", "status": "started"}

$ SESSION="a1b2c3d4"

# 2. 查看当前 GPU 状态
$ cuda-gdb-cli cuda-kernels -s $SESSION
$ cuda-gdb-cli cuda-threads -s $SESSION --limit 10

# 3. 设置条件断点（通过 exec 透传）
$ cuda-gdb-cli exec -s $SESSION "break kernel.cu:42 if threadIdx.x == 0" --safety-level full
{"command": "break kernel.cu:42 if threadIdx.x == 0", "output": "Breakpoint 1 at ..."}

# 4. 继续执行
$ cuda-gdb-cli exec -s $SESSION "continue" --safety-level full

# 5. 命中断点后检查状态
$ cuda-gdb-cli cuda-focus -s $SESSION
$ cuda-gdb-cli locals-cmd -s $SESSION
$ cuda-gdb-cli eval-cmd -s $SESSION "blockDim.x * blockIdx.x + threadIdx.x"

# 6. Detach
$ cuda-gdb-cli stop -s $SESSION
```

---

## 11. 与 gdb-cli 的改动清单

### 11.1 改动文件汇总

| 文件                             | 改动类型       | 改动内容                                                   |
| -------------------------------- | -------------- | ---------------------------------------------------------- |
| `pyproject.toml`               | 修改           | 包名 `gdb-cli` → `cuda-gdb-cli`，新增 CUDA 相关元数据 |
| `cli.py`                       | 扩展           | 新增 8 个 `cuda-*` 子命令                                |
| `launcher.py`                  | 修改           | 默认 `gdb_path` 改为 `cuda-gdb`，新增 CUDA 启动参数    |
| `session.py`                   | 扩展           | SessionMeta 新增 `cuda_version`, `gpu_device` 字段     |
| `safety.py`                    | 扩展           | 新增 CUDA 命令白名单                                       |
| `env_check.py`                 | 扩展           | 新增 cuda-gdb、CUDA Toolkit、GPU 驱动检查                  |
| `errors.py`                    | 扩展           | 新增 `CUDAError`, `CUDAMemoryError` 等                 |
| `gdb_rpc_server.py`            | 扩展           | 注册 CUDA handler                                          |
| `value_formatter.py`           | 扩展           | 地址空间修饰符处理                                         |
| `cuda_handlers.py`             | **新增** | 8 个 CUDA handler 实现                                     |
| `skills/cuda-gdb-cli/SKILL.md` | **新增** | CUDA 调试 Skill 定义                                       |

### 11.2 不改动的文件

| 文件              | 原因                                      |
| ----------------- | ----------------------------------------- |
| `client.py`     | 通用 Unix Socket 客户端，与调试器类型无关 |
| `formatters.py` | JSON 输出格式化，与调试器类型无关         |
| `heartbeat.py`  | 心跳超时机制，与调试器类型无关            |
| `handlers.py`   | CPU 端 handler 在 cuda-gdb 中同样可用     |

---

## 12. 演进路线

### Phase 1: Fork + 基础 CUDA 支持（2 周）

- [ ] Fork gdb-cli，改名为 cuda-gdb-cli
- [ ] launcher.py: gdb → cuda-gdb
- [ ] 新增 `cuda_handlers.py`：`cuda_threads`, `cuda_kernels`, `cuda_focus`, `cuda_exceptions`
- [ ] CLI 注册 4 个核心 CUDA 子命令
- [ ] env_check.py 扩展 CUDA 环境检查
- [ ] Coredump 模式端到端验证

### Phase 2: 完整 CUDA 能力（2 周）

- [ ] 新增 `cuda_memory`, `cuda_devices`, `cuda_warps`, `cuda_lanes` handler
- [ ] value_formatter.py 扩展地址空间支持
- [ ] safety.py 扩展 CUDA 命令白名单
- [ ] Live Attach 模式验证
- [ ] 完善 CUDA 异常类型映射

### Phase 3: Agent 集成（1 周）

- [ ] 编写 Claude Code Skill（SKILL.md）
- [ ] 编写 CUDA 调试命令速查文档
- [ ] 端到端 Skill 测试（coredump + live attach）

### Phase 4: 上游贡献（可选）

- [ ] 向 gdb-cli 提交 PR，将 CUDA 支持作为可选扩展合入上游
- [ ] 或维护独立 fork，定期同步上游更新
