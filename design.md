# CUDA-GDB CLI — 面向 AI Agent 的 CUDA 调试命令行工具

## 0. 设计哲学

**一句话概括：AI Agent 通过 bash 命令直接操控 `cuda-gdb`，像人类在终端里调试一样简单。**

不需要 RPC 服务器，不需要 JSON-RPC 协议栈，不需要 MCP。只需要一个 CLI 工具管理 `cuda-gdb` 进程的生命周期，Agent 发 bash 命令，拿回文本结果，继续推理，再发下一条命令。

```
┌─────────────────────────────────────────────────────┐
│  AI Agent (Claude Code / Codex CLI / Cursor / ...)  │
│                                                     │
│  工具：bash shell                                    │
│  能力：发送任意 shell 命令，读取 stdout/stderr       │
└──────────────────────┬──────────────────────────────┘
                       │  bash 命令
                       ▼
┌─────────────────────────────────────────────────────┐
│  cuda-gdb-cli (Python CLI 脚本)                     │
│                                                     │
│  • 管理 cuda-gdb 子进程生命周期                      │
│  • 将 Agent 的命令转发给 cuda-gdb                    │
│  • 捕获输出并返回给 Agent                            │
│  • 可选：结构化 JSON 输出                            │
└──────────────────────┬──────────────────────────────┘
                       │  stdin/stdout (pexpect)
                       ▼
┌─────────────────────────────────────────────────────┐
│  cuda-gdb 进程                                      │
│                                                     │
│  NVIDIA 官方 CUDA 调试器                             │
│  支持 GPU 线程、共享内存、Warp 异常等                │
└─────────────────────────────────────────────────────┘
```

---

## 1. 核心交互模型

### 1.1 Agent 视角：就是发 bash 命令

Agent 不需要学习任何新协议。它只需要会用 `bash` 工具：

```bash
# 启动调试会话（加载 coredump）
cuda-gdb-cli start --core /path/to/core --exec /path/to/app

# 启动调试会话（attach 到进程）
cuda-gdb-cli start --exec /path/to/app --args "--batch-size 64"

# 发送调试命令
cuda-gdb-cli send "info cuda threads"
cuda-gdb-cli send "cuda thread (1,0,0)"
cuda-gdb-cli send "bt"
cuda-gdb-cli send "print shared_mem[threadIdx.x]"

# 批量发送多条命令
cuda-gdb-cli send "break kernel.cu:42" "run" "info cuda threads"

# 查看会话状态
cuda-gdb-cli status

# 结束会话
cuda-gdb-cli stop
```

**就这么简单。** Agent 的 System Prompt 里告诉它有 `cuda-gdb-cli` 这个工具可用，它就能自主调试 CUDA 程序。

### 1.2 与 Claude Code 的集成方式

Claude Code 的 Agent 可以直接通过 `bash` 工具调用 `cuda-gdb-cli`：

```
User: "帮我分析这个 CUDA coredump，找出崩溃原因"

Agent 思考: 我需要用 cuda-gdb-cli 加载 coredump 并分析

Agent 执行:
  bash: cuda-gdb-cli start --core ./core.12345 --exec ./my_cuda_app
  → 输出: "Session started. cuda-gdb ready. Mode: coredump"

  bash: cuda-gdb-cli send "bt"
  → 输出: "#0 kernel<<<(1,1,1),(256,1,1)>>> at kernel.cu:42 ..."

  bash: cuda-gdb-cli send "info cuda threads"
  → 输出: "  BlockIdx ThreadIdx ... Exception ..."

  bash: cuda-gdb-cli send "cuda thread (1,0,0)" "print idx"
  → 输出: "idx = 1024"  (越界!)

Agent 回答: "崩溃原因是 kernel.cu:42 行的数组越界访问，
            线程 (1,0,0) 的 idx=1024 超出了数组边界..."
```

---

## 2. CLI 命令设计

### 2.1 命令总览

| 命令       | 用途                    | 示例                                               |
| ---------- | ----------------------- | -------------------------------------------------- |
| `start`  | 启动 cuda-gdb 会话      | `cuda-gdb-cli start --core dump.core --exec app` |
| `send`   | 发送一条或多条 GDB 命令 | `cuda-gdb-cli send "bt" "info locals"`           |
| `status` | 查看当前会话状态        | `cuda-gdb-cli status`                            |
| `stop`   | 终止会话，清理进程      | `cuda-gdb-cli stop`                              |

只有 4 个子命令，Agent 零学习成本。

### 2.2 `start` — 启动调试会话

```
cuda-gdb-cli start [OPTIONS]

OPTIONS:
  --exec PATH          可执行文件路径（必需）
  --core PATH          Coredump 文件路径（coredump 模式）
  --pid PID            Attach 到运行中的进程（attach 模式）
  --args "ARGS"        传给被调试程序的参数（live 模式）
  --cuda-gdb PATH      cuda-gdb 可执行文件路径（默认从 PATH 查找）
  --timeout SECONDS    命令超时时间（默认 30s）
  --session-id ID      指定会话 ID（默认自动生成）
  --json               输出 JSON 格式
```

**三种启动模式：**

```bash
# 模式 1: Coredump 分析（只读）
cuda-gdb-cli start --exec ./app --core ./core.12345

# 模式 2: Attach 到运行中进程
cuda-gdb-cli start --exec ./app --pid 12345

# 模式 3: 启动新进程调试
cuda-gdb-cli start --exec ./app --args "--input data.bin"
```

**输出示例（纯文本）：**

```
[cuda-gdb-cli] Session started
  Session ID : s_a1b2c3
  Mode       : coredump
  Executable : /home/user/app
  Core       : /home/user/core.12345
  cuda-gdb   : /usr/local/cuda/bin/cuda-gdb
  Status     : ready
```

**输出示例（JSON，`--json`）：**

```json
{
  "status": "ok",
  "session_id": "s_a1b2c3",
  "mode": "coredump",
  "executable": "/home/user/app",
  "core": "/home/user/core.12345"
}
```

### 2.3 `send` — 发送调试命令

```
cuda-gdb-cli send [OPTIONS] COMMAND [COMMAND ...]

OPTIONS:
  --session-id ID      指定会话（单会话时可省略）
  --timeout SECONDS    本次命令超时（覆盖默认值）
  --json               输出 JSON 格式
```

**单条命令：**

```bash
cuda-gdb-cli send "info cuda threads"
```

**输出（纯文本，直接透传 cuda-gdb 的原始输出）：**

```
  BlockIdx  ThreadIdx  To  Name  Filename  Line
* (0,0,0)  (0,0,0)    -   kern  kernel.cu  42
  (0,0,0)  (1,0,0)    -   kern  kernel.cu  42
  (0,0,0)  (2,0,0)    -   kern  kernel.cu  42
```

**批量命令：**

```bash
cuda-gdb-cli send "cuda thread (1,0,0)" "bt" "info locals"
```

**输出（每条命令的输出用分隔线隔开）：**

```
>>> cuda thread (1,0,0)
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (1,0,0)]

>>> bt
#0  kernel<<<(1,1,1),(256,1,1)>>> (data=0x7fff..., n=1024) at kernel.cu:42
#1  0x00007ffff7... in ?? ()

>>> info locals
idx = 1024
data = 0x7fffdeadbeef
n = 1024
```

**JSON 输出（`--json`）：**

```json
{
  "status": "ok",
  "results": [
    {
      "command": "cuda thread (1,0,0)",
      "output": "[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (1,0,0)]"
    },
    {
      "command": "bt",
      "output": "#0  kernel<<<(1,1,1),(256,1,1)>>> ..."
    },
    {
      "command": "info locals",
      "output": "idx = 1024\ndata = 0x7fffdeadbeef\nn = 1024"
    }
  ]
}
```

### 2.4 `status` — 查看会话状态

```bash
cuda-gdb-cli status [--session-id ID] [--json]
```

**输出：**

```
[cuda-gdb-cli] Session s_a1b2c3
  Status     : ready
  Mode       : coredump
  PID        : 54321 (cuda-gdb process)
  Uptime     : 2m 34s
  Commands   : 12 sent
```

### 2.5 `stop` — 终止会话

```bash
cuda-gdb-cli stop [--session-id ID]
```

**输出：**

```
[cuda-gdb-cli] Session s_a1b2c3 terminated.
```

---

## 3. 内部架构

### 3.1 进程模型

```
cuda-gdb-cli (Python)
    │
    ├── SessionManager          # 管理会话生命周期
    │     └── Session           # 单个调试会话
    │           ├── pexpect.spawn(cuda-gdb)   # 子进程
    │           ├── command_queue              # 命令队列
    │           └── state (ready/busy/dead)    # 状态
    │
    ├── CommandRouter           # 解析 CLI 参数，路由到对应 Session
    │
    └── OutputFormatter         # 格式化输出（text / json）
```

### 3.2 会话状态文件

每个会话在 `/tmp/cuda-gdb-cli/` 下维护一个状态文件，实现跨命令调用的会话持久化：

```
/tmp/cuda-gdb-cli/
  └── s_a1b2c3/
        ├── session.json        # 会话元信息
        ├── cuda-gdb.pid        # cuda-gdb 进程 PID
        └── socket               # Unix Domain Socket（进程间通信）
```

**为什么需要这个？** 因为每次 `cuda-gdb-cli send` 都是一个独立的 CLI 进程调用。我们需要一种机制让多次 CLI 调用共享同一个 `cuda-gdb` 进程。

### 3.3 进程间通信方案

```
                    首次 start                    后续 send
                    ─────────                    ─────────
cuda-gdb-cli ──→ 启动 Daemon 进程 ──→ spawn cuda-gdb
                  监听 Unix Socket
                       ▲
                       │ Unix Socket
cuda-gdb-cli ──────────┘
  (send 命令)    连接 Daemon，发送命令，接收结果
```

**Daemon 进程**：`cuda-gdb-cli start` 时 fork 一个后台 Daemon，它持有 `cuda-gdb` 子进程。后续的 `send` 命令通过 Unix Domain Socket 与 Daemon 通信。

这样 Agent 每次调用 `cuda-gdb-cli send` 都是一个短生命周期的 CLI 进程，但底层的 `cuda-gdb` 会话是持久的。

### 3.4 核心类设计

```python
class CudaGdbDaemon:
    """后台 Daemon，持有 cuda-gdb 进程"""

    def __init__(self, exec_path, core_path=None, pid=None, args=None):
        self.session_id = generate_session_id()
        self.gdb_process = None       # pexpect.spawn 实例
        self.socket_path = f"/tmp/cuda-gdb-cli/{self.session_id}/socket"
        self.mode = "coredump" | "attach" | "live"

    def start(self):
        """启动 cuda-gdb 子进程"""
        cmd = self._build_cuda_gdb_command()
        self.gdb_process = pexpect.spawn(cmd, timeout=30)
        self._wait_for_prompt()       # 等待 (cuda-gdb) 提示符
        self._start_socket_server()   # 开始监听 Unix Socket

    def execute_command(self, command: str) -> str:
        """向 cuda-gdb 发送命令并返回输出"""
        self.gdb_process.sendline(command)
        self.gdb_process.expect(r'\(cuda-gdb\)\s*')  # 等待下一个提示符
        return self.gdb_process.before.decode()

    def stop(self):
        """终止 cuda-gdb 进程，清理资源"""
        self.gdb_process.sendline("quit")
        self.gdb_process.close()
        cleanup_session_files(self.session_id)


class CudaGdbClient:
    """CLI 前端，通过 Unix Socket 与 Daemon 通信"""

    def __init__(self, session_id=None):
        self.session_id = session_id or self._find_active_session()

    def send(self, commands: list[str]) -> list[CommandResult]:
        """发送命令到 Daemon 并获取结果"""
        sock = connect_to_daemon(self.session_id)
        results = []
        for cmd in commands:
            sock.send(json.dumps({"action": "execute", "command": cmd}))
            response = json.loads(sock.recv())
            results.append(CommandResult(cmd, response["output"]))
        return results
```

### 3.5 提示符检测与输出捕获

`cuda-gdb` 的提示符是 `(cuda-gdb) `，这是我们判断命令执行完毕的标志：

```python
# 提示符正则（兼容多种场景）
PROMPT_PATTERN = r'\(cuda-gdb\)\s*$'

# 特殊情况处理
CONTINUE_PATTERNS = [
    r'---Type <return> to continue',    # 分页提示 → 自动发送回车
    r'\[New Thread',                     # 新线程通知 → 继续等待
    r'Make breakpoint pending',          # 断点确认 → 自动回答 y
]
```

**关键设计：自动处理交互式提示**

`cuda-gdb` 有时会弹出交互式提示（如分页、确认），CLI 工具自动处理这些，不让 Agent 看到：

```python
def execute_command(self, command: str) -> str:
    self.gdb_process.sendline(command)

    output_parts = []
    while True:
        index = self.gdb_process.expect([
            PROMPT_PATTERN,                          # 0: 正常结束
            r'---Type <return> to continue',         # 1: 分页
            r'Make breakpoint pending.*\(y or n\)',  # 2: 断点确认
            pexpect.TIMEOUT,                         # 3: 超时
        ])

        output_parts.append(self.gdb_process.before.decode())

        if index == 0:    # 命令完成
            break
        elif index == 1:  # 分页 → 自动翻页
            self.gdb_process.sendline("")
        elif index == 2:  # 断点确认 → 自动 yes
            self.gdb_process.sendline("y")
        elif index == 3:  # 超时
            raise TimeoutError(f"Command timed out: {command}")

    return "".join(output_parts).strip()
```

---

## 4. CUDA 特有调试能力

### 4.1 Agent 可用的 CUDA 调试命令速查

CLI 工具**不封装**这些命令，而是让 Agent 直接发送原生 `cuda-gdb` 命令。Agent 的 System Prompt 中会提供这份速查表：

#### GPU 线程导航

```bash
# 查看所有 CUDA 线程
cuda-gdb-cli send "info cuda threads"

# 切换到指定 GPU 线程（软件坐标）
cuda-gdb-cli send "cuda thread (1,0,0)"

# 切换到指定 Block
cuda-gdb-cli send "cuda block (2,0,0)"

# 切换到指定 Kernel
cuda-gdb-cli send "cuda kernel 0"

# 查看当前焦点
cuda-gdb-cli send "cuda thread"
```

#### GPU 内存检查

```bash
# 查看共享内存
cuda-gdb-cli send "print @shared float[32] shared_data"

# 查看全局内存
cuda-gdb-cli send "print @global int[10] global_array"

# 查看局部变量
cuda-gdb-cli send "info locals"

# 查看寄存器
cuda-gdb-cli send "info registers"
```

#### 断点与执行控制

```bash
# 在 CUDA kernel 中设置断点
cuda-gdb-cli send "break kernel.cu:42"

# 条件断点（只在特定线程触发）
cuda-gdb-cli send "break kernel.cu:42 if threadIdx.x == 0"

# 运行 / 继续 / 单步
cuda-gdb-cli send "run"
cuda-gdb-cli send "continue"
cuda-gdb-cli send "next"
cuda-gdb-cli send "step"
```

#### 异常与崩溃分析

```bash
# 查看 CUDA 异常信息
cuda-gdb-cli send "info cuda exceptions"

# 查看所有 Kernel
cuda-gdb-cli send "info cuda kernels"

# 查看 GPU 硬件拓扑
cuda-gdb-cli send "info cuda devices"

# 反汇编崩溃点
cuda-gdb-cli send "disassemble"
```

### 4.2 Coredump vs Live 模式差异

| 能力           | Coredump 模式 | Live 模式 |
| -------------- | :-----------: | :-------: |
| 查看调用栈     |      ✅      |    ✅    |
| 查看变量       |      ✅      |    ✅    |
| 查看 GPU 线程  |      ✅      |    ✅    |
| 查看共享内存   |      ✅      |    ✅    |
| 设置断点       |      ❌      |    ✅    |
| 单步执行       |      ❌      |    ✅    |
| 修改变量       |      ❌      |    ✅    |
| run / continue |      ❌      |    ✅    |

CLI 工具在 Coredump 模式下**不做额外限制**——如果 Agent 发了不支持的命令，`cuda-gdb` 自己会报错，Agent 会看到错误信息并自行调整。这比在 CLI 层做复杂的模态守卫要简单得多。

---

## 5. 错误处理

### 5.1 错误分类与输出

CLI 工具的错误处理原则：**透传 cuda-gdb 的错误，只在 CLI 层处理进程级错误。**

```bash
# cuda-gdb 命令错误 → 直接透传
$ cuda-gdb-cli send "print nonexistent_var"
[cuda-gdb-cli] >>> print nonexistent_var
No symbol "nonexistent_var" in current context.

# CLI 层错误 → 带 [ERROR] 前缀
$ cuda-gdb-cli send "bt"
[ERROR] No active session. Run 'cuda-gdb-cli start' first.

$ cuda-gdb-cli start --core /nonexistent/core --exec ./app
[ERROR] Core file not found: /nonexistent/core

# 超时错误
$ cuda-gdb-cli send "some_long_command" --timeout 5
[ERROR] Command timed out after 5s: some_long_command
```

### 5.2 进程崩溃恢复

```python
def execute_command(self, command: str) -> str:
    if not self.gdb_process.isalive():
        raise SessionDead("cuda-gdb process has terminated unexpectedly. "
                          "Run 'cuda-gdb-cli start' to begin a new session.")
    # ... 正常执行
```

Agent 看到 `SessionDead` 错误后，会自行决定是否重新启动会话。

### 5.3 退出码约定

| 退出码 | 含义                  |
| ------ | --------------------- |
| 0      | 成功                  |
| 1      | CLI 参数错误          |
| 2      | 会话不存在 / 已终止   |
| 3      | cuda-gdb 命令执行错误 |
| 4      | 超时                  |
| 5      | cuda-gdb 进程崩溃     |

---

## 6. 安装与依赖

### 6.1 依赖

```
Python >= 3.8
pexpect >= 4.8
cuda-gdb (随 CUDA Toolkit 安装)
```

### 6.2 安装

```bash
pip install cuda-gdb-cli

# 或从源码安装
git clone https://github.com/xxx/cuda-gdb-cli.git
cd cuda-gdb-cli
pip install -e .
```

### 6.3 验证安装

```bash
# 检查 cuda-gdb 是否可用
cuda-gdb-cli doctor
```

**输出：**

```
[cuda-gdb-cli] Environment Check
  Python     : 3.10.12 ✓
  pexpect    : 4.9.0 ✓
  cuda-gdb   : /usr/local/cuda-12.2/bin/cuda-gdb ✓
  CUDA       : 12.2 ✓
  GPU Driver : 535.104.05 ✓
  Status     : All checks passed ✓
```

---

## 7. Agent System Prompt 集成模板

以下是给 AI Agent 的 System Prompt 片段，告诉它如何使用 `cuda-gdb-cli`：

```markdown
## CUDA 调试工具

你可以使用 `cuda-gdb-cli` 命令行工具来调试 CUDA 程序。

### 基本用法

1. **启动调试会话**
   - Coredump: `cuda-gdb-cli start --exec <可执行文件> --core <coredump文件>`
   - Live:     `cuda-gdb-cli start --exec <可执行文件> --args "<参数>"`

2. **发送调试命令**
   `cuda-gdb-cli send "<cuda-gdb命令>"`

3. **结束会话**
   `cuda-gdb-cli stop`

### 常用调试命令

- `bt` — 查看调用栈
- `info cuda threads` — 查看所有 GPU 线程
- `cuda thread (x,y,z)` — 切换到指定 GPU 线程
- `info locals` — 查看局部变量
- `print <expr>` — 打印表达式
- `info cuda kernels` — 查看所有 Kernel
- `info cuda exceptions` — 查看 CUDA 异常
- `break <file>:<line>` — 设置断点（仅 Live 模式）
- `continue` / `next` / `step` — 执行控制（仅 Live 模式）

### 调试策略

1. 先用 `bt` 和 `info cuda threads` 获取全局视图
2. 切换到异常线程 `cuda thread (x,y,z)` 查看局部状态
3. 用 `info locals` 和 `print` 检查变量值
4. 如果是内存问题，检查数组索引是否越界
5. 如果是 Warp 异常，用 `info cuda exceptions` 查看异常类型
```

---

## 8. 完整调试 Workflow 示例

### 8.1 Coredump 分析 Workflow

```bash
# Step 1: 启动
$ cuda-gdb-cli start --exec ./matmul --core ./core.99421
[cuda-gdb-cli] Session started
  Session ID : s_f7e2a1
  Mode       : coredump
  Status     : ready

# Step 2: 查看崩溃调用栈
$ cuda-gdb-cli send "bt"
>>> bt
#0  matmul_kernel<<<(32,32,1),(16,16,1)>>> (A=0x7f..., B=0x7f..., C=0x7f..., N=1024)
    at matmul.cu:28
#1  0x00007f4a3c2... in ?? ()

# Step 3: 查看 GPU 线程状态
$ cuda-gdb-cli send "info cuda threads"
>>> info cuda threads
  BlockIdx    ThreadIdx  To  Name           Filename   Line
* (0,0,0)     (0,0,0)   -   matmul_kernel  matmul.cu  28
  (0,0,0)     (1,0,0)   -   matmul_kernel  matmul.cu  28
  ...
  (31,31,0)   (15,15,0) -   matmul_kernel  matmul.cu  28

# Step 4: 检查崩溃线程的局部变量
$ cuda-gdb-cli send "info locals"
>>> info locals
row = 0
col = 0
sum = 0
k = 1024

# Step 5: 检查可疑的索引计算
$ cuda-gdb-cli send "print row * N + col"
>>> print row * N + col
$1 = 0

$ cuda-gdb-cli send "print A[row * N + k]"
>>> print A[row * N + k]
Cannot access memory at address 0x7f4a3c000000

# Step 6: 确认越界 → k=1024 但 N=1024，A[row*1024+1024] 越界
$ cuda-gdb-cli send "print N"
>>> print N
$2 = 1024

# Step 7: 结束
$ cuda-gdb-cli stop
[cuda-gdb-cli] Session s_f7e2a1 terminated.
```

### 8.2 Live 调试 Workflow

```bash
# Step 1: 启动 Live 调试
$ cuda-gdb-cli start --exec ./vector_add --args "1000000"
[cuda-gdb-cli] Session started
  Session ID : s_b3c4d5
  Mode       : live
  Status     : ready

# Step 2: 设置条件断点
$ cuda-gdb-cli send "break vector_add.cu:15 if threadIdx.x == 255"
>>> break vector_add.cu:15 if threadIdx.x == 255
Breakpoint 1 at 0x... : file vector_add.cu, line 15.

# Step 3: 运行
$ cuda-gdb-cli send "run"
>>> run
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (255,0,0)]
Breakpoint 1, vector_add_kernel at vector_add.cu:15

# Step 4: 检查状态
$ cuda-gdb-cli send "info locals" "print idx" "print a[idx]"
>>> info locals
idx = 255
>>> print idx
$1 = 255
>>> print a[idx]
$2 = 255.0

# Step 5: 单步执行
$ cuda-gdb-cli send "next"
>>> next
16        c[idx] = a[idx] + b[idx];

$ cuda-gdb-cli send "print c[idx]"
>>> print c[idx]
$3 = 510.0

# Step 6: 继续执行到结束
$ cuda-gdb-cli send "continue"
>>> continue
[Inferior 1 (process 12345) exited normally]

# Step 7: 结束
$ cuda-gdb-cli stop
[cuda-gdb-cli] Session s_b3c4d5 terminated.
```

---

## 9. 项目结构

```
cuda-gdb-cli/
├── pyproject.toml              # 项目配置与依赖
├── README.md                   # 使用文档
├── src/
│   └── cuda_gdb_cli/
│       ├── __init__.py
│       ├── __main__.py         # CLI 入口 (python -m cuda_gdb_cli)
│       ├── cli.py              # argparse 命令解析
│       ├── daemon.py           # CudaGdbDaemon（后台进程，持有 cuda-gdb）
│       ├── client.py           # CudaGdbClient（CLI 前端，连接 Daemon）
│       ├── session.py          # SessionManager（会话状态管理）
│       ├── output.py           # OutputFormatter（text/json 格式化）
│       └── prompt.py           # cuda-gdb 提示符检测与交互处理
├── tests/
│   ├── test_cli.py
│   ├── test_daemon.py
│   └── test_prompt.py
└── examples/
    ├── agent_prompt.md         # Agent System Prompt 模板
    └── workflows/
        ├── coredump_analysis.sh
        └── live_debug.sh
```

---

## 10. 与其他方案的对比

| 维度                     | cuda-gdb-cli (本方案) |         RPC 服务方案         |     直接调用 cuda-gdb     |
| ------------------------ | :-------------------: | :--------------------------: | :------------------------: |
| **Agent 集成难度** | ⭐ 极低（bash 命令） | ⭐⭐⭐ 高（需要 RPC 客户端） | ⭐⭐ 中（需处理交互式 IO） |
| **部署复杂度**     |    `pip install`    |         需要启动服务         |          无需安装          |
| **会话管理**       |    自动（Daemon）    |           需要实现           |      无（每次新进程）      |
| **交互式提示处理** |         自动         |           需要实现           |          手动处理          |
| **多会话支持**     |     ✅ session-id     |              ✅              |             ❌             |
| **输出格式**       |      text + JSON      |             JSON             |          原始文本          |
| **学习成本**       |       4 个命令       |       学习 API Schema       |       学习 GDB 交互       |

**本方案的核心优势：Agent 不需要学任何新东西，它已经会用 bash，那就够了。**

---

## 11. 演进路线

### Phase 1: MVP（2 周）

- [X] `start` / `send` / `stop` / `status` 四个核心命令
- [X] Coredump 模式支持
- [X] pexpect 进程管理 + 提示符检测
- [X] 纯文本输出

### Phase 2: 增强（2 周）

- [ ] Live 调试模式（attach / launch）
- [ ] JSON 输出格式
- [ ] Daemon 模式（跨 CLI 调用的会话持久化）
- [ ] 多会话支持

### Phase 3: Agent 优化（2 周）

- [ ] `doctor` 环境检查命令
- [ ] 智能超时（根据命令类型自动调整）
- [ ] 输出截断保护（超长输出自动摘要）
- [ ] Agent System Prompt 模板库

### Phase 4: 可选扩展

- [ ] MCP Server 封装（如果需要与 MCP 生态集成）
- [ ] Web UI（可视化调试面板）
- [ ] 远程调试支持（SSH 隧道）
