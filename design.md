# 面向 AI Agent 的 CUDA-GDB-CLI RPC 服务架构详细设计方案

> **版本**: v1.0
> **日期**: 2026-03-23
> **状态**: 设计阶段

---

## 目录

- [1. 设计背景与目标](#1-设计背景与目标)
- [2. 系统架构总览](#2-系统架构总览)
- [3. 第一层：Agent 客户端层](#3-第一层agent-客户端层)
- [4. 第二层：宿主代理传输层](#4-第二层宿主代理传输层)
- [5. 第三层：内嵌式 Python RPC 引擎](#5-第三层内嵌式-python-rpc-引擎)
- [6. GPU 特有对象访问机制设计](#6-gpu-特有对象访问机制设计)
- [7. 模态守卫与状态机设计](#7-模态守卫与状态机设计)
- [8. 异常处理与容错机制](#8-异常处理与容错机制)
- [9. 部署方案与配置规范](#9-部署方案与配置规范)
- [10. 演进路线与扩展规划](#10-演进路线与扩展规划)
- [附录 A：CUDA_EXCEPTION 完整映射表](#附录-acuda_exception-完整映射表)
- [附录 B：调试模态对比矩阵](#附录-b调试模态对比矩阵)

---

## 1. 设计背景与目标

### 1.1 问题域

在 CUDA 异构计算调试场景中，AI Agent 面临三大核心挑战：

1. **非结构化输出**：cuda-gdb 的 CLI 输出面向人类视觉解析，包含大量格式化空白、ANSI 转义序列和上下文相关的提示符，AI Agent 直接解析极易导致状态误判。
2. **阻塞性执行**：`continue`、`step` 等控制流命令会无限期阻塞 Python 解释器的 GIL，若 RPC 服务与 GDB 执行引擎共享同一线程，Agent 通信链路将完全死锁。
3. **硬件状态的多维复杂性**：GPU 调试涉及 Grid/Block/Thread 软件坐标、Device/SM/Warp/Lane 硬件坐标、`@shared`/`@global`/`@local` 地址空间修饰符、以及 Warp 级异常码等多维度状态，远超传统 CPU 调试的复杂度。

### 1.2 设计目标

| 目标维度             | 具体要求                                                                     |
| -------------------- | ---------------------------------------------------------------------------- |
| **结构化**     | 所有 Agent 交互均通过 JSON-RPC 2.0 协议，返回严格 Schema 约束的 JSON 响应    |
| **无阻塞**     | 实时调试模式下，控制流命令不得阻塞 RPC 通信通道                              |
| **高保真**     | 直接通过 GDB Python 原生 API 提取 `gdb.Value` 对象，杜绝 CLI 文本正则解析  |
| **模态感知**   | 自动识别 Live/Coredump 模式，主动拦截非法操作，防止 Agent 幻觉导致的状态污染 |
| **Token 高效** | 响应 JSON 精简至最小必要字段，避免 DAP 协议式的冗余 GUI 状态数据             |
| **领域语义**   | 封装 CUDA 领域知识为高级工具端点，而非暴露底层裸命令                         |

### 1.3 设计哲学

```
前端保持 MCP 标准的无状态交互
后端依托内嵌 Python 解释器提供强状态、高保真且防阻塞的微观控制
```

核心原则：**不做 CLI 文本的搬运工，做 GDB 内存空间的原生翻译官。**

---

## 2. 系统架构总览

### 2.1 三层解耦架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 1: Agent Client Layer                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  Cursor   │  │ LangGraph│  │  Claude  │  │ Custom Agent  │   │
│  │  IDE      │  │  Agent   │  │ Desktop  │  │  Framework    │   │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └──────┬────────┘   │
│        │              │              │               │           │
│        └──────────────┴──────┬───────┴───────────────┘           │
│                              │                                   │
│                     JSON-RPC 2.0 / MCP                           │
│                     Tool Calling Request                         │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│              Layer 2: Host Proxy & Transport Layer               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                   MCP Transport Proxy                      │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐  │  │
│  │  │  stdio   │  │ Unix Domain  │  │  cuda-gdb Process   │  │  │
│  │  │  Bridge  │  │   Socket     │  │  Lifecycle Manager  │  │  │
│  │  └─────┬────┘  └──────┬───────┘  └──────────┬──────────┘  │  │
│  │        │               │                     │             │  │
│  │        └───────────────┴──────┬──────────────┘             │  │
│  │                               │                            │  │
│  │                    Internal IPC (pipe/UDS)                 │  │
│  └───────────────────────────────┬────────────────────────────┘  │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│          Layer 3: Embedded Python RPC Engine (核心)              │
│          运行于 cuda-gdb 进程内部 Python 解释器中                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              RPC Listener Thread (gdb.Thread)            │    │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │    │
│  │  │ JSON-RPC   │  │  Request     │  │  Response        │  │    │
│  │  │ Decoder    │  │  Validator   │  │  Serializer      │  │    │
│  │  └─────┬──────┘  └──────┬───────┘  └────────┬────────┘  │    │
│  │        └────────────────┴────────┬──────────┘            │    │
│  │                                  │                       │    │
│  │                    Thread-Safe Command Queue              │    │
│  │                    (queue.Queue + Condition)              │    │
│  └──────────────────────────────────┬───────────────────────┘    │
│                                     │                            │
│                          gdb.post_event(callable)                │
│                                     │                            │
│  ┌──────────────────────────────────▼───────────────────────┐    │
│  │              GDB Main Event Loop (主线程)                 │    │
│  │  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐  │    │
│  │  │ Modality Guard │  │ CUDA Focus   │  │ Memory      │  │    │
│  │  │ State Machine  │  │ Manager      │  │ Accessor    │  │    │
│  │  └────────────────┘  └──────────────┘  └─────────────┘  │    │
│  │  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐  │    │
│  │  │ Register       │  │ Exception    │  │ Execution   │  │    │
│  │  │ Probe          │  │ Analyzer     │  │ Controller  │  │    │
│  │  └────────────────┘  └──────────────┘  └─────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │           Async Event Dispatcher (事件推送)               │    │
│  │  gdb.events.stop.connect(callback)                       │    │
│  │  → JSON-RPC Notification → Agent                         │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流概览

```
Agent Request Flow (请求流):
  Agent → JSON-RPC Request → Transport Proxy → IPC pipe
    → RPC Listener Thread (解码/校验)
    → Command Queue (线程安全入队)
    → gdb.post_event() (调度到主线程)
    → GDB Main Loop 执行 (gdb.parse_and_eval / gdb.execute)
    → Result → Condition Variable 通知
    → RPC Listener Thread (序列化)
    → JSON-RPC Response → Transport Proxy → Agent

Async Event Flow (异步事件推送流):
  GPU Hardware Trap / Breakpoint Hit
    → gdb.events.stop callback (主线程)
    → 构造 JSON-RPC Notification (含焦点坐标快照)
    → 直接写入 IPC pipe
    → Transport Proxy → Agent
```

### 2.3 技术选型决策

| 决策点       | 选型                                  | 理由                                                            |
| ------------ | ------------------------------------- | --------------------------------------------------------------- |
| 通信协议     | JSON-RPC 2.0 (MCP 兼容)               | LLM 原生理解 JSON Schema，MCP 已成为 Agent 工具互操作的事实标准 |
| 传输层       | stdio + Unix Domain Socket            | stdio 用于 MCP 标准集成；UDS 用于高性能本地场景，避免 TCP 开销  |
| RPC 线程模型 | `gdb.Thread` + `gdb.post_event()` | GDB 内部唯一安全的异步线程方案，自动屏蔽信号干扰                |
| 序列化       | Python 标准库 `json`                | cuda-gdb 内嵌 Python 环境可直接使用，零外部依赖                 |
| 状态管理     | 内嵌状态机 (Finite State Machine)     | 模态守卫需要确定性的状态转移，FSM 是最可靠的实现                |

---

## 3. 第一层：Agent 客户端层

### 3.1 职责边界

Agent 客户端层是**纯无状态**的。它不维护任何底层调试器状态，仅负责：

1. 根据用户自然语言 Prompt 或 Agent 推理链，生成符合 Tool Schema 的 JSON-RPC 请求
2. 接收 JSON-RPC 响应，解析结构化结果用于下一步推理
3. 处理异步 Notification（如断点命中通知），更新 Agent 的上下文认知

### 3.2 MCP Tool 注册规范

Agent 客户端通过 MCP 协议发现可用工具。每个工具的注册信息包含：

```json
{
  "name": "cuda_set_focus",
  "description": "切换 GPU 线程焦点到指定的 Grid/Block/Thread 软件坐标。执行后，所有后续的变量求值和内存读取都将在该线程的栈帧上下文中进行。返回值包含该线程映射到的硬件坐标（SM/Warp/Lane）。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "block": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 3,
        "maxItems": 3,
        "description": "Thread Block 三维坐标 [x, y, z]"
      },
      "thread": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 3,
        "maxItems": 3,
        "description": "Block 内 Thread 三维坐标 [x, y, z]"
      }
    },
    "required": ["block", "thread"]
  }
}
```

### 3.3 Agent 侧的上下文管理策略

由于 LLM 存在上下文窗口限制和"遗忘"倾向，客户端层应遵循以下策略：

- **焦点锚定**：每次收到 `stop` 事件通知后，Agent 必须将通知中携带的焦点坐标作为后续所有操作的前置上下文
- **增量查询**：避免一次性请求全部寄存器或全部线程状态，应按需逐步查询
- **异常优先**：收到异常通知时，优先调用 `cuda_analyze_exception` 获取根因，再决定是否需要切换焦点或读取内存

---

## 4. 第二层：宿主代理传输层

### 4.1 职责定义

宿主代理传输层是一个**独立于 cuda-gdb 进程之外**的轻量级系统进程，承担以下职责：

| 职责             | 说明                                                                      |
| ---------------- | ------------------------------------------------------------------------- |
| 进程生命周期管理 | 启动 `cuda-gdb -x agent_rpc_server.py [target]`，监控其退出码与信号     |
| 传输协议桥接     | 将 MCP Client 的 stdio/SSE 请求转换为内部 IPC 消息                        |
| 崩溃恢复         | 捕获 cuda-gdb 的 SIGSEGV/SIGABRT，记录崩溃日志，向 Agent 返回结构化错误   |
| 超时守卫         | 对每个 RPC 请求设置可配置的超时阈值，防止 GDB 内部死锁导致 Agent 永久等待 |

### 4.2 进程启动与 IPC 通道建立

```
                    Transport Proxy Process
                    ┌─────────────────────────────────────┐
                    │                                     │
  MCP Client ──────►│  stdin/stdout (MCP stdio transport) │
                    │                                     │
                    │  ┌───────────────────────────────┐  │
                    │  │  cuda-gdb subprocess          │  │
                    │  │  ┌─────────────────────────┐  │  │
                    │  │  │ agent_rpc_server.py      │  │  │
                    │  │  │ (Embedded RPC Engine)    │  │  │
                    │  │  └─────────────────────────┘  │  │
                    │  │                               │  │
                    │  │  IPC: Unix Domain Socket      │  │
                    │  │  /tmp/cuda-gdb-agent-{pid}.sock│  │
                    │  └───────────────────────────────┘  │
                    └─────────────────────────────────────┘
```

**启动序列**：

1. Transport Proxy 启动，解析命令行参数确定调试目标（PID 或 Coredump 文件路径）
2. 创建 Unix Domain Socket 监听地址 `/tmp/cuda-gdb-agent-{proxy_pid}.sock`
3. 构造 cuda-gdb 启动命令：
   - Live 模式：`cuda-gdb -x agent_rpc_server.py --args <executable> [args...]` 或 `cuda-gdb -x agent_rpc_server.py -p <pid>`
   - Coredump 模式：`cuda-gdb -x agent_rpc_server.py <executable> -c <coredump_file>`
4. 通过环境变量 `CUDA_GDB_AGENT_SOCKET` 将 UDS 地址传递给内嵌脚本
5. 等待内嵌 RPC 引擎通过 UDS 发送 `{"jsonrpc":"2.0","method":"__rpc_ready","params":{}}` 握手消息
6. 握手成功后，Transport Proxy 开始接受 MCP Client 的请求

### 4.3 超时与崩溃处理

```
超时处理流程:

  Agent Request ──► Transport Proxy
                    │
                    ├── 启动计时器 (默认 30s, 可配置)
                    │
                    ├── 转发至 cuda-gdb IPC
                    │
                    ├── 等待响应...
                    │
                    ├── [正常] 收到响应 → 取消计时器 → 返回 Agent
                    │
                    └── [超时] 计时器触发
                         ├── 向 cuda-gdb 发送 SIGINT (尝试中断阻塞命令)
                         ├── 等待 5s 宽限期
                         ├── [恢复] 收到响应 → 返回 Agent (附加 timeout_warning)
                         └── [仍无响应] 返回 JSON-RPC Error:
                             {
                               "code": -32001,
                               "message": "Command timed out after 30s",
                               "data": {
                                 "hint": "The debugger may be blocked by a long-running kernel. Consider using cuda_execution_control with action='interrupt'."
                               }
                             }
```

**崩溃恢复流程**：

```
cuda-gdb 进程异常退出:

  Transport Proxy 检测到子进程退出
    │
    ├── 读取退出码与信号编号
    │
    ├── 收集 stderr 最后 100 行作为崩溃上下文
    │
    ├── 向所有挂起的 RPC 请求返回错误:
    │   {
    │     "code": -32002,
    │     "message": "cuda-gdb process crashed",
    │     "data": {
    │       "exit_signal": "SIGSEGV",
    │       "last_stderr": "...",
    │       "recovery_hint": "The debugger crashed. If debugging a coredump, reload with cuda_init. If live debugging, the target process may also have terminated."
    │     }
    │   }
    │
    └── 进入 CRASHED 状态，等待 Agent 发送 cuda_init 重新初始化
```

### 4.4 配置参数

| 参数名                          | 类型   | 默认值                             | 说明                                            |
| ------------------------------- | ------ | ---------------------------------- | ----------------------------------------------- |
| `CUDA_GDB_PATH`               | string | `cuda-gdb`                       | cuda-gdb 可执行文件路径                         |
| `CUDA_GDB_AGENT_SOCKET`       | string | `/tmp/cuda-gdb-agent-{pid}.sock` | IPC 套接字路径                                  |
| `RPC_TIMEOUT_SECONDS`         | int    | 30                                 | 单个 RPC 请求的超时时间                         |
| `RPC_INTERRUPT_GRACE_SECONDS` | int    | 5                                  | SIGINT 后的宽限等待时间                         |
| `MAX_RESPONSE_SIZE_BYTES`     | int    | 1048576 (1MB)                      | 单个响应的最大体积，防止大数组撑爆 Agent 上下文 |
| `LOG_LEVEL`                   | string | `INFO`                           | 日志级别                                        |
| `CUDA_GDB_EXTRA_ARGS`         | string | `""`                             | 传递给 cuda-gdb 的额外启动参数                  |

---

## 5. 第三层：内嵌式 Python RPC 引擎

### 5.1 模块架构

内嵌式 Python RPC 引擎是整个系统的**核心心脏**，通过 `cuda-gdb -x agent_rpc_server.py` 注入到调试器进程内部运行。

```
agent_rpc_server.py 内部模块结构:

┌─────────────────────────────────────────────────────────┐
│                    Entry Point                          │
│  - 解析环境变量                                          │
│  - 探测调试模态 (Live / Coredump)                        │
│  - 初始化状态机                                          │
│  - 启动 RPC Listener Thread                             │
│  - 注册 gdb.events 回调                                 │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│ rpc_listener│  │ modality_    │  │ tool_handlers    │
│             │  │ guard        │  │                  │
│ - Socket IO │  │              │  │ - set_focus()    │
│ - JSON-RPC  │  │ - FSM        │  │ - evaluate_var() │
│   decode    │  │ - Mode detect│  │ - dump_regs()    │
│ - Request   │  │ - Permission │  │ - analyze_exc()  │
│   dispatch  │  │   check      │  │ - exec_control() │
│ - Response  │  │ - Focus      │  │ - read_memory()  │
│   encode    │  │   tracker    │  │ - list_kernels() │
└──────┬──────┘  └──────┬───────┘  └────────┬─────────┘
       │                │                    │
       └────────────────┴──────┬─────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   gdb_safe_executor │
                    │                     │
                    │ - post_event()      │
                    │ - Condition wait    │
                    │ - Exception capture │
                    │ - Value serialize   │
                    └─────────────────────┘
```

### 5.2 线程隔离与无阻塞通信机制

这是整个架构中最关键的设计，解决实时调试模式下 `gdb.execute("continue")` 阻塞 Python GIL 导致 RPC 通道死锁的核心问题。

#### 5.2.1 线程模型

```
┌─────────────────────────────────────────────────────────────┐
│                    cuda-gdb Process                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  GDB Main Thread (主线程)                            │    │
│  │                                                     │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │  GDB Event Loop                             │    │    │
│  │  │                                             │    │    │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │    │
│  │  │  │ Breakpt  │  │ post_    │  │ Signal   │  │    │    │
│  │  │  │ Handler  │  │ event()  │  │ Handler  │  │    │    │
│  │  │  │ Callback │  │ Tasks    │  │          │  │    │    │
│  │  │  └──────────┘  └──────────┘  └──────────┘  │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  RPC Listener Thread (gdb.Thread 子类)               │    │
│  │  - 自动调用 gdb.block_signals()                      │    │
│  │  - 独立的 Socket I/O 循环                             │    │
│  │  - 不直接调用任何 gdb.execute() / gdb.parse_and_eval()│    │
│  │  - 通过 Command Queue 与主线程通信                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  通信机制:                                                   │
│  RPC Thread ──[queue.put()]──► Command Queue                │
│                                     │                       │
│                          gdb.post_event(dequeue_and_exec)   │
│                                     │                       │
│  RPC Thread ◄──[Condition.notify()]─┘                       │
└─────────────────────────────────────────────────────────────┘
```

#### 5.2.2 命令执行的完整生命周期

```python
# 伪代码：展示一个 RPC 请求从接收到返回的完整流程

class CommandTask:
    """封装一个待执行的 GDB 命令任务"""
    def __init__(self, request_id, method, params):
        self.request_id = request_id
        self.method = method
        self.params = params
        self.result = None
        self.error = None
        self.completed = threading.Event()  # 完成信号

class RPCListenerThread(gdb.Thread):
    """RPC 监听线程 - 运行在独立线程中"""

    def __init__(self, socket_path, command_queue):
        super().__init__()  # gdb.Thread 自动调用 gdb.block_signals()
        self.socket_path = socket_path
        self.command_queue = command_queue
        self.daemon = True

    def run(self):
        # 建立 UDS 连接
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(self.socket_path)
        server_socket.listen(1)

        # 发送就绪握手
        self._send_ready_notification()

        while True:
            conn, _ = server_socket.accept()
            self._handle_connection(conn)

    def _handle_connection(self, conn):
        while True:
            raw_data = self._read_message(conn)
            if not raw_data:
                break

            request = json.loads(raw_data)
            task = CommandTask(
                request_id=request.get("id"),
                method=request["method"],
                params=request.get("params", {})
            )

            # 将任务放入队列，并通过 post_event 通知主线程
            self.command_queue.put(task)
            gdb.post_event(lambda: self._process_next_task())

            # 等待主线程执行完毕
            task.completed.wait(timeout=25)  # 内部超时略小于外部超时

            # 构造 JSON-RPC 响应
            if task.error:
                response = {
                    "jsonrpc": "2.0",
                    "id": task.request_id,
                    "error": task.error
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": task.request_id,
                    "result": task.result
                }

            self._send_message(conn, json.dumps(response))

    def _process_next_task(self):
        """此方法通过 gdb.post_event() 在 GDB 主线程中执行"""
        if self.command_queue.empty():
            return

        task = self.command_queue.get()
        try:
            handler = TOOL_HANDLERS.get(task.method)
            if handler is None:
                task.error = {
                    "code": -32601,
                    "message": f"Unknown method: {task.method}"
                }
            else:
                task.result = handler(task.params)
        except gdb.error as gdb_err:
            task.error = {
                "code": -32000,
                "message": str(gdb_err),
                "data": {"source": "gdb_internal"}
            }
        except Exception as exc:
            task.error = {
                "code": -32603,
                "message": str(exc),
                "data": {"source": "rpc_engine"}
            }
        finally:
            task.completed.set()  # 通知 RPC 线程任务已完成
```

#### 5.2.3 阻塞命令的特殊处理

对于 `continue`、`step`、`next` 等会阻塞 GDB 主线程的命令，采用**异步执行 + 事件回调**模式：

```
Agent 发送 cuda_execution_control(action="continue"):

  1. RPC Thread 收到请求
  2. 通过 post_event 调度到主线程
  3. 主线程执行:
     a. 立即返回 {"status": "running", "blocked": true}
     b. 调用 gdb.execute("continue", to_string=True)
        (此调用会阻塞主线程直到程序停止)
  4. RPC Thread 收到立即返回的响应，转发给 Agent
  5. Agent 知道程序正在运行中

  ... 程序运行中 ...

  6. 程序命中断点或异常
  7. gdb.events.stop 回调触发 (在主线程中)
  8. 回调函数构造 Notification:
     {
       "jsonrpc": "2.0",
       "method": "cuda_stop_event",
       "params": {
         "reason": "breakpoint",
         "breakpoint_id": 3,
         "current_focus": {
           "kernel": "matmul_kernel",
           "block": [2, 0, 0],
           "thread": [15, 0, 0],
           "hardware": {"device": 0, "sm": 7, "warp": 3, "lane": 15}
         },
         "pc": "0x555555557a80",
         "source_location": {
           "file": "matmul.cu",
           "line": 42
         }
       }
     }
  9. Notification 通过 IPC 推送给 Transport Proxy → Agent
```

**关键实现细节**：

```python
# 阻塞命令的异步执行实现

class ExecutionController:
    """处理 continue/step/next 等阻塞命令"""

    def __init__(self, notification_sender):
        self.notification_sender = notification_sender
        self.is_running = False

    def handle_execution_control(self, params):
        action = params["action"]

        # 模态守卫检查
        if ModalityGuard.current_mode == "IMMUTABLE":
            return {
                "error": {
                    "code": -32003,
                    "message": "Coredump Mode: execution control is forbidden",
                    "data": {
                        "mode": "IMMUTABLE",
                        "hint": "This is a post-mortem coredump analysis session. "
                                "Execution control commands (continue, step, next) "
                                "are not available. Use read-only inspection tools."
                    }
                }
            }

        if self.is_running:
            if action == "interrupt":
                gdb.execute("interrupt", to_string=True)
                return {"status": "interrupt_sent"}
            return {
                "error": {
                    "code": -32004,
                    "message": "Target is already running",
                    "data": {"hint": "Send action='interrupt' to pause execution."}
                }
            }

        self.is_running = True

        # 立即返回"已启动"状态
        # 实际执行在 post_event 中异步进行
        def async_execute():
            try:
                gdb.execute(action, to_string=True)
            except gdb.error as err:
                self.notification_sender({
                    "method": "cuda_error_event",
                    "params": {"error": str(err)}
                })
            finally:
                self.is_running = False

        gdb.post_event(async_execute)

        return {"status": "running", "action": action, "blocked": True}
```

### 5.3 异步事件推送机制

```python
# Stop 事件回调注册与处理

class AsyncEventDispatcher:
    """异步事件分发器 - 将 GDB 内部事件转换为 JSON-RPC Notification"""

    def __init__(self, notification_channel):
        self.channel = notification_channel
        # 注册 GDB 事件回调
        gdb.events.stop.connect(self._on_stop)
        gdb.events.exited.connect(self._on_exit)

    def _on_stop(self, event):
        """程序停止时的回调 - 在 GDB 主线程中执行"""
        notification = {
            "jsonrpc": "2.0",
            "method": "cuda_stop_event",
            "params": self._build_stop_context(event)
        }
        self.channel.send(json.dumps(notification))

    def _build_stop_context(self, event):
        """构建停止事件的完整上下文"""
        context = {}

        # 1. 停止原因
        if isinstance(event, gdb.BreakpointEvent):
            context["reason"] = "breakpoint"
            context["breakpoint_id"] = event.breakpoints[0].number
        elif isinstance(event, gdb.SignalEvent):
            context["reason"] = "signal"
            context["signal_name"] = event.stop_signal
        else:
            context["reason"] = "unknown"

        # 2. 强制附加当前焦点坐标 (防止 Agent 上下文遗忘)
        try:
            focus_output = gdb.execute("cuda kernel block thread", to_string=True)
            context["current_focus"] = self._parse_focus_output(focus_output)
        except gdb.error:
            context["current_focus"] = None
            context["focus_warning"] = "Unable to determine GPU focus. May be on host code."

        # 3. 程序计数器
        try:
            pc_value = gdb.parse_and_eval("$pc")
            context["pc"] = hex(int(pc_value))
        except gdb.error:
            pass

        # 4. 源码位置
        try:
            frame = gdb.selected_frame()
            sal = frame.find_sal()
            if sal.symtab:
                context["source_location"] = {
                    "file": sal.symtab.filename,
                    "line": sal.line
                }
        except gdb.error:
            pass

        # 5. CUDA 异常检测
        try:
            exception_info = self._detect_cuda_exception()
            if exception_info:
                context["cuda_exception"] = exception_info
        except gdb.error:
            pass

        return context

    def _detect_cuda_exception(self):
        """检测当前是否存在 CUDA 硬件异常"""
        try:
            # 尝试读取 errorpc - 如果存在说明有硬件异常
            errorpc = gdb.parse_and_eval("$errorpc")
            if errorpc is not None:
                return {
                    "errorpc": hex(int(errorpc)),
                    "hint": "CUDA hardware exception detected. "
                            "Call cuda_analyze_exception for detailed analysis."
                }
        except gdb.error:
            pass
        return None

    def _on_exit(self, event):
        """程序退出时的回调"""
        notification = {
            "jsonrpc": "2.0",
            "method": "cuda_exit_event",
            "params": {
                "exit_code": event.exit_code if hasattr(event, 'exit_code') else None
            }
        }
        self.channel.send(json.dumps(notification))
```

### 5.4 gdb.Value 到 JSON 的序列化引擎

`gdb.Value` 是 GDB Python API 返回的核心对象，它封装了调试目标内存中的值。将其安全、准确地转换为 JSON 是 RPC 引擎的基础能力。

```python
class GdbValueSerializer:
    """将 gdb.Value 对象安全地序列化为 JSON 兼容的 Python 对象"""

    MAX_ARRAY_ELEMENTS = 256  # 防止大数组撑爆 Agent 上下文
    MAX_STRING_LENGTH = 4096  # 字符串截断阈值

    @staticmethod
    def serialize(gdb_value, depth=0, max_depth=5):
        """
        递归序列化 gdb.Value 到 JSON 兼容结构。

        返回格式:
        {
            "value": <实际值>,
            "type": <类型字符串>,
            "address": <内存地址, 如果可用>,
            "meta": {<元信息, 如 optimized_out 标记>}
        }
        """
        if depth > max_depth:
            return {"value": "<max_depth_exceeded>", "type": "truncated"}

        result = {"type": str(gdb_value.type)}

        # 检查是否被优化掉
        if gdb_value.is_optimized_out:
            result["value"] = None
            result["meta"] = {
                "optimized_out": True,
                "hint": "Variable was optimized out by the compiler. "
                        "Recompile with -g -G flags to preserve debug info."
            }
            return result

        # 获取地址 (如果可用)
        try:
            if gdb_value.address is not None:
                result["address"] = hex(int(gdb_value.address))
        except gdb.error:
            pass

        type_code = gdb_value.type.strip_typedefs().code

        # 基本整数类型
        if type_code in (gdb.TYPE_CODE_INT, gdb.TYPE_CODE_ENUM,
                         gdb.TYPE_CODE_CHAR, gdb.TYPE_CODE_BOOL):
            try:
                int_val = int(gdb_value)
                result["value"] = int_val
                result["hex"] = hex(int_val)
            except (gdb.error, OverflowError):
                result["value"] = str(gdb_value)
            return result

        # 浮点类型
        if type_code == gdb.TYPE_CODE_FLT:
            try:
                result["value"] = float(gdb_value)
            except (gdb.error, ValueError):
                result["value"] = str(gdb_value)
            return result

        # 指针类型
        if type_code == gdb.TYPE_CODE_PTR:
            try:
                ptr_val = int(gdb_value)
                result["value"] = hex(ptr_val)
                if ptr_val == 0:
                    result["meta"] = {"null_pointer": True}
            except (gdb.error, OverflowError):
                result["value"] = str(gdb_value)
            return result

        # 数组类型
        if type_code == gdb.TYPE_CODE_ARRAY:
            try:
                array_type = gdb_value.type.strip_typedefs()
                range_type = array_type.range()
                length = range_type[1] - range_type[0] + 1
                actual_length = min(length, GdbValueSerializer.MAX_ARRAY_ELEMENTS)

                elements = []
                for i in range(actual_length):
                    elem = GdbValueSerializer.serialize(
                        gdb_value[i], depth + 1, max_depth
                    )
                    elements.append(elem)

                result["value"] = elements
                result["meta"] = {
                    "total_length": length,
                    "displayed_length": actual_length,
                    "truncated": length > actual_length
                }
            except gdb.error as err:
                result["value"] = None
                result["meta"] = {"read_error": str(err)}
            return result

        # 结构体/联合体
        if type_code in (gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION):
            try:
                fields = {}
                for field in gdb_value.type.fields():
                    try:
                        field_val = gdb_value[field.name]
                        fields[field.name] = GdbValueSerializer.serialize(
                            field_val, depth + 1, max_depth
                        )
                    except gdb.error as err:
                        fields[field.name] = {
                            "value": None,
                            "meta": {"read_error": str(err)}
                        }
                result["value"] = fields
            except gdb.error as err:
                result["value"] = None
                result["meta"] = {"read_error": str(err)}
            return result

        # 兜底：使用 GDB 的字符串表示
        try:
            str_val = str(gdb_value)
            if len(str_val) > GdbValueSerializer.MAX_STRING_LENGTH:
                str_val = str_val[:GdbValueSerializer.MAX_STRING_LENGTH]
                result["meta"] = {"string_truncated": True}
            result["value"] = str_val
        except gdb.error:
            result["value"] = "<unreadable>"
        return result
```

---

## 6. GPU 特有对象访问机制设计

### 6.1 共享内存访问器

共享内存（Shared Memory）位于 SM 内部的片上 SRAM，其地址空间独立于全局显存。直接通过 `gdb.parse_and_eval()` 读取共享内存指针时，若不加 `@shared` 地址空间修饰符，GDB 会将其误解为全局地址，导致越界或脏数据。

```python
class SharedMemoryAccessor:
    """
    共享内存安全访问器。
    自动注入 @shared 地址空间修饰符，处理 IPC 内存访问拒绝异常。
    """

    # CUDA IPC 内存访问拒绝的特征错误信息
    IPC_REJECTION_PATTERNS = [
        "Cannot access memory imported via CUDA IPC",
        "IPC memory access denied",
    ]

    @staticmethod
    def read_by_variable(variable_name, array_length=None):
        """
        通过变量名读取共享内存。

        参数:
            variable_name: 共享内存变量名 (如 "s_data")
            array_length: 如果是数组，指定读取长度

        返回:
            序列化后的 JSON 兼容结构
        """
        try:
            if array_length and array_length > 0:
                # 使用 GDB 的数组切片语法
                expression = f"({variable_name})@{array_length}"
            else:
                expression = variable_name

            gdb_value = gdb.parse_and_eval(expression)
            return {
                "status": "ok",
                "memory_space": "shared",
                "data": GdbValueSerializer.serialize(gdb_value)
            }
        except gdb.error as err:
            return SharedMemoryAccessor._handle_error(err)

    @staticmethod
    def read_by_address(address, data_type, count=1):
        """
        通过物理地址读取共享内存。
        自动注入 @shared 修饰符。

        参数:
            address: 十六进制地址字符串 (如 "0x20") 或整数
            data_type: C 类型字符串 (如 "int", "float", "double")
            count: 读取元素数量

        返回:
            序列化后的 JSON 兼容结构
        """
        if isinstance(address, int):
            address = hex(address)

        try:
            if count > 1:
                # 读取连续数组: *((@shared TYPE*)ADDR)@COUNT
                expression = f"*((@shared {data_type}*){address})@{count}"
            else:
                # 读取单个值: *(@shared TYPE*)ADDR
                expression = f"*(@shared {data_type}*){address}"

            gdb_value = gdb.parse_and_eval(expression)
            return {
                "status": "ok",
                "memory_space": "shared",
                "address": address,
                "data_type": data_type,
                "count": count,
                "data": GdbValueSerializer.serialize(gdb_value)
            }
        except gdb.error as err:
            return SharedMemoryAccessor._handle_error(err)

    @staticmethod
    def _handle_error(err):
        """统一的共享内存错误处理"""
        error_msg = str(err)

        # 检测 IPC 内存访问拒绝
        for pattern in SharedMemoryAccessor.IPC_REJECTION_PATTERNS:
            if pattern in error_msg:
                return {
                    "status": "error",
                    "error_type": "ipc_access_denied",
                    "message": error_msg,
                    "hint": "This shared memory allocation was imported via "
                            "CUDA IPC (inter-process communication). cuda-gdb "
                            "explicitly prohibits accessing IPC-imported memory "
                            "for security reasons."
                }

        # 检测地址越界
        if "Address out of bounds" in error_msg or "out of bounds" in error_msg:
            return {
                "status": "error",
                "error_type": "address_out_of_bounds",
                "message": error_msg,
                "hint": "The address may be outside the shared memory "
                        "allocation for the current thread block. Verify "
                        "the block dimensions and shared memory size."
            }

        # 通用错误
        return {
            "status": "error",
            "error_type": "shared_memory_read_failed",
            "message": error_msg
        }
```

### 6.2 寄存器安全探测器

CUDA 硬件寄存器（`$R0`-`$R255`、`$P0`-`$P6`、`$CC`）的访问存在严格的硬件边界。盲目遍历所有 256 个通用寄存器会导致访问未分配的 SRAM 组，引发段错误或调试器挂起。

```python
class RegisterProbe:
    """
    CUDA 硬件寄存器安全探测器。
    先确定当前 Warp 的实际寄存器分配上限，再安全地收集寄存器状态。
    """

    # 谓词寄存器固定为 P0-P6
    PREDICATE_REGISTER_COUNT = 7

    @staticmethod
    def dump_warp_registers():
        """
        安全地转储当前聚焦 Warp 的所有已分配寄存器。

        返回:
        {
            "status": "ok",
            "warp_info": {"sm": 0, "warp": 12},
            "general_registers": {"R0": "0x00000042", "R1": "0x00000000", ...},
            "predicate_registers": {"P0": "0x1", "P1": "0x0", ...},
            "special_registers": {"CC": "0x0"},
            "register_count": 64,
            "max_possible": 255
        }
        """
        try:
            # 第一步：确定当前 Warp 的实际寄存器分配数量
            max_reg_index = RegisterProbe._detect_register_limit()

            # 第二步：安全地收集通用寄存器
            general_regs = {}
            for i in range(max_reg_index + 1):
                try:
                    val = gdb.parse_and_eval(f"$R{i}")
                    general_regs[f"R{i}"] = hex(int(val))
                except gdb.error:
                    # 到达实际边界，停止遍历
                    break

            # 第三步：收集谓词寄存器 (P0-P6 固定)
            predicate_regs = {}
            for i in range(RegisterProbe.PREDICATE_REGISTER_COUNT):
                try:
                    val = gdb.parse_and_eval(f"$P{i}")
                    predicate_regs[f"P{i}"] = hex(int(val))
                except gdb.error:
                    break

            # 第四步：收集条件码寄存器
            special_regs = {}
            try:
                cc_val = gdb.parse_and_eval("$CC")
                special_regs["CC"] = hex(int(cc_val))
            except gdb.error:
                pass

            # 第五步：获取 Warp 硬件坐标
            warp_info = RegisterProbe._get_warp_info()

            return {
                "status": "ok",
                "warp_info": warp_info,
                "general_registers": general_regs,
                "predicate_registers": predicate_regs,
                "special_registers": special_regs,
                "register_count": len(general_regs),
                "max_possible": 255
            }

        except gdb.error as err:
            return {
                "status": "error",
                "message": str(err),
                "hint": "Failed to probe registers. Ensure a valid GPU "
                        "thread is in focus and the kernel is active."
            }

    @staticmethod
    def _detect_register_limit():
        """
        检测当前 Warp 的实际寄存器分配上限。

        策略：
        1. 优先通过 'info registers system' 解析输出确定上限
        2. 回退方案：二分搜索法探测可访问的最大寄存器索引
        """
        # 策略 1：解析 info registers 输出
        try:
            reg_info = gdb.execute("info registers system", to_string=True)
            # 解析输出中最大的 R 寄存器编号
            max_index = 0
            for line in reg_info.split('\n'):
                line = line.strip()
                # 匹配形如 "R123  0x..." 的行
                if line.startswith('R') and line[1:].split()[0].isdigit():
                    index = int(line[1:].split()[0])
                    max_index = max(max_index, index)
            if max_index > 0:
                return max_index
        except gdb.error:
            pass

        # 策略 2：二分搜索法
        low, high = 0, 255
        last_valid = 0
        while low <= high:
            mid = (low + high) // 2
            try:
                gdb.parse_and_eval(f"$R{mid}")
                last_valid = mid
                low = mid + 1
            except gdb.error:
                high = mid - 1
        return last_valid

    @staticmethod
    def _get_warp_info():
        """获取当前聚焦的 Warp 硬件坐标"""
        try:
            output = gdb.execute("cuda sm warp lane", to_string=True)
            # 解析输出提取 SM/Warp/Lane 信息
            info = {}
            for token in output.split():
                if token.startswith("sm"):
                    info["sm"] = int(token.split()[-1]) if ' ' in token else None
                elif token.startswith("warp"):
                    info["warp"] = int(token.split()[-1]) if ' ' in token else None
            return info
        except gdb.error:
            return {}
```

### 6.3 CUDA 异常分析器

当 GPU 内核崩溃时，异常在 Warp 级别被硬件陷入（Trap）并抛出。异常分析器将底层的 `CUDA_EXCEPTION` 枚举码翻译为 Agent 可理解的语义描述。

```python
class CudaExceptionAnalyzer:
    """
    CUDA Warp 级异常分析器。
    将硬件异常码映射为结构化的语义描述，
    并提取触发异常的精确 SASS 指令。
    """

    # CUDA_EXCEPTION 枚举到语义描述的映射
    EXCEPTION_MAP = {
        "CUDA_EXCEPTION_1":  {
            "name": "Lane Illegal Address",
            "severity": "critical",
            "description": "A single lane within the warp attempted to access "
                          "an illegal memory address.",
            "common_causes": [
                "Array index out of bounds",
                "Dereferencing a null or dangling pointer",
                "Stack buffer overflow in device code"
            ]
        },
        "CUDA_EXCEPTION_2":  {
            "name": "Lane User Stack Overflow",
            "severity": "critical",
            "description": "A lane's call stack exceeded the allocated stack size.",
            "common_causes": [
                "Deep recursion in device code",
                "Large local arrays exceeding stack allocation",
                "Insufficient cudaLimitStackSize setting"
            ]
        },
        "CUDA_EXCEPTION_3":  {
            "name": "Device Hardware Stack Overflow",
            "severity": "critical",
            "description": "The hardware call/return stack overflowed.",
            "common_causes": [
                "Extremely deep function call chains",
                "Recursive kernel launches (Dynamic Parallelism)"
            ]
        },
        "CUDA_EXCEPTION_4":  {
            "name": "Warp Illegal Instruction",
            "severity": "critical",
            "description": "The warp encountered an illegal or undefined instruction.",
            "common_causes": [
                "Corrupted device code",
                "JIT compilation failure",
                "Architecture mismatch (running SM_80 code on SM_70 device)"
            ]
        },
        "CUDA_EXCEPTION_5":  {
            "name": "Warp Out-of-Range Address",
            "severity": "critical",
            "description": "The warp attempted to access an address outside "
                          "any valid memory region.",
            "common_causes": [
                "Accessing freed device memory (use-after-free)",
                "Integer overflow in address calculation",
                "Uninitialized pointer dereference"
            ]
        },
        "CUDA_EXCEPTION_6":  {
            "name": "Warp Misaligned Address",
            "severity": "warning",
            "description": "The warp performed a memory access that was not "
                          "properly aligned for the data type.",
            "common_causes": [
                "Casting pointers between types with different alignment requirements",
                "Packed struct access without proper alignment attributes"
            ]
        },
        "CUDA_EXCEPTION_7":  {
            "name": "Warp Invalid Address Space",
            "severity": "critical",
            "description": "The warp attempted to access memory in an invalid "
                          "address space (e.g., shared memory address used as global).",
            "common_causes": [
                "Passing shared memory pointer to a function expecting global memory",
                "Address space confusion in generic pointer operations"
            ]
        },
        "CUDA_EXCEPTION_8":  {
            "name": "Warp Invalid Program Counter",
            "severity": "critical",
            "description": "The warp's program counter jumped to an invalid address.",
            "common_causes": [
                "Corrupted function pointer",
                "Virtual function table corruption",
                "Stack smashing overwriting return address"
            ]
        },
        "CUDA_EXCEPTION_14": {
            "name": "Warp Illegal Address",
            "severity": "critical",
            "description": "Any lane within the warp accessed an illegal memory "
                          "address. This is the most common CUDA memory error.",
            "common_causes": [
                "Global memory buffer overflow",
                "Accessing device memory after cudaFree()",
                "Race condition corrupting pointer values",
                "Incorrect grid/block dimension causing out-of-bounds thread indices"
            ]
        },
        "CUDA_EXCEPTION_32": {
            "name": "Warp Shared Memory Issue",
            "severity": "critical",
            "description": "An uncorrectable error occurred during cluster-level "
                          "shared memory access.",
            "common_causes": [
                "Hardware ECC error in shared memory",
                "Cluster-level distributed shared memory access violation"
            ]
        },
        "CUDA_EXCEPTION_35": {
            "name": "Warp User Stack Overflow",
            "severity": "critical",
            "description": "The warp's user-level stack overflowed during dynamic "
                          "allocation or deep call chains.",
            "common_causes": [
                "Recursive device functions without proper depth limits",
                "alloca() or variable-length arrays exceeding stack space",
                "Insufficient per-thread stack size (cudaDeviceSetLimit)"
            ]
        },
    }

    @staticmethod
    def analyze():
        """
        分析当前的 CUDA 异常状态。

        返回完整的异常上下文，包括：
        - 异常类型与语义描述
        - errorpc 与 pc 的值
        - 触发异常的精确 SASS 指令
        - 当前焦点坐标
        - 可能的根因提示
        """
        result = {}

        # 1. 获取 errorpc (异常触发点的精确程序计数器)
        try:
            errorpc = gdb.parse_and_eval("$errorpc")
            result["errorpc"] = hex(int(errorpc))
        except gdb.error:
            return {
                "status": "no_exception",
                "message": "No CUDA exception detected in current context. "
                           "$errorpc is not available."
            }

        # 2. 获取当前 pc
        try:
            pc = gdb.parse_and_eval("$pc")
            result["pc"] = hex(int(pc))
        except gdb.error:
            result["pc"] = None

        # 3. 反汇编 errorpc 附近的指令，定位触发异常的 SASS 指令
        try:
            disasm_output = gdb.execute(
                f"disassemble {result['errorpc']},{result['errorpc']}+32",
                to_string=True
            )
            faulting_instruction = None
            for line in disasm_output.split('\n'):
                # errorpc 对应的指令通常以 *> 或 => 标记
                if '*>' in line or '=>' in line:
                    faulting_instruction = line.strip()
                    break
            if faulting_instruction is None and disasm_output.strip():
                # 取第一条指令作为近似
                lines = [l.strip() for l in disasm_output.split('\n') if l.strip()]
                if lines:
                    faulting_instruction = lines[0]
            result["faulting_instruction"] = faulting_instruction
        except gdb.error:
            result["faulting_instruction"] = None

        # 4. 检测异常类型
        try:
            # 通过 info cuda exception 或停止原因获取异常码
            exception_output = gdb.execute("info cuda exception", to_string=True)
            exception_code = CudaExceptionAnalyzer._parse_exception_code(
                exception_output
            )
            if exception_code and exception_code in CudaExceptionAnalyzer.EXCEPTION_MAP:
                exception_info = CudaExceptionAnalyzer.EXCEPTION_MAP[exception_code]
                result["exception_code"] = exception_code
                result["exception_name"] = exception_info["name"]
                result["severity"] = exception_info["severity"]
                result["description"] = exception_info["description"]
                result["common_causes"] = exception_info["common_causes"]
            else:
                result["exception_code"] = exception_code or "UNKNOWN"
                result["description"] = "Unknown CUDA exception type"
        except gdb.error:
            result["exception_code"] = "DETECTION_FAILED"

        # 5. 附加当前焦点坐标
        try:
            focus_output = gdb.execute("cuda kernel block thread", to_string=True)
            result["focus_at_exception"] = focus_output.strip()
        except gdb.error:
            pass

        # 6. 尝试获取相关寄存器快照 (用于地址计算反推)
        try:
            key_registers = {}
            for reg_name in ["$R0", "$R1", "$R2", "$R3", "$R4", "$R5"]:
                try:
                    val = gdb.parse_and_eval(reg_name)
                    key_registers[reg_name.replace("$", "")] = hex(int(val))
                except gdb.error:
                    break
            if key_registers:
                result["key_registers_snapshot"] = key_registers
        except gdb.error:
            pass

        result["status"] = "exception_detected"
        return result

    @staticmethod
    def _parse_exception_code(output):
        """从 'info cuda exception' 输出中解析异常码"""
        # 匹配形如 "CUDA_EXCEPTION_14" 的模式
        import re
        match = re.search(r'CUDA_EXCEPTION_(\d+)', output)
        if match:
            return f"CUDA_EXCEPTION_{match.group(1)}"
        return None
```

### 6.4 焦点管理器

GPU 线程焦点（Focus）的切换是所有后续操作的前提。焦点管理器封装了底层的 `cuda thread` 命令，并提供坐标校验与硬件映射。

```python
class CudaFocusManager:
    """
    GPU 线程焦点管理器。
    封装 cuda thread/block/kernel 命令，提供坐标校验与硬件映射。
    """

    @staticmethod
    def set_focus(params):
        """
        切换 GPU 线程焦点。

        参数:
            params: {
                "block": [x, y, z],    # Thread Block 坐标
                "thread": [x, y, z],   # Block 内 Thread 坐标
                "kernel": int          # 可选，Kernel 编号
            }

        返回:
            {
                "status": "ok",
                "software_coords": {...},
                "hardware_mapping": {"device": 0, "sm": 7, "warp": 3, "lane": 15}
            }
        """
        block = params.get("block", [0, 0, 0])
        thread = params.get("thread", [0, 0, 0])
        kernel = params.get("kernel")

        # 构造 cuda 命令
        commands = []
        if kernel is not None:
            commands.append(f"cuda kernel {kernel}")
        commands.append(
            f"cuda block {block[0]},{block[1]},{block[2]} "
            f"thread {thread[0]},{thread[1]},{thread[2]}"
        )

        try:
            for cmd in commands:
                gdb.execute(cmd, to_string=True)

            # 获取硬件映射
            hardware_mapping = CudaFocusManager._get_hardware_mapping()

            # 验证焦点是否成功切换
            verification = CudaFocusManager._verify_focus(block, thread)

            return {
                "status": "ok",
                "software_coords": {
                    "block": block,
                    "thread": thread,
                    "kernel": kernel
                },
                "hardware_mapping": hardware_mapping,
                "verification": verification
            }

        except gdb.error as err:
            error_msg = str(err)

            # 检测常见的焦点切换失败原因
            if "not within" in error_msg or "invalid" in error_msg.lower():
                return {
                    "status": "error",
                    "error_type": "invalid_coordinates",
                    "message": error_msg,
                    "hint": "The specified block/thread coordinates are outside "
                            "the active grid dimensions. Use cuda_list_kernels "
                            "to check the current grid configuration."
                }
            if "no active" in error_msg.lower() or "no kernel" in error_msg.lower():
                return {
                    "status": "error",
                    "error_type": "no_active_kernel",
                    "message": error_msg,
                    "hint": "No CUDA kernel is currently active on the GPU. "
                            "The program may be executing host (CPU) code. "
                            "Set a breakpoint inside a __global__ function first."
                }
            return {
                "status": "error",
                "error_type": "focus_switch_failed",
                "message": error_msg
            }

    @staticmethod
    def _get_hardware_mapping():
        """获取当前焦点的硬件坐标映射"""
        mapping = {}
        try:
            output = gdb.execute("cuda device sm warp lane", to_string=True)
            # 解析输出
            import re
            for key in ["device", "sm", "warp", "lane"]:
                match = re.search(rf'{key}\s+(\d+)', output, re.IGNORECASE)
                if match:
                    mapping[key] = int(match.group(1))
        except gdb.error:
            pass
        return mapping

    @staticmethod
    def _verify_focus(expected_block, expected_thread):
        """验证焦点是否成功切换到预期坐标"""
        try:
            thread_x = int(gdb.parse_and_eval("threadIdx.x"))
            thread_y = int(gdb.parse_and_eval("threadIdx.y"))
            thread_z = int(gdb.parse_and_eval("threadIdx.z"))
            block_x = int(gdb.parse_and_eval("blockIdx.x"))
            block_y = int(gdb.parse_and_eval("blockIdx.y"))
            block_z = int(gdb.parse_and_eval("blockIdx.z"))

            actual_thread = [thread_x, thread_y, thread_z]
            actual_block = [block_x, block_y, block_z]

            return {
                "verified": (actual_thread == expected_thread
                             and actual_block == expected_block),
                "actual_thread": actual_thread,
                "actual_block": actual_block
            }
        except gdb.error:
            return {"verified": False, "reason": "Unable to read threadIdx/blockIdx"}
```

---

## 7. 模态守卫与状态机设计

### 7.1 调试模态状态机

```
                    ┌──────────────────────────────────────┐
                    │         Modality Guard FSM           │
                    └──────────────────────────────────────┘

  ┌─────────────┐     detect live target     ┌──────────────┐
  │             │ ──────────────────────────► │              │
  │ INITIALIZING│                             │  MUTABLE     │
  │             │     detect cudacore file    │  (Live Mode) │
  │             │ ──────────┐                 │              │
  └─────────────┘           │                 └──────┬───────┘
                            │                        │
                            ▼                        │ continue/step
                   ┌──────────────┐                  ▼
                   │              │           ┌──────────────┐
                   │  IMMUTABLE   │           │              │
                   │  (Coredump)  │           │   RUNNING    │
                   │              │           │              │
                   └──────────────┘           └──────┬───────┘
                                                     │
                                              breakpoint/exception
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │              │
                                              │   STOPPED    │
                                              │              │
                                              └──────┬───────┘
                                                     │
                                              continue/step
                                                     │
                                                     ▼
                                              ┌──────────────┐
                                              │              │
                                              │   RUNNING    │
                                              │              │
                                              └──────────────┘
```

### 7.2 状态机实现

```python
from enum import Enum, auto


class DebugModality(Enum):
    """调试模态枚举"""
    INITIALIZING = auto()  # 初始化中
    MUTABLE = auto()       # Live 模式 - 可读写可执行
    IMMUTABLE = auto()     # Coredump 模式 - 只读
    RUNNING = auto()       # 目标正在运行中 (Live 模式子状态)
    STOPPED = auto()       # 目标已停止 (Live 模式子状态)
    CRASHED = auto()       # cuda-gdb 自身崩溃


class ModalityGuard:
    """
    模态守卫 - 根据当前调试模态拦截非法操作。

    核心职责：
    1. 启动时自动探测调试模态 (Live vs Coredump)
    2. 在 RPC 入口处拦截模态不兼容的操作
    3. 跟踪 Live 模式下的运行/停止子状态
    4. 在状态转换时强制附加焦点坐标快照
    """

    def __init__(self):
        self.current_mode = DebugModality.INITIALIZING
        self.last_focus_snapshot = None

    def detect_modality(self):
        """
        自动探测当前 cuda-gdb 的调试模态。
        通过检查 target 类型判断是 Live 进程还是 Coredump 文件。
        """
        try:
            target_info = gdb.execute("info target", to_string=True)

            if "cudacore" in target_info.lower() or ".cudacore" in target_info:
                self.current_mode = DebugModality.IMMUTABLE
                return {
                    "mode": "IMMUTABLE",
                    "description": "Post-mortem coredump analysis mode. "
                                   "Execution control is disabled. "
                                   "All data is read-only.",
                    "capabilities": {
                        "read_variables": True,
                        "read_registers": True,
                        "read_memory": True,
                        "set_focus": True,
                        "execution_control": False,
                        "modify_memory": False,
                        "set_breakpoints": False
                    }
                }
            else:
                self.current_mode = DebugModality.STOPPED
                return {
                    "mode": "MUTABLE",
                    "description": "Live interactive debugging mode. "
                                   "Full execution control available.",
                    "capabilities": {
                        "read_variables": True,
                        "read_registers": True,
                        "read_memory": True,
                        "set_focus": True,
                        "execution_control": True,
                        "modify_memory": True,
                        "set_breakpoints": True
                    }
                }
        except gdb.error as err:
            self.current_mode = DebugModality.INITIALIZING
            return {
                "mode": "UNKNOWN",
                "error": str(err)
            }

    def check_permission(self, method_name):
        """
        检查当前模态下是否允许执行指定方法。

        返回:
            None - 允许执行
            dict - 拒绝信息 (JSON-RPC Error 格式)
        """
        # 定义各方法的模态要求
        EXECUTION_METHODS = {
            "cuda_execution_control",
            "cuda_set_breakpoint",
            "cuda_remove_breakpoint",
            "cuda_modify_variable",
            "cuda_modify_register",
        }

        READ_ONLY_METHODS = {
            "cuda_set_focus",
            "cuda_evaluate_var",
            "cuda_dump_warp_registers",
            "cuda_analyze_exception",
            "cuda_read_shared_memory",
            "cuda_list_kernels",
            "cuda_backtrace",
            "cuda_disassemble",
        }

        # Coredump 模式下拦截执行类方法
        if self.current_mode == DebugModality.IMMUTABLE:
            if method_name in EXECUTION_METHODS:
                return {
                    "code": -32003,
                    "message": f"Method '{method_name}' is forbidden in "
                               f"Coredump (IMMUTABLE) mode",
                    "data": {
                        "current_mode": "IMMUTABLE",
                        "hint": "This is a post-mortem coredump analysis session. "
                                "Execution control and memory modification are "
                                "not available. Only read-only inspection tools "
                                "can be used.",
                        "available_methods": sorted(READ_ONLY_METHODS)
                    }
                }

        # RUNNING 状态下拦截需要停止状态的方法
        if self.current_mode == DebugModality.RUNNING:
            if method_name in READ_ONLY_METHODS:
                return {
                    "code": -32004,
                    "message": f"Method '{method_name}' requires the target "
                               f"to be stopped",
                    "data": {
                        "current_mode": "RUNNING",
                        "hint": "The target is currently running. Send "
                                "cuda_execution_control with action='interrupt' "
                                "to pause execution first."
                    }
                }

        return None  # 允许执行

    def on_target_stopped(self):
        """目标停止时的状态转换"""
        if self.current_mode == DebugModality.RUNNING:
            self.current_mode = DebugModality.STOPPED
        # 更新焦点快照
        self.last_focus_snapshot = self._capture_focus_snapshot()

    def on_target_running(self):
        """目标开始运行时的状态转换"""
        if self.current_mode in (DebugModality.STOPPED, DebugModality.MUTABLE):
            self.current_mode = DebugModality.RUNNING

    def _capture_focus_snapshot(self):
        """捕获当前焦点坐标快照"""
        try:
            output = gdb.execute(
                "cuda kernel block thread", to_string=True
            )
            return output.strip()
        except gdb.error:
            return None
```

### 7.3 Coredump 模式下的内存裁剪感知

```python
class CoredumpMemoryGuard:
    """
    Coredump 模式下的内存裁剪感知层。

    当 Coredump 文件在生成时通过 CUDA_COREDUMP_GENERATION_FLAGS 裁剪了
    特定内存段（如 skip_global_memory、skip_shared_memory 等），
    读取这些被裁剪的内存区域会返回"地址越界"错误。

    本模块负责区分"Coredump 裁剪导致的不可读"与"程序逻辑错误导致的野指针"，
    避免 Agent 产生误判。
    """

    # Coredump 内存读取失败的特征模式
    COREDUMP_MEMORY_PATTERNS = [
        "Failed to read global memory at address",
        "Failed to read shared memory at address",
        "Failed to read local memory at address",
        "Cannot access memory at address",
    ]

    @staticmethod
    def wrap_memory_read(read_func, *args, **kwargs):
        """
        包装内存读取函数，在 Coredump 模式下添加裁剪感知。

        如果读取失败且当前处于 Coredump 模式，
        在错误响应中添加元数据标签提示 Agent 这可能是裁剪导致的。
        """
        result = read_func(*args, **kwargs)

        if (result.get("status") == "error"
                and ModalityGuard.current_mode == DebugModality.IMMUTABLE):
            error_msg = result.get("message", "")
            for pattern in CoredumpMemoryGuard.COREDUMP_MEMORY_PATTERNS:
                if pattern in error_msg:
                    result["coredump_context"] = {
                        "likely_cause": "coredump_memory_truncation",
                        "explanation": "This memory region was likely excluded "
                                       "when the coredump was generated. The "
                                       "CUDA_COREDUMP_GENERATION_FLAGS environment "
                                       "variable controls which memory segments "
                                       "are included in the dump file.",
                        "not_a_bug": "This read failure does NOT indicate a "
                                     "runtime pointer error in the original "
                                     "program. It is an artifact of the coredump "
                                     "generation configuration.",
                        "suggestion": "Focus on register values and the call "
                                      "stack for debugging. If global memory "
                                      "contents are needed, regenerate the "
                                      "coredump without skip_global_memory flag."
                    }
                    break

        return result
```

---

## 8. 异常处理与容错机制

### 8.1 异常分类与处理策略

系统中的异常分为四个层级，每个层级有不同的处理策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    异常分层处理模型                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 1: GDB Python API 异常 (gdb.error)                  │
│  ├── 变量不存在 / 符号未找到                                  │
│  ├── 内存地址越界                                            │
│  ├── 变量被优化 (optimized out)                              │
│  ├── 焦点切换失败 (无效坐标)                                  │
│  └── 处理策略: 捕获 → 分类 → 结构化 JSON Error → 返回 Agent  │
│                                                             │
│  Level 2: CUDA 硬件异常 (CUDA_EXCEPTION)                    │
│  ├── Warp Illegal Address (EXCEPTION_14)                    │
│  ├── Warp Stack Overflow (EXCEPTION_35)                     │
│  ├── Warp Shared Memory Issue (EXCEPTION_32)                │
│  └── 处理策略: 事件回调 → 异常分析 → Notification → Agent    │
│                                                             │
│  Level 3: RPC 通信异常                                      │
│  ├── JSON 解析失败                                           │
│  ├── 未知方法名                                              │
│  ├── 参数校验失败                                            │
│  ├── 请求超时                                                │
│  └── 处理策略: JSON-RPC 标准错误码 → 返回 Agent              │
│                                                             │
│  Level 4: 系统级致命异常                                     │
│  ├── cuda-gdb 进程崩溃 (SIGSEGV/SIGABRT)                    │
│  ├── Python 解释器异常                                       │
│  ├── IPC 通道断开                                            │
│  └── 处理策略: Transport Proxy 捕获 → 崩溃报告 → Agent       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 JSON-RPC 错误码规范

| 错误码     | 含义               | 使用场景                    |
| ---------- | ------------------ | --------------------------- |
| `-32700` | Parse Error        | JSON 格式错误               |
| `-32600` | Invalid Request    | 缺少必要字段                |
| `-32601` | Method Not Found   | 未知的工具方法名            |
| `-32602` | Invalid Params     | 参数类型或值不合法          |
| `-32603` | Internal Error     | RPC 引擎内部异常            |
| `-32000` | GDB Error          | `gdb.error` 异常（通用）  |
| `-32001` | Timeout            | 命令执行超时                |
| `-32002` | Process Crashed    | cuda-gdb 进程崩溃           |
| `-32003` | Modality Forbidden | 当前模态不允许此操作        |
| `-32004` | Target Running     | 目标正在运行，需先暂停      |
| `-32005` | Optimized Out      | 变量被编译器优化            |
| `-32006` | No Active Kernel   | 无活跃的 GPU 内核           |
| `-32007` | Memory Truncated   | Coredump 内存裁剪导致不可读 |

### 8.3 `<optimized out>` 的专项处理

```python
class OptimizedOutHandler:
    """
    处理 CUDA 变量被编译器优化掉的情况。

    当 CUDA 程序未使用 -g -G 编译时，变量可能在其生命周期内
    被优化出寄存器，导致 gdb.parse_and_eval 返回 <optimized out>。

    本模块确保 Agent 收到的是明确的"变量已优化"提示，
    而非模糊的"读取失败"错误。
    """

    @staticmethod
    def check_and_wrap(gdb_value, expression):
        """
        检查 gdb.Value 是否为 optimized out，
        如果是则返回带有详细提示的结构化响应。
        """
        if gdb_value.is_optimized_out:
            return {
                "status": "optimized_out",
                "expression": expression,
                "value": None,
                "type": str(gdb_value.type),
                "meta": {
                    "optimized_out": True,
                    "explanation": f"The variable '{expression}' has been "
                                   f"optimized out by the CUDA compiler. "
                                   f"Its value is not available at this point "
                                   f"in the execution.",
                    "remediation": [
                        "Recompile the CUDA source with '-g -G' flags to "
                        "disable optimizations and preserve all debug info",
                        "The '-G' flag specifically disables GPU code "
                        "optimizations including dead code elimination "
                        "and register spilling",
                        "Try reading the variable at a different point in "
                        "the kernel execution where it may still be live"
                    ]
                }
            }
        return None  # 变量正常，不需要特殊处理
```

### 8.4 Warp 隐式推进的语义警告

```python
class WarpAdvanceWarning:
    """
    Warp 隐式推进警告生成器。

    当 Agent 在某个线程焦点下执行 step/next 后，
    同 Warp 内的其他 31 个线程也会被隐式推进。
    本模块在执行控制返回中附加此警告，
    防止 Agent 在切换到同 Warp 线程后产生状态认知偏差。
    """

    @staticmethod
    def generate_warning(action, current_focus):
        """生成 Warp 隐式推进的语义警告"""
        if action in ("step", "next", "stepi"):
            warp_id = current_focus.get("hardware_mapping", {}).get("warp")
            lane_id = current_focus.get("hardware_mapping", {}).get("lane")

            if warp_id is not None:
                return {
                    "warp_advance_warning": {
                        "message": f"Executing '{action}' advanced the entire "
                                   f"Warp {warp_id} (all 32 lanes) synchronously, "
                                   f"not just Lane {lane_id}.",
                        "implication": "If you switch focus to another thread "
                                       "within the same Warp, its registers and "
                                       "local variables will reflect the advanced "
                                       "state, not the previous state.",
                        "affected_lanes": list(range(32)),
                        "current_lane": lane_id
                    }
                }
        return {}
```

---

## 9. 部署方案与配置规范

### 9.1 环境要求

| 组件         | 最低版本        | 推荐版本                | 说明                                     |
| ------------ | --------------- | ----------------------- | ---------------------------------------- |
| CUDA Toolkit | 11.4+           | 12.x+                   | cuda-gdb 需要 GDB 10.1+ 的 Python 3 支持 |
| cuda-gdb     | 随 Toolkit      | 最新                    | 底层 GDB 版本越新，Python API 越完善     |
| Python 3     | 3.8+            | 3.10+                   | cuda-gdb 通过 dlopen 加载系统 libpython3 |
| 操作系统     | Linux x86_64    | Ubuntu 22.04+ / RHEL 8+ | cuda-gdb 仅支持 Linux                    |
| GPU Driver   | 与 Toolkit 匹配 | 最新稳定版              | 驱动版本必须与 Toolkit 兼容              |

### 9.2 目录结构

```
cuda-gdb-agent/
├── transport_proxy.py          # 第二层：宿主代理传输层
├── agent_rpc_server.py         # 第三层：内嵌式 Python RPC 引擎 (注入 cuda-gdb)
├── config.yaml                 # 配置文件
├── mcp_manifest.json           # MCP Tool 注册清单
├── tests/
│   ├── test_serializer.py      # gdb.Value 序列化单元测试
│   ├── test_modality_guard.py  # 模态守卫单元测试
│   └── test_integration.py     # 端到端集成测试
└── docs/
    └── design.md               # 本设计文档
```

### 9.3 启动流程

```bash
# 方式 1: Live 调试 - 启动新进程
python transport_proxy.py \
  --mode live \
  --executable ./my_cuda_app \
  --args "arg1 arg2" \
  --cuda-gdb-path /usr/local/cuda/bin/cuda-gdb

# 方式 2: Live 调试 - Attach 到已有进程
python transport_proxy.py \
  --mode live \
  --pid 12345

# 方式 3: Coredump 分析
python transport_proxy.py \
  --mode coredump \
  --executable ./my_cuda_app \
  --corefile ./core.cuda.12345.cudacore

# 方式 4: 作为 MCP Server 通过 stdio 启动 (供 Cursor/Claude Desktop 使用)
# 在 MCP 配置文件中:
{
  "mcpServers": {
    "cuda-gdb": {
      "command": "python",
      "args": ["transport_proxy.py", "--mode", "live", "--executable", "./app"],
      "env": {
        "CUDA_GDB_PATH": "/usr/local/cuda/bin/cuda-gdb"
      }
    }
  }
}
```

### 9.4 配置文件规范

```yaml
# config.yaml

# Transport Layer 配置
transport:
  # IPC 通道类型: "uds" (Unix Domain Socket) 或 "pipe" (命名管道)
  ipc_type: "uds"
  # UDS 路径模板 ({pid} 会被替换为实际进程 ID)
  socket_path_template: "/tmp/cuda-gdb-agent-{pid}.sock"

# RPC Engine 配置
rpc:
  # 单个请求的内部超时 (秒)
  internal_timeout: 25
  # 最大响应体积 (字节)
  max_response_size: 1048576
  # gdb.Value 序列化的最大递归深度
  max_serialization_depth: 5
  # 数组序列化的最大元素数
  max_array_elements: 256

# 调试器配置
debugger:
  # cuda-gdb 可执行文件路径
  cuda_gdb_path: "cuda-gdb"
  # 传递给 cuda-gdb 的额外参数
  extra_args: ""
  # 是否启用 TUI 模式 (通常应禁用)
  enable_tui: false

# 超时与恢复配置
resilience:
  # Transport 层的外部超时 (秒)
  external_timeout: 30
  # SIGINT 后的宽限期 (秒)
  interrupt_grace_period: 5
  # cuda-gdb 崩溃后是否自动重启
  auto_restart_on_crash: false

# 日志配置
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  # 日志输出目标: "stderr", "file"
  output: "stderr"
  # 日志文件路径 (仅当 output 为 "file" 时)
  file_path: "/tmp/cuda-gdb-agent.log"
```

### 9.5 Coredump 生成配置指南

为了确保 Coredump 文件包含足够的调试信息，建议在目标程序运行前设置以下环境变量：

```bash
# 启用 CUDA Coredump 生成
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1

# 同时生成 CPU 和 GPU 的 coredump
export CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION=1

# 控制 Coredump 包含的内存段 (去掉 skip 前缀以包含对应内存)
# 默认会裁剪大量内存以控制文件体积
# 完整 dump (文件可能非常大):
export CUDA_COREDUMP_GENERATION_FLAGS=""

# 推荐配置 (保留寄存器和共享内存，跳过全局显存):
export CUDA_COREDUMP_GENERATION_FLAGS="skip_global_memory"

# 最小 dump (仅保留寄存器和栈):
export CUDA_COREDUMP_GENERATION_FLAGS="skip_global_memory,skip_shared_memory,skip_local_memory"

# 编译时务必加入调试标志
nvcc -g -G -O0 -o my_cuda_app my_cuda_app.cu
```

---

## 10. 演进路线与扩展规划

### 10.1 版本演进路线

```
Phase 1 (MVP - 4 周)
├── 核心 RPC 引擎骨架
├── 基础 Tool: set_focus, evaluate_var, dump_registers
├── Coredump 模式支持
├── 模态守卫 (IMMUTABLE/MUTABLE)
└── stdio Transport (MCP 兼容)

Phase 2 (增强 - 4 周)
├── Live 调试完整支持 (线程隔离 + 异步事件)
├── 异常分析器 (CUDA_EXCEPTION 映射)
├── 共享内存访问器 (@shared 修饰符)
├── 断点管理 (set/remove/list)
└── 反汇编工具 (SASS/PTX)

Phase 3 (生产化 - 4 周)
├── 崩溃恢复与自动重启
├── 多 GPU 设备支持
├── 性能剖析集成 (nsys/ncu 数据关联)
├── 会话持久化与恢复
└── 安全加固 (认证、权限控制)

Phase 4 (智能化 - 持续)
├── Agent 推理链模板 (预置调试策略)
├── 自动根因分析 (结合 LLM 推理)
├── 历史调试会话知识库
└── 多 Agent 协作调试
```

### 10.2 扩展工具端点规划

| 工具名称                    | 阶段    | 说明                                   |
| --------------------------- | ------- | -------------------------------------- |
| `cuda_set_breakpoint`     | Phase 2 | 在指定源码行或地址设置硬件断点         |
| `cuda_remove_breakpoint`  | Phase 2 | 移除断点                               |
| `cuda_list_breakpoints`   | Phase 2 | 列出所有活跃断点                       |
| `cuda_backtrace`          | Phase 2 | 获取当前线程的调用栈回溯               |
| `cuda_disassemble`        | Phase 2 | 反汇编指定地址范围的 SASS/PTX 代码     |
| `cuda_list_kernels`       | Phase 1 | 列出所有活跃的 CUDA 内核及其 Grid 配置 |
| `cuda_read_global_memory` | Phase 2 | 读取全局显存 (带 `@global` 修饰)     |
| `cuda_read_local_memory`  | Phase 2 | 读取线程本地内存 (带 `@local` 修饰)  |
| `cuda_device_info`        | Phase 1 | 获取 GPU 设备信息 (SM 数量、架构等)    |
| `cuda_modify_variable`    | Phase 3 | 修改变量值 (仅 Live 模式)              |
| `cuda_watchpoint`         | Phase 3 | 设置数据观察点                         |
| `cuda_profile_region`     | Phase 3 | 对指定代码区域进行性能采样             |

### 10.3 与现有生态的集成策略

```
┌─────────────────────────────────────────────────────────────┐
│                    集成生态全景                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MCP 生态:                                                  │
│  ├── Cursor IDE → 通过 MCP stdio 直接集成                    │
│  ├── Claude Desktop → MCP Server 配置                       │
│  └── 自定义 Agent → MCP Client SDK                          │
│                                                             │
│  LangChain/LangGraph 生态:                                  │
│  ├── 将每个 Tool 封装为 LangChain Tool                       │
│  ├── 构建 ReAct Agent 调试链                                 │
│  └── 集成 Memory 模块保持调试上下文                            │
│                                                             │
│  NVIDIA 工具链:                                              │
│  ├── Nsight Systems (nsys) → 性能数据关联                    │
│  ├── Nsight Compute (ncu) → 内核级性能指标                    │
│  └── CUDA Sanitizer → 内存错误检测结果关联                    │
│                                                             │
│  CI/CD 集成:                                                │
│  ├── GitHub Actions → 自动化 Coredump 分析                   │
│  ├── Jenkins → 崩溃报告自动生成                               │
│  └── 训练集群 → 批量 Coredump 分析流水线                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 附录 A：CUDA_EXCEPTION 完整映射表

| 异常码                | 名称                           | 严重级别 | 描述                                    | 常见原因                              |
| --------------------- | ------------------------------ | -------- | --------------------------------------- | ------------------------------------- |
| `CUDA_EXCEPTION_1`  | Lane Illegal Address           | Critical | 单个 Lane 访问非法内存地址              | 数组越界、空指针解引用                |
| `CUDA_EXCEPTION_2`  | Lane User Stack Overflow       | Critical | Lane 调用栈溢出                         | 深度递归、大局部数组                  |
| `CUDA_EXCEPTION_3`  | Device Hardware Stack Overflow | Critical | 硬件调用栈溢出                          | 极深函数调用链                        |
| `CUDA_EXCEPTION_4`  | Warp Illegal Instruction       | Critical | 非法指令                                | 代码损坏、架构不匹配                  |
| `CUDA_EXCEPTION_5`  | Warp Out-of-Range Address      | Critical | 地址超出所有有效内存区域                | use-after-free、地址计算溢出          |
| `CUDA_EXCEPTION_6`  | Warp Misaligned Address        | Warning  | 内存访问未对齐                          | 类型转换对齐问题                      |
| `CUDA_EXCEPTION_7`  | Warp Invalid Address Space     | Critical | 无效地址空间访问                        | shared/global 指针混淆                |
| `CUDA_EXCEPTION_8`  | Warp Invalid PC                | Critical | 程序计数器跳转到非法地址                | 函数指针损坏、栈溢出覆盖返回地址      |
| `CUDA_EXCEPTION_14` | Warp Illegal Address           | Critical | Warp 内任意 Lane 访问非法地址（最常见） | 缓冲区溢出、cudaFree 后访问、竞争条件 |
| `CUDA_EXCEPTION_32` | Warp Shared Memory Issue       | Critical | 集群级共享内存不可纠正错误              | 硬件 ECC 错误、分布式共享内存违规     |
| `CUDA_EXCEPTION_35` | Warp User Stack Overflow       | Critical | Warp 用户栈溢出                         | 递归无深度限制、alloca 超限           |

---

## 附录 B：调试模态对比矩阵

| 对比维度                  | 实时调试模式 (Live)                       | 事后转储模式 (Coredump)               | RPC 策略                                   |
| ------------------------- | ----------------------------------------- | ------------------------------------- | ------------------------------------------ |
| **执行流控制**      | 完全支持 (continue, step, next, 硬件断点) | 绝对禁止                              | 模态守卫拦截，返回 `-32003` 错误         |
| **内存/寄存器修改** | 支持                                      | 完全只读                              | 修改请求返回带权限提示的 JSON Error        |
| **内存空间完备性**  | 完整可用                                  | 残缺不连续 (受 GENERATION_FLAGS 裁剪) | 读取失败时附加 `coredump_context` 元数据 |
| **异常时序精度**    | 受抢占和缓存影响，可能有时序扭曲          | 指令级精确 (严格停留在 errorpc)       | Coredump 优先用于精确错误定位              |
| **API 阻塞性**      | 极强 (内核运行期间阻塞 GIL)               | 无阻塞 (静态二进制解析)               | Live 模式强制异步线程队列架构              |
| **断点支持**        | 支持硬件断点和条件断点                    | 不支持                                | 模态守卫拦截                               |
| **焦点切换**        | 支持，但 Warp 隐式推进                    | 支持，状态完全确定                    | 附加 Warp 推进警告                         |
| **性能影响**        | 目标程序减速数个数量级                    | 无 (离线分析)                         | Live 模式需考虑 TDR 超时风险               |

---

## 附录 C：关键 GDB Python API 速查

| API                                  | 用途          | CUDA 扩展行为                                                                                        |
| ------------------------------------ | ------------- | ---------------------------------------------------------------------------------------------------- |
| `gdb.parse_and_eval(expr)`         | 表达式求值    | 支持 `@shared`/`@global`/`@local` 修饰符、`$R0`-`$R255` 寄存器、`threadIdx.x` 等内置变量 |
| `gdb.execute(cmd, to_string=True)` | 执行 GDB 命令 | 支持 `cuda thread/block/kernel` 焦点切换命令                                                       |
| `gdb.Value`                        | 内存值对象    | `.is_optimized_out` 属性标识变量是否被优化                                                         |
| `gdb.Breakpoint(spec)`             | 设置断点      | 支持在 `__global__` 函数中设置断点                                                                 |
| `gdb.Thread`                       | 后台线程      | 自动调用 `gdb.block_signals()` 屏蔽信号                                                            |
| `gdb.post_event(callable)`         | 主线程调度    | 将任务安全地调度到 GDB 主事件循环                                                                    |
| `gdb.events.stop.connect(cb)`      | 停止事件回调  | 捕获断点命中和 CUDA 硬件异常                                                                         |
| `gdb.events.exited.connect(cb)`    | 退出事件回调  | 捕获程序正常/异常退出                                                                                |
| `gdb.selected_frame()`             | 获取当前栈帧  | 焦点切换后返回 GPU 线程的栈帧                                                                        |
| `gdb.Type`                         | 类型系统      | 支持 CUDA 特有类型                                                                                   |
