# CUDA-GDB-CLI RPC 服务 — Agent Tool Schema 与 JSON-RPC 接口详细规范

> **版本**: v1.0
> **日期**: 2026-03-23
> **协议**: JSON-RPC 2.0 (MCP 兼容)

---

## 目录

- [1. 协议基础](#1-协议基础)
- [2. 核心工具接口](#2-核心工具接口)
  - [2.1 cuda_set_focus](#21-cuda_set_focus)
  - [2.2 cuda_evaluate_var](#22-cuda_evaluate_var)
  - [2.3 cuda_dump_warp_registers](#23-cuda_dump_warp_registers)
  - [2.4 cuda_analyze_exception](#24-cuda_analyze_exception)
  - [2.5 cuda_execution_control](#25-cuda_execution_control)
  - [2.6 cuda_read_shared_memory](#26-cuda_read_shared_memory)
  - [2.7 cuda_list_kernels](#27-cuda_list_kernels)
  - [2.8 cuda_backtrace](#28-cuda_backtrace)
  - [2.9 cuda_disassemble](#29-cuda_disassemble)
  - [2.10 cuda_set_breakpoint](#210-cuda_set_breakpoint)
  - [2.11 cuda_remove_breakpoint](#211-cuda_remove_breakpoint)
  - [2.12 cuda_device_info](#212-cuda_device_info)
- [3. 异步通知事件](#3-异步通知事件)
- [4. 错误码规范](#4-错误码规范)
- [5. 通用响应元数据](#5-通用响应元数据)

---

## 1. 协议基础

### 1.1 请求格式

所有请求遵循 JSON-RPC 2.0 规范：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "cuda_set_focus",
  "params": {
    "block": [0, 0, 0],
    "thread": [1, 0, 0]
  }
}
```

### 1.2 成功响应格式

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "status": "ok",
    "data": { ... }
  }
}
```

### 1.3 错误响应格式

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32000,
    "message": "Human-readable error description",
    "data": {
      "source": "gdb_internal | rpc_engine | modality_guard",
      "hint": "Actionable suggestion for the Agent",
      "details": { ... }
    }
  }
}
```

### 1.4 异步通知格式（无 id 字段）

```json
{
  "jsonrpc": "2.0",
  "method": "cuda_stop_event",
  "params": { ... }
}
```

---

## 2. 核心工具接口

### 2.1 cuda_set_focus

**功能**：切换 GPU 线程焦点到指定的软件坐标。后续所有变量求值、内存读取均在该线程的栈帧上下文中执行。

**MCP Tool 注册**：

```json
{
  "name": "cuda_set_focus",
  "description": "切换 GPU 线程焦点到指定的 Block/Thread 软件坐标。执行后所有后续的变量求值和内存读取都将在该线程的栈帧上下文中进行。返回值包含该线程映射到的硬件坐标（Device/SM/Warp/Lane）。注意：同一 Warp 内的 32 个线程共享执行状态，对任一线程执行 step/next 会推进整个 Warp。",
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
      },
      "kernel": {
        "type": "integer",
        "description": "可选。Kernel 编号，用于多 Kernel 并发场景。省略时使用当前活跃 Kernel。"
      }
    },
    "required": ["block", "thread"]
  }
}
```

**请求示例**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "cuda_set_focus",
  "params": {
    "block": [2, 0, 0],
    "thread": [15, 0, 0]
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "status": "ok",
    "software_coords": {
      "block": [2, 0, 0],
      "thread": [15, 0, 0],
      "kernel": null
    },
    "hardware_mapping": {
      "device": 0,
      "sm": 7,
      "warp": 3,
      "lane": 15
    },
    "verification": {
      "verified": true,
      "actual_thread": [15, 0, 0],
      "actual_block": [2, 0, 0]
    }
  }
}
```

**错误响应 — 坐标越界**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32000,
    "message": "Thread (256,0,0) is not within the valid range for block (2,0,0)",
    "data": {
      "source": "gdb_internal",
      "error_type": "invalid_coordinates",
      "hint": "The specified thread coordinates exceed the block dimensions. Use cuda_list_kernels to check the current grid/block configuration."
    }
  }
}
```

**错误响应 — 无活跃 Kernel**：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32006,
    "message": "No CUDA kernel is currently active on the GPU",
    "data": {
      "source": "gdb_internal",
      "error_type": "no_active_kernel",
      "hint": "The program may be executing host (CPU) code. Set a breakpoint inside a __global__ function and continue execution first."
    }
  }
}
```

---

### 2.2 cuda_evaluate_var

**功能**：在当前焦点线程的栈帧上下文中求值表达式。支持 C/C++ 变量、CUDA 内置变量（threadIdx 等）、地址空间修饰符和数组切片。

**MCP Tool 注册**：

```json
{
  "name": "cuda_evaluate_var",
  "description": "在当前 GPU 线程焦点的栈帧上下文中求值一个表达式。支持 C/C++ 变量、CUDA 内置变量（如 threadIdx.x、blockDim.y）、指针解引用、数组切片。可通过 cast_space 参数指定地址空间修饰符（shared/global/local）以正确访问 GPU 特有内存区域。返回结构化的值、类型和元数据信息。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "要求值的 C/C++ 表达式。示例：'threadIdx.x'、'my_array[10]'、'*ptr'、'my_struct.field'"
      },
      "cast_space": {
        "type": "string",
        "enum": ["auto", "global", "shared", "local"],
        "description": "地址空间修饰符。'auto' 使用 GDB 默认推断；'shared' 强制注入 @shared 修饰符用于共享内存访问；'global' 用于全局显存；'local' 用于线程本地内存。默认 'auto'。"
      },
      "array_length": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256,
        "description": "如果表达式是数组或指针，指定要读取的连续元素数量。使用 GDB 的 @N 语法。"
      }
    },
    "required": ["expression"]
  }
}
```

**请求示例 — 读取 CUDA 内置变量**：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "cuda_evaluate_var",
  "params": {
    "expression": "threadIdx.x"
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "status": "ok",
    "data": {
      "value": 15,
      "hex": "0xf",
      "type": "unsigned int"
    }
  }
}
```

**请求示例 — 读取共享内存数组**：

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "cuda_evaluate_var",
  "params": {
    "expression": "s_data",
    "cast_space": "shared",
    "array_length": 4
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "status": "ok",
    "memory_space": "shared",
    "data": {
      "value": [
        {"value": 1.5, "type": "float"},
        {"value": 2.3, "type": "float"},
        {"value": 0.0, "type": "float"},
        {"value": -1.2, "type": "float"}
      ],
      "type": "float [4]",
      "meta": {
        "total_length": 4,
        "displayed_length": 4,
        "truncated": false
      }
    }
  }
}
```

**错误响应 — 变量被优化**：

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "error": {
    "code": -32005,
    "message": "Variable 'loop_counter' has been optimized out",
    "data": {
      "source": "gdb_internal",
      "error_type": "optimized_out",
      "expression": "loop_counter",
      "type": "int",
      "hint": "The variable was optimized out by the CUDA compiler. Recompile with '-g -G' flags to disable optimizations and preserve all debug info. The '-G' flag specifically disables GPU code optimizations.",
      "remediation": [
        "Recompile with: nvcc -g -G -O0 source.cu",
        "Try reading the variable at a different execution point where it may still be live in a register"
      ]
    }
  }
}
```

**错误响应 — Coredump 内存裁剪**：

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "error": {
    "code": -32007,
    "message": "Failed to read global memory at address 0x7f1234560000",
    "data": {
      "source": "gdb_internal",
      "error_type": "memory_read_failed",
      "coredump_context": {
        "likely_cause": "coredump_memory_truncation",
        "explanation": "This memory region was likely excluded when the coredump was generated via CUDA_COREDUMP_GENERATION_FLAGS.",
        "not_a_bug": "This read failure does NOT indicate a runtime pointer error in the original program.",
        "suggestion": "Focus on register values and the call stack. If global memory contents are needed, regenerate the coredump without skip_global_memory flag."
      }
    }
  }
}
```

---

### 2.3 cuda_dump_warp_registers

**功能**：安全地转储当前聚焦 Warp 的所有已分配硬件寄存器。自动检测寄存器分配上限，避免访问越界。

**MCP Tool 注册**：

```json
{
  "name": "cuda_dump_warp_registers",
  "description": "转储当前聚焦 Warp 的所有已分配 CUDA 硬件寄存器。包括通用寄存器（R0-R255 中已分配的子集）、谓词寄存器（P0-P6）和条件码寄存器（CC）。自动检测当前 Kernel 的实际寄存器分配数量，避免访问未分配寄存器导致的越界错误。无需参数，隐式作用于当前焦点所在的 Warp。",
  "inputSchema": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

**请求示例**：

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "cuda_dump_warp_registers",
  "params": {}
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {
    "status": "ok",
    "warp_info": {
      "device": 0,
      "sm": 7,
      "warp": 3
    },
    "general_registers": {
      "R0": "0x00000042",
      "R1": "0x00000000",
      "R2": "0x7f123456",
      "R3": "0x00000010",
      "R4": "0xdeadbeef",
      "R5": "0x00000001"
    },
    "predicate_registers": {
      "P0": "0x1",
      "P1": "0x0",
      "P2": "0x1",
      "P3": "0x0",
      "P4": "0x0",
      "P5": "0x0",
      "P6": "0x0"
    },
    "special_registers": {
      "CC": "0x0"
    },
    "register_count": 6,
    "max_possible": 255
  }
}
```

---

### 2.4 cuda_analyze_exception

**功能**：分析当前 CUDA 硬件异常的完整上下文。将底层 `CUDA_EXCEPTION` 枚举码翻译为结构化的语义描述，提取触发异常的精确 SASS 指令，并附加关键寄存器快照用于地址反推。

**MCP Tool 注册**：

```json
{
  "name": "cuda_analyze_exception",
  "description": "分析当前 CUDA 硬件异常的完整上下文。当 GPU 内核崩溃时，异常在 Warp 级别被硬件陷入并抛出。本工具将底层的 CUDA_EXCEPTION 枚举码翻译为结构化的语义描述，提取触发异常的精确 SASS 汇编指令（通过 errorpc 反汇编），并附加关键寄存器快照用于内存地址计算反推。无需参数，自动分析当前停止状态。",
  "inputSchema": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

**请求示例**：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "cuda_analyze_exception",
  "params": {}
}
```

**成功响应 — 检测到异常**：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "status": "exception_detected",
    "exception_code": "CUDA_EXCEPTION_14",
    "exception_name": "Warp Illegal Address",
    "severity": "critical",
    "description": "Any lane within the warp accessed an illegal memory address. This is the most common CUDA memory error.",
    "common_causes": [
      "Global memory buffer overflow",
      "Accessing device memory after cudaFree()",
      "Race condition corrupting pointer values",
      "Incorrect grid/block dimension causing out-of-bounds thread indices"
    ],
    "errorpc": "0x555555557a80",
    "pc": "0x555555557a84",
    "faulting_instruction": "0x555555557a80: *>  ST.E.U8 [R2], R0",
    "focus_at_exception": "kernel 0, block (2,0,0), thread (15,0,0)",
    "key_registers_snapshot": {
      "R0": "0x000000ff",
      "R1": "0x00000000",
      "R2": "0x7f00deadbeef",
      "R3": "0x00000010",
      "R4": "0x00001000",
      "R5": "0x0000000f"
    }
  }
}
```

**成功响应 — 无异常**：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "status": "no_exception",
    "message": "No CUDA exception detected in current context. $errorpc is not available."
  }
}
```

---

### 2.5 cuda_execution_control

**功能**：控制目标程序的执行流。支持 step（单步进入）、next（单步跳过）、continue（继续运行）、interrupt（中断运行中的程序）。在 Coredump 模式下自动拦截并返回模态错误。

**MCP Tool 注册**：

```json
{
  "name": "cuda_execution_control",
  "description": "控制目标程序的执行流。仅在 Live 调试模式下可用，Coredump 模式下会被自动拦截。注意：step/next 会推进当前焦点线程所在的整个 Warp（32 个线程）同步执行。continue 是异步操作，程序停止时会通过 cuda_stop_event 通知推送。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["step", "next", "stepi", "continue", "finish", "interrupt"],
        "description": "'step' 单步进入（进入函数调用）；'next' 单步跳过（不进入函数）；'stepi' 单条指令步进（SASS 级别）；'continue' 继续运行直到下一个断点或异常；'finish' 运行到当前函数返回；'interrupt' 中断正在运行的程序。"
      }
    },
    "required": ["action"]
  }
}
```

**请求示例 — step**：

```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "method": "cuda_execution_control",
  "params": {
    "action": "step"
  }
}
```

**成功响应 — step/next（同步完成）**：

```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "result": {
    "status": "stopped",
    "action": "step",
    "current_focus": {
      "kernel": "matmul_kernel",
      "block": [2, 0, 0],
      "thread": [15, 0, 0],
      "hardware": {"device": 0, "sm": 7, "warp": 3, "lane": 15}
    },
    "source_location": {
      "file": "matmul.cu",
      "line": 43
    },
    "pc": "0x555555557a84",
    "warp_advance_warning": {
      "message": "Executing 'step' advanced the entire Warp 3 (all 32 lanes) synchronously, not just Lane 15.",
      "implication": "If you switch focus to another thread within the same Warp, its registers and local variables will reflect the advanced state, not the previous state.",
      "affected_lanes": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
      "current_lane": 15
    }
  }
}
```

**成功响应 — continue（异步启动）**：

```json
{
  "jsonrpc": "2.0",
  "id": 9,
  "result": {
    "status": "running",
    "action": "continue",
    "blocked": true,
    "message": "Target is now running. You will receive a cuda_stop_event notification when the program hits a breakpoint, exception, or exits."
  }
}
```

**错误响应 — Coredump 模态拦截**：

```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "error": {
    "code": -32003,
    "message": "Method 'cuda_execution_control' is forbidden in Coredump (IMMUTABLE) mode",
    "data": {
      "source": "modality_guard",
      "current_mode": "IMMUTABLE",
      "hint": "This is a post-mortem coredump analysis session. Execution control and memory modification are not available. Only read-only inspection tools can be used.",
      "available_methods": [
        "cuda_analyze_exception",
        "cuda_backtrace",
        "cuda_device_info",
        "cuda_disassemble",
        "cuda_dump_warp_registers",
        "cuda_evaluate_var",
        "cuda_list_kernels",
        "cuda_read_shared_memory",
        "cuda_set_focus"
      ]
    }
  }
}
```

---

### 2.6 cuda_read_shared_memory

**功能**：通过物理地址安全地读取共享内存。自动注入 `@shared` 地址空间修饰符，处理 IPC 内存访问拒绝异常。

**MCP Tool 注册**：

```json
{
  "name": "cuda_read_shared_memory",
  "description": "通过物理地址或变量名安全地读取 GPU 共享内存（Shared Memory）。共享内存位于 SM 内部片上 SRAM，地址空间独立于全局显存。本工具自动注入 @shared 地址空间修饰符以确保正确寻址，并处理 CUDA IPC 跨进程内存访问拒绝异常。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "address": {
        "type": "string",
        "description": "共享内存的起始地址（十六进制字符串，如 '0x20'）或变量名（如 's_data'）"
      },
      "data_type": {
        "type": "string",
        "description": "C 数据类型。如 'int'、'float'、'double'、'int4'、'float4'。当 address 为变量名时可省略。",
        "default": "int"
      },
      "count": {
        "type": "integer",
        "minimum": 1,
        "maximum": 256,
        "description": "要读取的连续元素数量。默认 1。",
        "default": 1
      }
    },
    "required": ["address"]
  }
}
```

**请求示例**：

```json
{
  "jsonrpc": "2.0",
  "id": 11,
  "method": "cuda_read_shared_memory",
  "params": {
    "address": "0x20",
    "data_type": "float",
    "count": 8
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 11,
  "result": {
    "status": "ok",
    "memory_space": "shared",
    "address": "0x20",
    "data_type": "float",
    "count": 8,
    "data": {
      "value": [1.0, 2.5, 3.14, 0.0, -1.0, 42.0, 0.001, 100.0],
      "type": "float [8]"
    }
  }
}
```

**错误响应 — IPC 内存访问拒绝**：

```json
{
  "jsonrpc": "2.0",
  "id": 11,
  "error": {
    "code": -32000,
    "message": "Cannot access memory imported via CUDA IPC",
    "data": {
      "source": "gdb_internal",
      "error_type": "ipc_access_denied",
      "hint": "This shared memory allocation was imported via CUDA IPC (inter-process communication). cuda-gdb explicitly prohibits accessing IPC-imported memory for security reasons."
    }
  }
}
```

---

### 2.7 cuda_list_kernels

**功能**：列出当前所有活跃的 CUDA 内核及其 Grid/Block 配置。

**MCP Tool 注册**：

```json
{
  "name": "cuda_list_kernels",
  "description": "列出当前所有活跃的 CUDA 内核（Kernel）及其 Grid/Block 维度配置。用于在切换焦点前确认有效的坐标范围，避免坐标越界错误。",
  "inputSchema": {
    "type": "object",
    "properties": {},
    "required": []
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "result": {
    "status": "ok",
    "kernels": [
      {
        "kernel_id": 0,
        "function_name": "matmul_kernel",
        "grid_dim": [128, 128, 1],
        "block_dim": [32, 32, 1],
        "shared_memory_bytes": 4096,
        "device": 0,
        "state": "stopped"
      }
    ],
    "total_active_kernels": 1
  }
}
```

---

### 2.8 cuda_backtrace

**功能**：获取当前焦点线程的调用栈回溯。

**MCP Tool 注册**：

```json
{
  "name": "cuda_backtrace",
  "description": "获取当前焦点 GPU 线程的调用栈回溯（backtrace）。显示从当前执行点到内核入口的完整函数调用链，包括每一帧的函数名、源文件位置和参数信息。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "max_frames": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "最大回溯帧数。默认 20。",
        "default": 20
      }
    },
    "required": []
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 13,
  "result": {
    "status": "ok",
    "frames": [
      {
        "level": 0,
        "function": "device_helper",
        "file": "matmul.cu",
        "line": 15,
        "address": "0x555555557a80"
      },
      {
        "level": 1,
        "function": "matmul_kernel",
        "file": "matmul.cu",
        "line": 42,
        "address": "0x555555557b20"
      }
    ],
    "total_frames": 2
  }
}
```

---

### 2.9 cuda_disassemble

**功能**：反汇编指定地址范围的 SASS/PTX 代码。用于精确定位异常触发的汇编指令。

**MCP Tool 注册**：

```json
{
  "name": "cuda_disassemble",
  "description": "反汇编指定地址范围或当前 PC 附近的 SASS（GPU 原生汇编）代码。用于精确定位异常触发的汇编指令。触发异常的指令会以 '*>' 前缀标记（errorpc）。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "address": {
        "type": "string",
        "description": "起始地址（十六进制）。省略时使用当前 $pc。"
      },
      "instruction_count": {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
        "description": "要反汇编的指令数量。默认 10。",
        "default": 10
      },
      "format": {
        "type": "string",
        "enum": ["sass", "ptx"],
        "description": "反汇编格式。'sass' 为 GPU 原生汇编；'ptx' 为中间表示。默认 'sass'。",
        "default": "sass"
      }
    },
    "required": []
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 14,
  "result": {
    "status": "ok",
    "format": "sass",
    "instructions": [
      {"address": "0x555555557a70", "instruction": "MOV R2, R5", "is_current_pc": false, "is_errorpc": false},
      {"address": "0x555555557a78", "instruction": "IMAD.WIDE R2, R3, R4, R2", "is_current_pc": false, "is_errorpc": false},
      {"address": "0x555555557a80", "instruction": "ST.E.U8 [R2], R0", "is_current_pc": false, "is_errorpc": true},
      {"address": "0x555555557a84", "instruction": "EXIT", "is_current_pc": true, "is_errorpc": false}
    ]
  }
}
```

---

### 2.10 cuda_set_breakpoint

**功能**：在指定位置设置断点。仅 Live 模式可用。

**MCP Tool 注册**：

```json
{
  "name": "cuda_set_breakpoint",
  "description": "在指定的源码位置或函数名设置断点。仅在 Live 调试模式下可用。支持在 __global__ 和 __device__ 函数中设置断点。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "断点位置。格式：'filename:line'（如 'matmul.cu:42'）或函数名（如 'matmul_kernel'）"
      },
      "condition": {
        "type": "string",
        "description": "可选的条件表达式。仅当条件为真时触发断点。如 'threadIdx.x == 0 && blockIdx.x == 5'"
      }
    },
    "required": ["location"]
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 15,
  "result": {
    "status": "ok",
    "breakpoint_id": 3,
    "location": "matmul.cu:42",
    "condition": "threadIdx.x == 0 && blockIdx.x == 5",
    "enabled": true
  }
}
```

---

### 2.11 cuda_remove_breakpoint

**功能**：移除指定编号的断点。

**MCP Tool 注册**：

```json
{
  "name": "cuda_remove_breakpoint",
  "description": "移除指定编号的断点。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "breakpoint_id": {
        "type": "integer",
        "description": "要移除的断点编号（由 cuda_set_breakpoint 返回）"
      }
    },
    "required": ["breakpoint_id"]
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 16,
  "result": {
    "status": "ok",
    "breakpoint_id": 3,
    "message": "Breakpoint 3 removed"
  }
}
```

---

### 2.12 cuda_device_info

**功能**：获取 GPU 设备的硬件信息。

**MCP Tool 注册**：

```json
{
  "name": "cuda_device_info",
  "description": "获取当前 GPU 设备的硬件信息，包括设备名称、SM 数量、计算能力、显存大小等。用于 Agent 理解硬件约束。",
  "inputSchema": {
    "type": "object",
    "properties": {
      "device_id": {
        "type": "integer",
        "description": "GPU 设备编号。默认 0。",
        "default": 0
      }
    },
    "required": []
  }
}
```

**成功响应**：

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "result": {
    "status": "ok",
    "device": {
      "device_id": 0,
      "name": "NVIDIA A100-SXM4-80GB",
      "compute_capability": "8.0",
      "sm_count": 108,
      "max_threads_per_sm": 2048,
      "max_threads_per_block": 1024,
      "warp_size": 32,
      "max_registers_per_thread": 255,
      "shared_memory_per_sm_bytes": 167936,
      "global_memory_bytes": 85899345920
    }
  }
}
```

---

## 3. 异步通知事件

异步通知是服务器主动推送给 Agent 的事件，**没有 `id` 字段**，Agent 无需回复。

### 3.1 cuda_stop_event

**触发条件**：目标程序命中断点、触发 CUDA 硬件异常、或收到信号而停止。

```json
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
      "hardware": {
        "device": 0,
        "sm": 7,
        "warp": 3,
        "lane": 15
      }
    },
    "pc": "0x555555557a80",
    "source_location": {
      "file": "matmul.cu",
      "line": 42
    },
    "cuda_exception": null
  }
}
```

**带 CUDA 异常的停止事件**：

```json
{
  "jsonrpc": "2.0",
  "method": "cuda_stop_event",
  "params": {
    "reason": "signal",
    "signal_name": "CUDA_EXCEPTION",
    "current_focus": {
      "kernel": "matmul_kernel",
      "block": [5, 0, 0],
      "thread": [31, 0, 0],
      "hardware": {
        "device": 0,
        "sm": 12,
        "warp": 7,
        "lane": 31
      }
    },
    "pc": "0x555555557a84",
    "source_location": {
      "file": "matmul.cu",
      "line": 45
    },
    "cuda_exception": {
      "errorpc": "0x555555557a80",
      "hint": "CUDA hardware exception detected. Call cuda_analyze_exception for detailed analysis."
    }
  }
}
```

### 3.2 cuda_exit_event

**触发条件**：目标程序正常退出或被信号终止。

```json
{
  "jsonrpc": "2.0",
  "method": "cuda_exit_event",
  "params": {
    "exit_code": 0,
    "message": "Program exited normally"
  }
}
```

### 3.3 cuda_error_event

**触发条件**：RPC 引擎内部发生非请求相关的错误。

```json
{
  "jsonrpc": "2.0",
  "method": "cuda_error_event",
  "params": {
    "error": "cuda-gdb internal assertion failure",
    "severity": "critical",
    "hint": "The debugger may be in an inconsistent state. Consider restarting the session."
  }
}
```

---

## 4. 错误码规范

| 错误码     | 常量名                 | 含义          | 典型场景                               |
| ---------- | ---------------------- | ------------- | -------------------------------------- |
| `-32700` | `PARSE_ERROR`        | JSON 解析错误 | 请求不是合法 JSON                      |
| `-32600` | `INVALID_REQUEST`    | 无效请求      | 缺少 `jsonrpc` 或 `method` 字段    |
| `-32601` | `METHOD_NOT_FOUND`   | 方法不存在    | 未知的工具方法名                       |
| `-32602` | `INVALID_PARAMS`     | 参数无效      | 参数类型错误、缺少必要参数、值超出范围 |
| `-32603` | `INTERNAL_ERROR`     | 内部错误      | RPC 引擎 Python 异常                   |
| `-32000` | `GDB_ERROR`          | GDB 内部错误  | `gdb.error` 异常（通用）             |
| `-32001` | `TIMEOUT`            | 命令超时      | 命令执行超过配置的超时阈值             |
| `-32002` | `PROCESS_CRASHED`    | 进程崩溃      | cuda-gdb 进程异常退出                  |
| `-32003` | `MODALITY_FORBIDDEN` | 模态禁止      | Coredump 模式下尝试执行控制流命令      |
| `-32004` | `TARGET_RUNNING`     | 目标运行中    | 目标正在运行，需先暂停才能执行查询     |
| `-32005` | `OPTIMIZED_OUT`      | 变量被优化    | 变量被 CUDA 编译器优化掉               |
| `-32006` | `NO_ACTIVE_KERNEL`   | 无活跃内核    | 当前无 CUDA 内核在 GPU 上执行          |
| `-32007` | `MEMORY_TRUNCATED`   | 内存裁剪      | Coredump 文件中该内存段被裁剪          |

---

## 5. 通用响应元数据

### 5.1 模态信息

每个成功响应可选地包含 `_meta` 字段，提供当前会话的元信息：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "status": "ok",
    "data": { ... },
    "_meta": {
      "mode": "IMMUTABLE",
      "target_type": "cudacore",
      "cuda_gdb_version": "13.2",
      "gdb_version": "16.3"
    }
  }
}
```

### 5.2 响应体积控制

为防止大数组或深层结构体撑爆 Agent 的上下文窗口，所有响应遵循以下约束：

| 约束项             | 限制值    | 超出时行为                                 |
| ------------------ | --------- | ------------------------------------------ |
| 数组最大元素数     | 256       | 截断并附加 `meta.truncated: true`        |
| 字符串最大长度     | 4096 字符 | 截断并附加 `meta.string_truncated: true` |
| 结构体最大递归深度 | 5 层      | 截断并返回 `<max_depth_exceeded>`        |
| 单个响应最大体积   | 1 MB      | 返回错误并建议缩小查询范围                 |

### 5.3 Agent 使用建议

以下建议嵌入在 MCP Tool 的 description 中，引导 Agent 正确使用工具：

1. **焦点优先**：在调用 `cuda_evaluate_var` 或 `cuda_dump_warp_registers` 前，始终先确认或设置正确的焦点（`cuda_set_focus`）
2. **异常优先**：收到 `cuda_stop_event` 通知且包含 `cuda_exception` 时，优先调用 `cuda_analyze_exception`
3. **增量查询**：避免一次性请求大量数据，按需逐步查询
4. **Warp 感知**：执行 step/next 后，同 Warp 内所有 32 个线程的状态都已改变
5. **模态感知**：Coredump 模式下不要尝试执行控制流命令
6. **编译标志**：如果遇到大量 `optimized_out` 错误，建议用户使用 `-g -G -O0` 重新编译
