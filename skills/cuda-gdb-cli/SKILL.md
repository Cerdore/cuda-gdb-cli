---
name: cuda-gdb-cli
description: |
  CUDA GPU debugging assistant that combines source code analysis with GPU runtime state.
  Use this skill when the user wants to:
  - Analyze CUDA core dumps or GPU crash dumps
  - Debug running CUDA processes with GDB attach
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

## Prerequisites

```bash
# Install cuda-gdb-cli
pip install cuda-gdb-cli

# Verify CUDA Toolkit is installed
cuda-gdb --version
nvcc --version
nvidia-smi
```

## Workflow

### Step 1: Initialize Debug Session

**For CUDA core dump:**
```bash
cuda-gdb-cli load --binary ./app --core ./core.12345
```

**For live CUDA process:**
```bash
cuda-gdb-cli attach --pid 9876 --binary ./app
```

Output:
```json
{"session_id": "f465d650", "mode": "core", "status": "started"}
```

### Step 2: Gather GPU Overview

```bash
SESSION="f465d650"

# GPU device information
cuda-gdb-cli cuda-devices -s $SESSION

# Active kernel list
cuda-gdb-cli cuda-kernels -s $SESSION

# GPU thread overview
cuda-gdb-cli cuda-threads -s $SESSION --limit 20

# CUDA exceptions
cuda-gdb-cli cuda-exceptions -s $SESSION
```

### Step 3: Focus on Crash Point

```bash
# CPU-side backtrace
cuda-gdb-cli bt -s $SESSION

# Switch to exception GPU thread
cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"

# View that thread's backtrace and locals
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
# Check shared memory contents
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "smem" --type float --count 32

# Check global memory
cuda-gdb-cli cuda-memory -s $SESSION --space global --expr "d_output" --type int --count 10

# View GPU registers
cuda-gdb-cli exec -s $SESSION "info registers" --safety-level readonly

# Disassemble crash point
cuda-gdb-cli disasm -s $SESSION --count 10

# Check multiple GPU threads' status
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

### Step 7: End Session

```bash
cuda-gdb-cli stop -s $SESSION
```

## Common CUDA Debugging Patterns

### Pattern: Illegal Memory Access

**Indicators:** `CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS`

**Investigation:**
1. Use `cuda-threads` to find threads with exceptions
2. Use `cuda-focus` to switch to that thread
3. Use `locals-cmd` to see index variables
4. Check array bounds in source code

**Common causes:**
- Out-of-bounds array access
- Null pointer dereference
- Accessing freed memory
- Incorrect pointer arithmetic

### Pattern: Warp Divergence Issues

**Indicators:** Performance problems, unexpected results

**Investigation:**
1. Use `cuda-warps` to check warp status
2. Check conditional branches in kernel code
3. Look for `if/else` within warps where threads take different paths

**Example:**
```cuda
// BAD: Divergent within warp
if (threadIdx.x % 2 == 0) {
    // slow path
} else {
    // fast path
}
```

### Pattern: Shared Memory Race Condition

**Indicators:** Non-deterministic results

**Investigation:**
1. Use `cuda-memory --space shared` to inspect shared memory
2. Check `__syncthreads()` placement in source code
3. Verify write-before-read ordering
4. Look for bank conflicts

**Example bug:**
```cuda
__shared__ float data[256];
data[tid] = value;          // Write
// MISSING: __syncthreads();
float other = data[other_tid]; // Race condition!
```

### Pattern: Kernel Launch Failure

**Indicators:** No GPU threads visible, kernel not executing

**Investigation:**
1. Use `cuda-kernels` to check if kernel launched
2. Check launch parameters (grid/block dimensions)
3. Verify CUDA API return codes on host side
4. Check GPU memory allocation status

### Pattern: Matrix Index Errors

**Indicators:** Wrong results or crashes in matrix operations

**Investigation:**
1. Check `blockIdx`, `threadIdx`, `blockDim` calculations
2. Verify row-major vs column-major ordering
3. Check boundary conditions for edge threads

**Example:**
```cuda
// Common bug: Off-by-one in boundary check
// BAD:
if (row <= M && col <= N)  // Should be <
// GOOD:
if (row < M && col < N)
```

## CUDA Address Spaces

| Space | Syntax | Description |
|-------|--------|-------------|
| Global | `@global` | Device memory, visible to all threads |
| Shared | `@shared` | Per-block shared memory, fast |
| Local | `@local` | Per-thread private memory |
| Generic | `@generic` | Default address space |

**Reading shared memory:**
```bash
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "tile_A" --type float --count 16
```

## CUDA Exception Types

| Exception | Description |
|-----------|-------------|
| `CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS` | Thread accessed invalid memory |
| `CUDA_EXCEPTION_WARP_ASSERT` | GPU `assert()` triggered |
| `CUDA_EXCEPTION_LANE_INVALID_PC` | Invalid program counter |
| `CUDA_EXCEPTION_WARP_MISALIGNED` | Misaligned memory access |
| `CUDA_EXCEPTION_HARDWARE_STACK_OVERFLOW` | Thread stack overflow |

## Tips and Best Practices

1. **Always check exceptions first** - Start with `cuda-exceptions` to find crash points
2. **Use filters** - `--kernel` and `--block` filters help narrow down large thread counts
3. **Correlate with source** - Always read the source code around crash points
4. **Check launch config** - Verify grid/block dimensions match expectations
5. **Inspect shared memory** - Race conditions often show up in shared memory state
6. **Use GPU coordinates** - Hardware coords (SM, warp, lane) help understand execution

## Error Handling

If you encounter errors:

```bash
# Check if cuda-gdb is installed
which cuda-gdb

# Check CUDA environment
nvidia-smi
nvcc --version

# Check if process exists (for attach)
ps aux | grep <process_name>

# Check if core file exists
file ./core.12345
```

## Example Debugging Session

```bash
# 1. Load coredump
$ cuda-gdb-cli load --binary ./matmul --core ./core.99421
{"session_id": "f465d650", "mode": "core", "status": "started"}

$ SESSION="f465d650"

# 2. GPU overview
$ cuda-gdb-cli cuda-devices -s $SESSION
{"devices": [{"device_id": 0, "name": "NVIDIA A100", "sm_count": 108}]}

$ cuda-gdb-cli cuda-kernels -s $SESSION
{"kernels": [{"id": 0, "function": "matmul_kernel", "grid_dim": [32,32,1], "block_dim": [16,16,1]}]}

# 3. Check exceptions
$ cuda-gdb-cli cuda-exceptions -s $SESSION
{"exceptions": [{"kernel": 0, "block": [0,0,0], "thread": [1,0,0], "type": "CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS"}]}

# 4. Switch to exception thread
$ cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"
{"focus": {"kernel": 0, "block_idx": [0,0,0], "thread_idx": [1,0,0]}, "frame": {"function": "matmul_kernel", "file": "matmul.cu", "line": 28}}

# 5. View backtrace and locals
$ cuda-gdb-cli bt -s $SESSION --full
{"frames": [{"number": 0, "function": "matmul_kernel", "file": "matmul.cu", "line": 28}]}

$ cuda-gdb-cli locals-cmd -s $SESSION
{"locals": {"row": 0, "col": 0, "N": 1024, "k": 1024}}

# 6. Evaluate expressions
$ cuda-gdb-cli eval-cmd -s $SESSION "row * N + k"
{"expression": "row * N + k", "value": 1024}

# 7. Check shared memory
$ cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "tile_A" --type float --count 16
{"space": "shared", "elements": [1.0, 2.0, 3.0, ...]}

# 8. End session
$ cuda-gdb-cli stop -s $SESSION
{"session_id": "f465d650", "status": "stopped"}
```

## Reference

- [CUDA-GDB Documentation](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)