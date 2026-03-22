# CUDA-GDB CLI — 基于 gdb-cli 扩展的 CUDA 调试工具

基于 [gdb-cli](https://github.com/Cerdore/gdb-cli) fork 扩展，让 AI Agent（Claude Code / Codex CLI / Cursor 等）通过 bash 命令调试 CUDA GPU 程序。

## 设计策略

**Fork gdb-cli，复用 80% 基础设施，专注 CUDA 差异化的 20%。**

```bash
SESSION="f465d650"

# 加载 CUDA coredump
cuda-gdb-cli load --binary ./app --core ./core.12345

# CPU 端调试（继承 gdb-cli 全部能力）
cuda-gdb-cli bt -s $SESSION
cuda-gdb-cli eval-cmd -s $SESSION "my_var"

# CUDA GPU 调试（新增能力）
cuda-gdb-cli cuda-kernels -s $SESSION
cuda-gdb-cli cuda-threads -s $SESSION --kernel 0
cuda-gdb-cli cuda-focus -s $SESSION --block "0,0,0" --thread "1,0,0"
cuda-gdb-cli cuda-exceptions -s $SESSION
cuda-gdb-cli cuda-memory -s $SESSION --space shared --expr "smem" --type float --count 32

# 结束
cuda-gdb-cli stop -s $SESSION
```

## 文档

| 文件                    | 内容         |
| ----------------------- | ------------ |
| [`design.md`](design.md) | 完整设计方案 |

## 快速导航

1. **设计策略：为什么 fork gdb-cli** → `design.md` 第 0 章
2. **架构（继承 thin CLI + 内嵌 RPC）** → `design.md` 第 1 章
3. **新增 CUDA 子命令设计** → `design.md` 第 2 章
4. **CUDA Handler 实现** → `design.md` 第 3 章
5. **Launcher 改动（gdb → cuda-gdb）** → `design.md` 第 4 章
6. **Value Formatter 扩展（地址空间）** → `design.md` 第 5 章
7. **Safety 扩展（CUDA 命令白名单）** → `design.md` 第 6 章
8. **CLI 命令注册** → `design.md` 第 7 章
9. **Claude Code Skill 定义** → `design.md` 第 8 章
10. **项目结构** → `design.md` 第 9 章
11. **Agent 使用示例** → `design.md` 第 10 章
12. **改动清单** → `design.md` 第 11 章
13. **演进路线** → `design.md` 第 12 章
