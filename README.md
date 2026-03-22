# CUDA-GDB-CLI RPC 服务架构设计方案

面向 AI Agent 的 CUDA-GDB-CLI RPC 服务系统详细设计文档。

## 文档结构

| 文件              | 内容                                       |
| ----------------- | ------------------------------------------ |
| `design.md`     | 完整的架构设计方案（主文档）               |
| `api-schema.md` | Agent Tool Schema 与 JSON-RPC 接口详细规范 |
| `src/`          | 核心模块参考实现骨架代码                   |

## 快速导航

1. **整体架构** → `design.md` 第 1-3 章
2. **线程隔离与无阻塞通信** → `design.md` 第 4 章
3. **GPU 特有对象访问机制** → `design.md` 第 5 章
4. **模态守卫与状态机** → `design.md` 第 6 章
5. **API 接口规范** → `api-schema.md`
6. **参考实现** → `src/`
