# 附录 A：推荐阅读路径

本书涵盖了 TensorRT-LLM 从入门到深度源码分析的各个方面。不同背景和目标的读者，可以根据自身需求选择最适合的阅读路径，而不必逐章顺序通读全书。以下提供四条典型路径供参考。

---

## 路径一：快速上手路径

**适合读者：** 希望尽快将 TensorRT-LLM 用起来的应用工程师、算法工程师。

**章节顺序：** 第 1 章 → 第 5 章 → 第 6 章 → 第 10 章

| 章节 | 内容 | 选取理由 |
|------|------|----------|
| 第 1 章：总体概览 | TensorRT-LLM 的架构全貌、核心概念与代码仓库结构 | 建立整体认知，了解 `tensorrt_llm/` 顶层目录组织方式 |
| 第 5 章：模型定义与权重加载 | 以 LLaMA 为例讲解 `tensorrt_llm/models/llama/model.py` 中的模型定义，以及 `convert_checkpoint.py` 的权重转换流程 | 掌握如何将 HuggingFace 模型转换为 TensorRT-LLM 可用的 checkpoint |
| 第 6 章：Engine 构建流程 | `trtllm-build` 命令的完整调用链，从 `tensorrt_llm/commands/build.py` 入口到 TensorRT Engine 序列化 | 学会构建 Engine，这是使用 TensorRT-LLM 最核心的一步 |
| 第 10 章：Python Runtime 与使用示例 | `tensorrt_llm/runtime/generation.py` 中的推理接口、`examples/` 下的典型用法 | 能够加载 Engine 并执行推理，跑通端到端流程 |

**预期成果：** 读完这四章后，读者可以独立完成「模型转换 → Engine 构建 → 推理部署」的完整流程。

---

## 路径二：深度理解路径

**适合读者：** 希望深入理解 TensorRT-LLM 内部实现机制的框架开发者、研究人员。

**章节顺序：** 第 1 章 → 第 2 章 → 第 3 章 → ... → 第 19 章（全部章节顺序阅读）

| 阶段 | 章节范围 | 核心内容 |
|------|----------|----------|
| 基础篇 | 第 1 ~ 4 章 | TensorRT 基础回顾、TensorRT-LLM 架构概览、Graph Rewriting 机制、`tensorrt_llm/graph_rewriting.py` 中的 pattern matching |
| 模型篇 | 第 5 ~ 7 章 | 模型定义体系 (`tensorrt_llm/models/`)、Engine 构建流水线、Plugin 系统 (`cpp/tensorrt_llm/plugins/`) |
| 运行时篇 | 第 8 ~ 11 章 | C++ Runtime (`cpp/tensorrt_llm/runtime/`)、Batch Manager、调度策略、Python Runtime 封装 |
| 内核篇 | 第 12 ~ 14 章 | Attention Kernel 实现 (`cpp/tensorrt_llm/kernels/`)、GEMM 优化、量化内核 |
| 分布式篇 | 第 15 ~ 16 章 | Tensor Parallelism / Pipeline Parallelism 的通信原语、`cpp/tensorrt_llm/kernels/customAllReduceKernels/` |
| 工程篇 | 第 17 ~ 19 章 | Triton 集成、性能分析方法论、生产环境最佳实践 |

**预期成果：** 全面掌握 TensorRT-LLM 从上层 Python API 到底层 CUDA Kernel 的完整实现，具备参与框架开发和提交 PR 的能力。

---

## 路径三：部署运维路径

**适合读者：** 负责将 LLM 推理服务部署到生产环境的 DevOps 工程师、SRE。

**章节顺序：** 第 1 章 → 第 6 章 → 第 8 章 → 第 16 章 → 第 17 章 → 第 19 章

| 章节 | 内容 | 选取理由 |
|------|------|----------|
| 第 1 章：总体概览 | 架构全貌与核心概念 | 理解系统边界，明确各组件的职责划分 |
| 第 6 章：Engine 构建流程 | `trtllm-build` 的参数体系与构建选项 | 掌握 `BuildConfig` 中影响部署的关键参数（如 `max_batch_size`、`max_input_len`），参见附录 B |
| 第 8 章：C++ Runtime 架构 | `cpp/tensorrt_llm/runtime/gptSession.cpp` 中的会话管理、内存分配策略 | 了解 Runtime 的资源占用模式，便于容量规划 |
| 第 16 章：多卡分布式部署 | Tensor Parallelism 和 Pipeline Parallelism 的部署拓扑、`mpirun` 启动方式 | 掌握多节点多卡部署方案 |
| 第 17 章：Triton Inference Server 集成 | `tensorrtllm_backend/` 的配置、`model_config.pbtxt` 编写 | 生产环境通常通过 Triton 对外提供服务 |
| 第 19 章：生产环境最佳实践 | 监控指标、故障排查、滚动升级策略 | 保障服务稳定运行的运维要点 |

**预期成果：** 能够独立完成从单卡到多卡、从测试到生产的完整部署流程，并建立有效的监控和运维体系。

---

## 路径四：性能优化路径

**适合读者：** 希望榨取最大推理性能的性能工程师、CUDA 开发者。

**章节顺序：** 第 8 章 → 第 9 章 → 第 12 章 → 第 13 章 → 第 14 章 → 第 19 章

| 章节 | 内容 | 选取理由 |
|------|------|----------|
| 第 8 章：C++ Runtime 架构 | Runtime 内存管理、KV Cache 分配策略（`cpp/tensorrt_llm/runtime/kvCacheManager.cpp`） | 理解推理过程中的内存瓶颈与调度开销 |
| 第 9 章：Batch Manager 与调度 | In-flight Batching 实现（`cpp/tensorrt_llm/batch_manager/`）、请求调度策略 | Batching 策略直接决定吞吐量上限 |
| 第 12 章：Attention Kernel 深度剖析 | Flash Attention、Paged Attention 的 CUDA 实现（`cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/`） | Attention 是 LLM 推理中最关键的性能热点 |
| 第 13 章：GEMM 优化与量化 | CUTLASS 集成、FP8/INT8/INT4 量化内核（`cpp/tensorrt_llm/kernels/cutlass_kernels/`） | GEMM 占据推理计算量的主要部分，量化是提升吞吐的利器 |
| 第 14 章：自定义 Kernel 开发 | Plugin 开发流程、Kernel 注册机制（`cpp/tensorrt_llm/plugins/api/tllmPlugin.cpp`） | 当内置优化不够时，能够开发自定义算子 |
| 第 19 章：生产环境最佳实践 | 性能 Profiling 方法论、`nsys` / `ncu` 使用技巧、端到端基准测试 | 将优化落地，用数据验证效果 |

**预期成果：** 深入理解 TensorRT-LLM 性能关键路径，能够通过参数调优、量化策略选择和自定义 Kernel 开发来最大化推理性能。

---

> **提示：** 无论选择哪条路径，都建议先通读第 1 章建立全局认知。各路径之间并不互斥，读者可以在完成一条路径后，按需补充其他路径的章节。
