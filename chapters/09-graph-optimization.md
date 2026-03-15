# 第 9 章：Graph 优化

> "TensorRT 的核心价值不在于执行计算，而在于在执行之前重新组织计算。图优化就是这种重组的艺术。"

前面几章我们详细分析了模型定义、编译流程和量化技术。本章将聚焦于 TensorRT 在 engine 构建过程中执行的图优化策略——正是这些优化将一个朴素的计算图转化为高度优化的推理引擎，实现了数倍甚至数十倍的性能提升。

## 9.1 TensorRT 图优化概述

当 `trtllm-build` 调用 `builder.build_serialized_network()` 时，TensorRT 会对整个计算图进行多轮优化。这些优化可以归纳为以下几个大类：

1. **Layer Fusion**（层融合）：将多个连续算子合并为一个 kernel
2. **Op Reorder**（算子重排）：调整算子执行顺序以提升数据局部性
3. **Constant Folding**（常量折叠）：在编译期预计算常量表达式
4. **Kernel Selection**（核函数选择）：为每一层选择最快的 CUDA kernel 实现
5. **Memory Optimization**（内存优化）：优化中间 tensor 的内存分配和复用

这些优化是 TensorRT 作为推理引擎的核心竞争力。与 PyTorch 等训练框架的即时执行不同，TensorRT 利用编译期的全局视角，进行跨层的深度优化。

## 9.2 Layer Fusion：多算子合并

Layer Fusion 是 TensorRT 最重要的优化手段之一。它将多个相邻的算子合并为一个 CUDA kernel 执行，消除了中间结果对全局内存（Global Memory/HBM）的读写操作。

### 常见的 Fusion 模式

**Conv/GEMM + Bias + Activation Fusion**：这是最经典的融合模式。在 LLM 中，线性投影层后通常跟着偏置加法和激活函数：

```python
# 融合前：3 个独立的 kernel 调用
x = linear(x, weight)      # GEMM kernel，结果写回 HBM
x = x + bias               # 逐元素加法 kernel，结果写回 HBM
x = gelu(x)                # 激活函数 kernel

# 融合后：1 个 kernel 调用
x = fused_gemm_bias_gelu(x, weight, bias)  # 结果只写回 HBM 一次
```

这种融合的性能收益主要来自减少 HBM 访问。以 hidden_size=4096 的线性层为例，融合可以节省约 32 KB × batch_size 的 HBM 读写量，在大批量场景下效果显著。

**Residual + LayerNorm Fusion**：Transformer 每个子层都有残差连接和归一化操作。TensorRT 可以将这两个操作融合：

```python
# 融合前
x = x + residual            # 残差加法
x = layer_norm(x, gamma, beta)  # 归一化

# 融合后
x = fused_residual_layernorm(x, residual, gamma, beta)
```

**Multi-Head Attention 内部融合**：GPTAttention plugin 本身就是一个大型融合操作，将 QKV 重排、RoPE、softmax、attention 计算等多个步骤融合为一个 kernel。这不是 TensorRT 自动完成的融合，而是通过 plugin 机制手动实现的"超级融合"。

### TRT-LLM 特有的 Pattern Matching

TensorRT-LLM 在标准 TRT 优化之上，还实现了针对 LLM 的特定模式匹配优化。这些优化在构建 TRT Network 时就已经完成，位于 Python 层面：

```python
# tensorrt_llm 内部的图优化示例
# 识别 SwiGLU 模式并替换为融合实现
# 原始模式: gate = silu(linear1(x)); up = linear2(x); output = gate * up
# 优化后: output = fused_swiglu(x, weight1, weight2)
```

这种 Python 层面的 pattern matching 与 TensorRT 引擎层面的 Layer Fusion 互为补充——前者处理高层语义模式，后者处理底层算子融合。

## 9.3 Op Reorder：算子重排

Op Reorder 通过调整算子的执行顺序来提升数据局部性和并行度。TensorRT 在保持计算语义不变的前提下，可能会：

- **将数据格式转换操作前移或后移**：TensorRT 内部可能使用不同于用户指定的数据布局（如 NHWC vs NCHW）。通过重排格式转换操作的位置，可以减少不必要的 transpose。

- **重排独立算子的执行顺序**：如果两个算子之间没有数据依赖，TensorRT 可以选择更优的执行顺序，使得后续的 Layer Fusion 有更多的机会触发。

在 LLM 的 Transformer 结构中，Op Reorder 的一个典型应用是 **AllReduce 与计算的重叠**。在张量并行场景下，AllReduce 通信操作可以与后续不依赖其结果的计算操作并行执行：

```python
# 重排前（串行执行）
attn_output = attention(x)            # 计算
all_reduce(attn_output)               # 通信（等待）
ffn_input = layer_norm(attn_output)   # 计算

# 重排后（通信与计算重叠）
attn_output = attention(x)            # 计算
all_reduce_async(attn_output)         # 异步通信
# 在等待通信完成的同时，可以执行其他不依赖 attn_output 的计算
```

## 9.4 Constant Folding：编译期常量计算

Constant Folding 将编译时可确定结果的计算提前完成，避免在推理时重复计算。在 LLM 中常见的场景包括：

- **位置编码的预计算**：RoPE 的频率矩阵在推理时固定，可以在编译期预计算
- **量化参数的合并**：将多个连续的 scale/zero-point 操作合并为一个
- **常量 reshape/transpose**：对常量 tensor 的形状变换在编译期完成

```python
# Constant Folding 示例
# 编译前的计算图
freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))  # 常量表达式
cos_cached = torch.cos(freq * positions)                    # 常量表达式

# 编译后：cos_cached 直接作为常量嵌入 engine，推理时零计算开销
```

TensorRT 的 constant folding 是自动的——任何输入为常量的子图都会被求值并替换为结果常量。

## 9.5 BuilderConfig 的优化相关参数

`tensorrt_llm/builder.py` 中的 `BuilderConfig` 暴露了多个影响优化行为的参数：

```python
# builder 优化级别
builder_config = builder.create_builder_config()

# optimization level: 0-5，级别越高优化越激进，编译时间越长
# 默认为 3，推荐生产环境使用 3 或 4
config.builder_optimization_level = 3

# workspace size: TRT 优化时可使用的临时显存
# 更大的 workspace 允许更多的 kernel 选项
config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    4 * 1024 * 1024 * 1024  # 4 GB
)
```

`builder_optimization_level` 的各级别含义：

- **Level 0-1**：最少优化，编译速度快，适合开发调试
- **Level 2-3**：中等优化，是编译时间与推理性能的良好平衡
- **Level 4-5**：最大化优化，会尝试更多的 kernel 变体和融合策略，编译时间可能翻倍

在 `trtllm-build` 中通过 `--builder_opt` 参数设置：

```python
trtllm-build \
    --checkpoint_dir ./model-checkpoint \
    --output_dir ./model-engine \
    --builder_opt 4
```

## 9.6 Kernel Profiling：自动选择最优 Kernel

TensorRT 图优化的最后阶段是 **kernel auto-tuning**。对于每一层操作，TensorRT 会维护多种 CUDA kernel 实现（不同的 tile size、不同的数据类型路径、不同的算法变体），并在目标 GPU 上实际运行每一种，选择耗时最短的。

这个过程称为 **profiling** 或 **tactic selection**，是编译耗时的主要来源。以一个 7B 参数的模型为例：

- 每层有多种 GEMM 配置（不同的 M/N/K 组合对应不同的 tactic）
- 每种 tactic 需要实际运行数次取平均
- 全部 32 层 × 多个 GEMM × 多种 tactic = 数千次 kernel 执行

对于 FP16/BF16/FP8 等不同精度，可用的 kernel 集合不同。TensorRT 会在约束条件下（如精度要求、显存限制）搜索最优解：

```python
# TensorRT 的 kernel 选择过程（概念性描述）
for layer in network.layers:
    best_tactic = None
    best_time = float('inf')
    for tactic in get_available_tactics(layer):
        time = profile_tactic(tactic, layer)
        if time < best_time:
            best_time = time
            best_tactic = tactic
    layer.set_tactic(best_tactic)
```

## 9.7 Timing Cache：加速重复编译

由于 kernel profiling 非常耗时，TensorRT 提供了 **Timing Cache** 机制来缓存 profiling 结果。当同一个 layer 配置（相同的算子类型、输入形状、数据类型）再次出现时，直接从 cache 中读取最优 tactic，无需重新 profiling。

```python
# 使用 timing cache 加速编译
trtllm-build \
    --checkpoint_dir ./model-checkpoint \
    --output_dir ./model-engine \
    --timing_cache ./timing_cache.bin
```

Timing Cache 的工作机制：

1. **首次编译**：所有层都需要 profiling，耗时较长（可能 30-60 分钟）
2. **缓存写入**：profiling 结果写入 `timing_cache.bin`
3. **再次编译**：如果模型结构和参数不变，大部分层可以命中 cache，编译时间大幅缩短（可能 5-10 分钟）

Timing Cache 的命中条件非常严格——不仅算子类型和形状要匹配，GPU 型号、驱动版本、TensorRT 版本也必须一致。因此 timing cache 文件不应跨机器共享，除非硬件和软件环境完全相同。

```python
# 在 Python API 中使用 timing cache
from tensorrt_llm.builder import Builder

builder = Builder()
# 加载已有的 timing cache
builder.load_timing_cache("./timing_cache.bin")
# 编译
engine = builder.build_engine(network, build_config)
# 保存更新后的 timing cache
builder.save_timing_cache("./timing_cache.bin")
```

## 9.8 Engine 构建耗时分析与优化建议

理解编译耗时的构成有助于针对性地优化：

| 阶段 | 占比 | 可优化空间 |
|------|------|-----------|
| Checkpoint 加载 | 5-10% | 使用 SSD 或 ramdisk |
| Network 构建 | 5-10% | 代码优化（较少） |
| Graph 优化 | 10-15% | 降低 builder_opt level |
| Kernel Profiling | 60-75% | Timing Cache 命中 |
| 序列化输出 | 5% | I/O 优化 |

**优化建议**：

1. **善用 Timing Cache**：这是最有效的编译加速手段。在 CI/CD 流水线中，应该将 timing cache 作为构建产物保留。

2. **合理设置 builder_opt level**：开发阶段使用 level 2，生产部署使用 level 3-4。Level 5 的收益通常很小，但编译时间显著增加。

3. **减少 optimization profile 数量**：每个 profile 都需要独立的 profiling。如果可以确定运行时的输入形状范围，尽量缩小 min/opt/max 的差距。

4. **并行编译多 rank engine**：多 GPU 模型的各 rank engine 之间互相独立，可以在多张 GPU 上同时编译：

```python
# 使用 mpirun 在 4 张 GPU 上并行编译
mpirun -n 4 --allow-run-as-root \
    trtllm-build \
    --checkpoint_dir ./model-checkpoint \
    --output_dir ./model-engine \
    --tp_size 4 \
    --timing_cache ./timing_cache.bin
```

5. **增大 workspace**：更大的 workspace 允许 TensorRT 考虑更多的 kernel 变体，可能找到更优的实现。但要注意不要超过 GPU 可用显存。

6. **关注 TensorRT 版本升级**：新版本的 TensorRT 通常会引入新的 fusion pattern 和更优的 kernel 实现。升级 TensorRT 版本可能在不修改任何代码的情况下获得性能提升，但需要注意旧的 timing cache 可能不再兼容。

## 本章小结

本章深入分析了 TensorRT 在 engine 构建过程中执行的图优化策略。Layer Fusion 通过合并多个算子消除冗余内存访问；Op Reorder 通过重排算子顺序提升数据局部性；Constant Folding 将常量计算提前到编译期；Kernel Profiling 通过实际测量确保每一层使用最优的 CUDA kernel。Timing Cache 机制则有效缓解了 profiling 带来的编译耗时问题。这些优化共同构成了 TensorRT 的性能护城河——同样的模型，经过 TensorRT 编译后的推理速度通常是 PyTorch eager mode 的 2-5 倍。理解这些优化机制，不仅有助于调试编译问题，更能指导我们写出对优化器更友好的模型代码。
