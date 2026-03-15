# 第 7 章：TensorRT 插件体系

> "插件是 TensorRT 的扩展点，是自定义算子进入推理引擎的唯一通道。对于 LLM 推理而言，没有插件就没有 Flash Attention，没有 Paged KV Cache，也就没有真正的高性能。"

TensorRT 作为通用推理优化引擎，其内置算子覆盖了绝大多数标准神经网络操作。然而 LLM 推理有许多特殊需求——融合注意力机制、自定义量化核函数、高效的 LayerNorm 实现——这些都需要通过插件（Plugin）机制来扩展。本章将深入分析 TensorRT-LLM 的插件体系架构。

## 7.1 TensorRT Plugin 机制概述

TensorRT 的 plugin 机制允许开发者将自定义的 CUDA kernel 注册为 TRT 算子，参与到整个图优化和 engine 构建流程中。其核心接口是 `IPluginV2DynamicExt`，这是一个支持动态形状的插件基类。

一个完整的 TensorRT plugin 需要实现以下关键方法：

```cpp
class MyPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    // 根据输入形状推导输出形状
    DimsExprs getOutputDimensions(int outputIndex,
        const DimsExprs* inputs, int nbInputs,
        IExprBuilder& exprBuilder) override;

    // 推导输出数据类型
    DataType getOutputDataType(int index,
        const DataType* inputTypes, int nbInputs) override;

    // 配置插件（在 engine 构建时调用）
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs,
        const DynamicPluginTensorDesc* out, int nbOutputs) override;

    // 执行推理（实际运行 CUDA kernel）
    int enqueue(const PluginTensorDesc* inputDesc,
        const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) override;
};
```

`enqueue()` 方法是插件的核心——它在推理时被调用，负责启动实际的 CUDA kernel 执行计算。

## 7.2 TRT-LLM 自定义 Plugin 目录结构

TensorRT-LLM 的所有自定义 plugin 源码位于 `cpp/tensorrt_llm/plugins/` 目录下，组织结构清晰：

```
cpp/tensorrt_llm/plugins/
├── api/
│   └── tllmPlugin.cpp           # Plugin 注册总入口
├── gptAttentionPlugin/
│   ├── gptAttentionPlugin.h
│   └── gptAttentionPlugin.cpp   # GPT 注意力融合插件
├── layernormPlugin/
│   ├── layernormPlugin.h
│   └── layernormPlugin.cpp      # LayerNorm 插件
├── rmsnormPlugin/
│   ├── rmsnormPlugin.h
│   └── rmsnormPlugin.cpp        # RmsNorm 插件
├── smoothQuantGemmPlugin/
│   ├── smoothQuantGemmPlugin.h
│   └── smoothQuantGemmPlugin.cpp # SmoothQuant 矩阵乘法
├── weightOnlyQuantMatmulPlugin/
│   ├── weightOnlyQuantMatmulPlugin.h
│   └── weightOnlyQuantMatmulPlugin.cpp # 仅权重量化矩阵乘法
├── quantizePerTokenPlugin/
│   └── ...                      # Per-token 动态量化
├── quantizeTensorPlugin/
│   └── ...                      # Tensor 级别量化
└── common/
    └── ...                      # 公共工具代码
```

每个子目录对应一个 plugin，包含头文件和实现文件。这种模块化设计使得添加新 plugin 时不会影响已有代码。

## 7.3 GPTAttention Plugin 详解

`GPTAttention plugin` 是 TensorRT-LLM 中最重要也最复杂的插件，位于 `cpp/tensorrt_llm/plugins/gptAttentionPlugin/`。它将多头注意力机制的多个步骤融合为一个高效的 CUDA kernel，避免了中间结果的显存读写。

该插件融合了以下操作：

1. **QKV 重排**：将线性投影后的 QKV tensor 重排为注意力计算所需的布局
2. **RoPE 位置编码**：对 Q 和 K 应用旋转位置编码
3. **KV Cache 管理**：将当前步的 K、V 写入 cache，并读取历史 cache
4. **Attention 计算**：执行 scaled dot-product attention（支持 Flash Attention 和 Paged Attention）
5. **输出投影准备**：将注意力输出重排为后续层所需的形状

```cpp
// cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.cpp
int GPTAttentionPlugin::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream)
{
    // 根据配置选择不同的 attention kernel
    if (mPagedKVCache) {
        // Paged KV Cache 路径：支持动态内存管理
        runPagedAttention(inputs, outputs, workspace, stream);
    } else {
        // 连续 KV Cache 路径
        runContiguousAttention(inputs, outputs, workspace, stream);
    }
    return 0;
}
```

GPTAttention plugin 的一个关键特性是支持 **Paged KV Cache**。与传统的连续内存分配不同，Paged KV Cache 将 cache 划分为固定大小的 block，按需分配和回收，极大提升了显存利用率。这对于大批量推理场景至关重要。

该插件还支持多种注意力变体：Multi-Head Attention（MHA）、Multi-Query Attention（MQA）、Grouped-Query Attention（GQA），通过 `num_kv_heads` 参数区分。

## 7.4 LayerNorm 与 RmsNorm Plugin

归一化层是 Transformer 中的高频操作。TensorRT-LLM 为 LayerNorm 和 RmsNorm 分别提供了优化的 plugin 实现。

**LayerNorm plugin**（`cpp/tensorrt_llm/plugins/layernormPlugin/`）融合了均值计算、方差计算、归一化和仿射变换四个步骤，使用单个 CUDA kernel 完成。关键优化包括：

- 利用 warp-level reduction 高效计算均值和方差
- 支持 FP16/BF16 输入与 FP32 累加，兼顾精度与性能
- 支持 residual connection 融合，避免额外的显存访问

**RmsNorm plugin**（`cpp/tensorrt_llm/plugins/rmsnormPlugin/`）是 LLaMA 等模型使用的归一化方式，省去了均值计算步骤，计算量更少：

```cpp
// RmsNorm 的核心计算逻辑
// output = x * rsqrt(mean(x^2) + eps) * gamma
```

这两个 plugin 看似简单，但在实际推理中，由于归一化操作出现在每一层的前后，其累积性能影响不可忽视。

## 7.5 量化相关 Plugin

量化 plugin 是 TensorRT-LLM 实现低精度推理的核心组件：

**SmoothQuant GEMM Plugin**（`cpp/tensorrt_llm/plugins/smoothQuantGemmPlugin/`）实现了 SmoothQuant 论文提出的 INT8 矩阵乘法。该 plugin 在矩阵乘法前对激活值进行 per-token 动态量化，对权重进行 per-channel 静态量化，然后调用 INT8 GEMM kernel：

```cpp
// SmoothQuant GEMM 的简化流程
// 1. 激活值 per-token 量化: X_int8 = quantize(X, scale_x)
// 2. INT8 矩阵乘法: Y_int32 = X_int8 @ W_int8
// 3. 反量化: Y = dequantize(Y_int32, scale_x, scale_w)
```

**WeightOnly Quant Plugin**（`cpp/tensorrt_llm/plugins/weightOnlyQuantMatmulPlugin/`）支持 INT8 和 INT4 的仅权重量化。权重在编译时量化为低精度存储，推理时在 GEMM kernel 内部反量化为 FP16 进行计算。这种方式减少了显存占用和带宽消耗，特别适合 decode 阶段的 memory-bound 场景。

## 7.6 Plugin 注册机制

TensorRT 使用全局注册表管理所有 plugin。TRT-LLM 通过 `REGISTER_TENSORRT_PLUGIN` 宏将自定义 plugin 注册到 TRT 的 plugin registry 中：

```cpp
// 每个 plugin 文件末尾的注册代码
static auto const kGPT_ATTENTION_PLUGIN_NAME = "GPTAttention";
static auto const kGPT_ATTENTION_PLUGIN_VERSION = "1";

class GPTAttentionPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const override {
        return kGPT_ATTENTION_PLUGIN_NAME;
    }
    const char* getPluginVersion() const override {
        return kGPT_ATTENTION_PLUGIN_VERSION;
    }
    // ...
};

REGISTER_TENSORRT_PLUGIN(GPTAttentionPluginCreator);
```

注册后，TRT 在反序列化 engine 或构建 network 时，可以通过 plugin 名称和版本号自动找到对应的 plugin 实现。`REGISTER_TENSORRT_PLUGIN` 宏的本质是利用 C++ 的静态初始化机制，在动态库加载时自动完成注册。

所有 plugin 的注册汇总在 `cpp/tensorrt_llm/plugins/api/tllmPlugin.cpp` 中，该文件通过 `initTrtLlmPlugins()` 函数确保所有 plugin 在使用前被正确初始化。

## 7.7 PluginConfig Python 类

在 Python 层面，`PluginConfig` 类定义在 `tensorrt_llm/plugin/plugin.py` 中，负责控制编译时哪些 plugin 被启用：

```python
# tensorrt_llm/plugin/plugin.py
class PluginConfig:
    def __init__(self):
        self.gpt_attention_plugin = None    # 'float16', 'bfloat16', or None
        self.gemm_plugin = None             # 'float16', 'bfloat16', or None
        self.nccl_plugin = None             # 多 GPU 通信
        self.layernorm_plugin = None        # LayerNorm 加速
        self.rmsnorm_plugin = None          # RmsNorm 加速
        self.smooth_quant_gemm_plugin = None
        self.weight_only_quant_matmul_plugin = None
        self.context_fmha = True            # Flash Attention in context phase
        self.paged_kv_cache = True          # Paged KV Cache
```

当某个 plugin 字段为 `None` 时，对应的操作会使用 TensorRT 原生算子实现；设为精度字符串（如 `"float16"`）时，则使用自定义 plugin。

## 7.8 Plugin vs 原生 TRT 算子：何时使用

并非所有场景都应该使用 plugin。选择的基本原则是：

**优先使用 plugin 的场景**：
- **注意力层**：GPTAttention plugin 融合度极高，性能远超原生算子组合，几乎总是应该启用
- **量化计算**：SmoothQuant、WeightOnly 等量化 GEMM 只能通过 plugin 实现
- **多 GPU 通信**：NCCL plugin 封装了 AllReduce 等集合通信操作

**可以使用原生 TRT 算子的场景**：
- **简单逐元素操作**：如 activation（GELU、SiLU），TRT 原生实现已经足够高效
- **标准 GEMM**：在非量化场景下，TRT 原生的 cuBLAS 调用性能与 plugin 接近
- **调试阶段**：关闭 plugin 可以更容易地定位精度问题

在生产部署中，推荐的最佳实践是启用 `gpt_attention_plugin` 和 `gemm_plugin`，同时根据量化需求启用对应的量化 plugin。这一组合在绝大多数 LLM 推理场景中能提供最优性能。

## 本章小结

本章系统分析了 TensorRT-LLM 的插件体系。从 TensorRT 的 `IPluginV2DynamicExt` 接口出发，我们深入了解了 GPTAttention、LayerNorm/RmsNorm、量化 GEMM 等核心 plugin 的设计与实现。Plugin 注册机制通过 `REGISTER_TENSORRT_PLUGIN` 宏实现自动化，而 Python 层的 `PluginConfig` 类提供了灵活的启用/禁用控制。理解插件体系是进行性能优化和定制开发的关键——当内置算子无法满足需求时，plugin 是将自定义 CUDA kernel 无缝集成到 TensorRT 推理流程中的标准途径。
