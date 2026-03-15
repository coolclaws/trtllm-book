# 第 2 章：Repo 结构全景

> "Show me your data structures, and I won't need your code. Show me your directory layout, and I won't need your architecture doc."
> —— 改编自 Fred Brooks

在深入源码之前，我们需要先建立对整个项目的全局认知。TensorRT-LLM 是一个典型的 Python + C++/CUDA 混合项目，代码规模庞大、模块众多。本章将带你从顶层目录开始，逐步建立对整个代码仓库的空间感。

## 2.1 顶层目录结构

克隆 `NVIDIA/TensorRT-LLM` 仓库后，顶层目录结构如下：

```
TensorRT-LLM/
├── tensorrt_llm/          # Python API 层（核心）
├── cpp/                   # C++ 运行时与 CUDA kernel（核心）
├── examples/              # 各模型的使用示例
├── scripts/               # 构建脚本与工具
├── 3rdparty/              # 第三方依赖（git submodule）
├── tests/                 # 测试套件
├── benchmarks/            # 性能基准测试
├── docker/                # Docker 构建文件
├── docs/                  # 文档
├── setup.py               # Python 包安装入口
├── CMakeLists.txt         # C++ 构建入口
└── README.md
```

从代码量的角度粗略估计，整个项目的构成大致为：

| 语言 | 占比（估算） | 主要分布 |
|------|------------|---------|
| Python | ~40% | `tensorrt_llm/`、`examples/`、`tests/` |
| C++ | ~35% | `cpp/` 下的运行时与插件 |
| CUDA | ~20% | `cpp/` 下的 kernel 实现 |
| 其他（CMake、Shell 等） | ~5% | `scripts/`、构建文件 |

Python 和 C++/CUDA 的代码量几乎对半分，这反映了 TensorRT-LLM 的双层架构设计：**Python 负责易用性，C++/CUDA 负责极致性能**。

## 2.2 tensorrt_llm/ —— Python API 层

这是用户直接接触最多的部分，提供了模型定义、编译构建、运行时调用的完整 Python 接口。

```
tensorrt_llm/
├── models/                # 各模型架构的 Python 定义
│   ├── llama/
│   │   ├── model.py       # LLaMA 模型结构定义
│   │   └── convert.py     # 权重转换逻辑
│   ├── gpt/
│   ├── chatglm/
│   ├── qwen/
│   ├── falcon/
│   └── ...                # 50+ 模型支持
├── runtime/               # Python 运行时
│   ├── generation.py      # 文本生成逻辑
│   └── model_runner.py    # 模型加载与执行
├── builder.py             # TRT engine 构建器
├── functional.py          # 函数式 API（类比 torch.nn.functional）
├── module.py              # 基础 Module 类（类比 torch.nn.Module）
├── layers/                # 通用层定义
│   ├── linear.py          # 线性层
│   ├── attention.py       # 注意力层
│   ├── embedding.py       # 嵌入层
│   └── moe.py             # Mixture of Experts 层
├── quantization/          # 量化相关
│   ├── quantize.py
│   └── mode.py
└── plugin/                # TRT plugin 的 Python 封装
    └── plugin.py
```

几个关键文件值得特别关注：

**`builder.py`** 是整个编译流程的入口。它负责将用户定义的模型转换为 TensorRT network，然后调用 TensorRT 编译器生成优化后的 engine。

```python
# tensorrt_llm/builder.py 中的核心流程（简化）
class Builder:
    def build_engine(self, model, build_config):
        # 1. 创建 TensorRT network
        network = tensorrt_llm.Network()

        # 2. 在 network 上下文中执行模型前向传播
        #    这一步不是真正的计算，而是构建计算图
        with network:
            inputs = model.prepare_inputs(...)
            model(**inputs)

        # 3. 调用 TensorRT 编译器优化并生成 engine
        engine = trt_builder.build_serialized_network(
            network.trt_network, config
        )
        return engine
```

**`functional.py`** 提供了所有基础算子的函数式接口，如 `matmul`、`softmax`、`gelu`、`rope` 等。这些函数并不执行实际计算，而是向 TensorRT network 中添加相应的计算节点。这种设计与 PyTorch 的 `torch.nn.functional` 类似，但底层机制完全不同。

**`models/` 目录**是模型支持的核心。每个模型子目录通常包含：
- `model.py`：用 TensorRT-LLM 的 `Module` API 定义模型结构
- `convert.py`：从 HuggingFace checkpoint 转换权重格式

## 2.3 cpp/ —— C++ 运行时与 CUDA Kernel

这是 TensorRT-LLM 性能的根基所在。所有对延迟敏感的操作都在 C++ 层实现。

```
cpp/
├── tensorrt_llm/
│   ├── runtime/               # C++ 运行时核心
│   │   ├── gptSession.cpp     # 推理会话管理
│   │   ├── gptDecoder.cpp     # 解码器逻辑
│   │   ├── bufferManager.cpp  # 显存管理
│   │   └── tllmRuntime.cpp    # TRT engine 加载与执行
│   ├── kernels/               # 自定义 CUDA kernel
│   │   ├── decoderMaskedMultiheadAttention/  # MHA kernel
│   │   ├── samplingTopKKernels.cu            # Top-K 采样
│   │   ├── samplingTopPKernels.cu            # Top-P 采样
│   │   ├── beamSearchKernels.cu              # Beam Search
│   │   └── quantization/                      # 量化 kernel
│   ├── plugins/               # TensorRT 插件
│   │   ├── gptAttentionPlugin/    # GPT Attention 插件
│   │   ├── ncclPlugin/            # NCCL 通信插件
│   │   ├── lookupPlugin/          # Embedding lookup 插件
│   │   └── smoothQuantGemmPlugin/ # SmoothQuant GEMM 插件
│   ├── batch_manager/         # 批处理调度器
│   │   ├── inferenceRequest.cpp
│   │   ├── schedulerPolicy.cpp
│   │   └── kvCacheManager.cpp     # KV Cache 分页管理
│   └── common/                # 公共工具
│       ├── memoryUtils.cu
│       └── cudaUtils.cpp
├── include/                   # 头文件
│   └── tensorrt_llm/
│       ├── runtime/
│       ├── kernels/
│       └── batch_manager/
├── tests/                     # C++ 单元测试
└── CMakeLists.txt
```

**`kernels/` 目录**是 CUDA 工程的核心。以 `decoderMaskedMultiheadAttention/` 为例，这个目录下实现了 generation phase 的 multi-head attention kernel，针对不同的 head size（64、128、256）、不同的数据类型（FP16、BF16、FP8）、不同的 KV Cache 布局做了高度特化的实现。这些 kernel 直接操作 GPU 共享内存和寄存器，手工优化了数据加载、计算和存储的每一个步骤。

```cpp
// cpp/tensorrt_llm/kernels/ 下的 kernel 设计哲学（概念示例）
// 不是逐行读代码，而是理解其架构决策

// 1. 针对 head_size 做编译期特化，避免运行时分支
template <int HEAD_SIZE, typename T, typename KVCacheT>
__global__ void masked_multihead_attention_kernel(
    const AttentionParams<T> params) {
    // 2. 每个 thread block 处理一个 head
    // 3. 利用 shared memory 做 Q*K^T 的分块计算
    // 4. 在线 softmax（streaming softmax）避免两遍扫描
    // 5. 最终的 attention_output 直接写回 global memory
}

// 6. 通过 dispatcher 在运行时选择正确的模板实例
void dispatch_mha_kernel(const AttentionParams& params) {
    switch (params.head_size) {
        case 64:  launch<64, half, ...>(); break;
        case 128: launch<128, half, ...>(); break;
        // ...
    }
}
```

**`plugins/` 目录**实现了 TensorRT 插件。TensorRT 的标准算子库无法覆盖 LLM 的所有需求（如带 KV Cache 的 attention、RoPE 位置编码等），因此 TensorRT-LLM 通过插件机制将自定义 CUDA kernel 注册到 TensorRT 编译器中。`gptAttentionPlugin/` 是最核心的插件之一，它将 attention 计算、KV Cache 更新、RoPE 编码等操作封装为一个整体插件，避免了拆分为多个小算子带来的性能损失。

**`batch_manager/` 目录**实现了 In-flight Batching 的调度逻辑。`kvCacheManager.cpp` 管理分页 KV Cache 的分配与释放，`schedulerPolicy.cpp` 决定每一步哪些请求参与计算。这是 TensorRT-LLM 在服务化场景下获得高吞吐量的关键组件。

## 2.4 examples/ —— 模型使用示例

```
examples/
├── llama/
│   ├── convert_checkpoint.py   # 权重转换
│   └── README.md
├── gpt/
├── chatglm/
├── qwen/
├── whisper/                    # 语音模型也支持
└── ...
```

每个模型目录通常提供从权重转换、engine 构建到推理执行的完整流程示例。这些示例是学习 TensorRT-LLM 使用方法的最佳起点。

## 2.5 其他重要目录

**`3rdparty/`** 以 git submodule 形式引入第三方依赖：

- **CUTLASS**：NVIDIA 的 CUDA 模板线性代数库，TensorRT-LLM 中大量 GEMM kernel 基于它实现
- **NCCL**：NVIDIA 集合通信库，用于多卡并行推理
- **FasterTransformer**：部分遗留的高性能 kernel 仍来自 FT
- **json**：nlohmann/json，C++ JSON 解析
- **googletest**：C++ 单元测试框架

**`tests/`** 包含 Python 端的测试套件，覆盖模型正确性验证、精度对比、功能测试等。`cpp/tests/` 下是 C++ 单元测试。

**`benchmarks/`** 提供了标准化的性能基准测试脚本，可以用来对比不同配置下的吞吐量和延迟。

**`scripts/`** 包含构建脚本、Docker 镜像构建脚本和各种工具脚本。`build_wheel.py` 是构建 Python wheel 包的入口。

## 2.6 核心模块关系图

用文字描述 TensorRT-LLM 各模块之间的依赖关系：

```
用户代码
  │
  ▼
tensorrt_llm.models (模型定义)
  │  调用
  ▼
tensorrt_llm.layers (通用层: attention, linear, ...)
  │  调用
  ▼
tensorrt_llm.functional (基础算子: matmul, softmax, ...)
  │  构建
  ▼
TensorRT Network (计算图)
  │  编译 (Builder)
  ▼
TensorRT Engine (.engine 二进制文件)
  │  加载
  ▼
cpp/runtime (C++ 运行时)
  │  调用                    │  调度
  ▼                         ▼
cpp/plugins (TRT 插件)   cpp/batch_manager (批调度)
  │  调用                    │  管理
  ▼                         ▼
cpp/kernels (CUDA kernel)  KV Cache / 显存管理
  │
  ▼
cuBLAS / CUTLASS / NCCL (底层库)
```

从这张关系图可以看出，TensorRT-LLM 的架构是一个清晰的**分层设计**：Python 层负责模型定义和编译流程的编排，C++ 层负责运行时的高性能执行。两层之间通过 TensorRT engine 文件作为桥梁——这也意味着编译阶段和运行阶段可以完全分离。

## 2.7 从入口到引擎：一次推理的调用链

为了建立端到端的直觉，我们简述一次完整推理请求经历的调用链：

```python
# 1. 用户发起请求
output = model_runner.generate(input_ids, max_new_tokens=128)
```

```
generate()                          # tensorrt_llm/runtime/model_runner.py
  └─► Session.run()                 # Python → C++ 绑定
       └─► GptSession::generateV2() # cpp/tensorrt_llm/runtime/gptSession.cpp
            ├─► Context Phase        # 处理完整 prompt
            │   └─► TRT Engine 执行  # attention plugin → custom kernel
            │       └─► KV Cache 填充
            └─► Generation Phase     # 逐 token 生成
                ├─► TRT Engine 执行  # masked MHA kernel
                ├─► Sampling         # top-k / top-p 采样 kernel
                └─► KV Cache 追加
```

**Context Phase**（也叫 prefill phase）处理整个输入 prompt，是一个计算密集的矩阵乘操作，主要瓶颈在 GEMM 的计算带宽上。**Generation Phase** 逐 token 生成，每一步只有一个新 token 的 query 向量，主要瓶颈在显存带宽（读取 KV Cache）上。这两个阶段的性能特征截然不同，TensorRT-LLM 为它们分别优化了不同的 kernel 实现。

理解了这条调用链，后续章节对每个模块的深入分析就有了清晰的锚点。

## 本章小结

本章对 TensorRT-LLM 代码仓库进行了全景式的梳理。项目采用 Python + C++/CUDA 双层架构：`tensorrt_llm/` 目录提供用户友好的 Python API，涵盖模型定义（`models/`）、编译构建（`builder.py`）、函数式算子（`functional.py`）等核心模块；`cpp/` 目录实现高性能运行时，包括 CUDA kernel（`kernels/`）、TensorRT 插件（`plugins/`）、批处理调度（`batch_manager/`）等关键组件。两层之间通过 TensorRT engine 文件连接，编译阶段与运行阶段清晰分离。`examples/`、`tests/`、`benchmarks/` 等目录提供了完善的示例、测试与基准支持。在后续章节中，我们将沿着本章建立的全景地图，逐一深入每个核心模块的实现细节。
