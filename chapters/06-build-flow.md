# 第 6 章：trtllm-build 编译流程

> "编译是将模型从训练世界带入推理世界的桥梁。理解编译流程，就是理解 TensorRT-LLM 的核心生命周期。"

在前面的章节中，我们已经了解了 TensorRT-LLM 的模型定义与权重加载机制。本章将深入分析整个编译流程——从用户敲下 `trtllm-build` 命令开始，到最终生成可部署的 `.engine` 文件结束，这一完整链路是如何运作的。

## 6.1 CLI 入口：trtllm-build

`trtllm-build` 是 TensorRT-LLM 提供的命令行编译工具，其入口位于 `tensorrt_llm/commands/build.py`。当用户在终端执行 `trtllm-build` 时，Python 的 entry_points 机制会将调用路由到该模块的 `main()` 函数。

```python
# tensorrt_llm/commands/build.py
def main():
    args = parse_arguments()
    build_config = BuildConfig.from_dict(args)
    build_model(build_config)
```

入口函数的职责非常清晰：解析命令行参数、构建配置对象、执行编译。这种分层设计使得 `trtllm-build` 既可以作为 CLI 工具使用，也可以在 Python 脚本中以编程方式调用。

典型的命令行调用如下：

```python
trtllm-build \
    --checkpoint_dir ./llama-7b-checkpoint \
    --output_dir ./llama-7b-engine \
    --max_batch_size 8 \
    --max_input_len 1024 \
    --max_seq_len 2048 \
    --gemm_plugin float16
```

## 6.2 BuildConfig 配置类

`BuildConfig` 是编译流程的核心配置载体，定义在 `tensorrt_llm/builder.py` 中。它封装了所有影响 engine 生成的参数，主要包括以下几个维度：

**形状参数**（Shape Parameters）决定了 engine 能处理的输入规模上限：

- `max_batch_size`：最大批次大小，决定了 engine 能同时处理多少个请求。设得过大会占用更多显存，设得过小则无法充分利用 GPU 并行能力。
- `max_input_len`：最大输入长度，即单个请求的 prompt token 上限。
- `max_seq_len`：最大序列长度，包含 prompt 和生成的 token 总和。这个参数直接决定了 KV Cache 的显存分配上限。

**精度参数** 控制计算精度与性能的权衡。用户可以通过 `--gemm_plugin float16` 或 `--gemm_plugin bfloat16` 指定矩阵乘法的精度。BuildConfig 会将这些选项传递给底层的 PluginConfig。

**插件参数** 通过 `plugin_config` 子配置指定哪些算子使用 TensorRT plugin 加速，哪些使用原生 TRT 算子实现。这部分内容将在第 7 章详细展开。

```python
# tensorrt_llm/builder.py 简化示意
class BuildConfig:
    def __init__(self):
        self.max_batch_size = 256
        self.max_input_len = 1024
        self.max_seq_len = 2048
        self.max_beam_width = 1
        self.max_num_tokens = None
        self.plugin_config = PluginConfig()
        self.builder_opt = None  # TensorRT builder optimization level
```

值得注意的是 `max_num_tokens` 参数。在 inflight batching 场景下，不同请求的长度各异，`max_num_tokens` 定义了一个 batch 中所有请求的 token 总数上限，这比简单的 `max_batch_size × max_input_len` 更加灵活和节省显存。

## 6.3 编译三阶段

整个 build 流程可以划分为三个清晰的阶段：**加载 checkpoint**、**构建 TRT Network**、**编译为 engine**。

### 阶段一：加载 Checkpoint

编译的第一步是从磁盘读取模型的 checkpoint。TensorRT-LLM 定义了统一的 checkpoint 格式，位于 `--checkpoint_dir` 指定的目录下，通常包含：

- `config.json`：模型架构配置（层数、隐藏维度、注意力头数等）
- `rank0.safetensors`（或多个 rank 文件）：模型权重

```python
# 加载 checkpoint 的核心逻辑
def build_model(build_config):
    # 1. 从 config.json 读取模型架构
    model_config = ModelConfig.from_json_file(
        os.path.join(build_config.checkpoint_dir, 'config.json')
    )
    # 2. 实例化对应的模型类
    model_cls = MODEL_MAP[model_config.architecture]
    # 3. 加载权重
    model = model_cls.from_checkpoint(build_config.checkpoint_dir)
```

`MODEL_MAP` 是一个字典，将架构名称（如 `"LlamaForCausalLM"`）映射到对应的 Python 模型类。这种注册机制使得添加新模型非常方便。

### 阶段二：构建 TRT Network

这是编译流程中最核心的阶段。TensorRT-LLM 的模型类（如 `LlamaForCausalLM`）本质上是 TensorRT Network 的构建器——调用模型的 `forward()` 方法并不会执行实际计算，而是在 TensorRT 的 `INetworkDefinition` 上注册算子节点。

```python
# tensorrt_llm/builder.py 中的 network 构建
def build_engine(model, build_config):
    builder = trt.Builder(logger)
    network = builder.create_network()

    # 定义输入 tensor 的动态形状
    with net_guard(network):
        inputs = model.prepare_inputs(
            max_batch_size=build_config.max_batch_size,
            max_input_len=build_config.max_input_len,
            max_seq_len=build_config.max_seq_len,
        )
        # 执行 forward 构建计算图
        model(**inputs)
```

`prepare_inputs()` 方法会创建具有动态形状的输入 tensor（使用 TensorRT 的 optimization profile），这样同一个 engine 就可以在运行时处理不同大小的输入。

### 阶段三：编译为 Engine

TensorRT Network 构建完成后，TensorRT 的 builder 会对计算图进行一系列优化（Layer Fusion、Kernel Selection 等，详见第 9 章），然后编译生成序列化的 engine 文件。

```python
    # 配置 builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    # 编译
    engine = builder.build_serialized_network(network, config)

    # 写入文件
    with open(output_path, 'wb') as f:
        f.write(engine)
```

编译过程中，TensorRT 会对每一层进行 kernel profiling——即实际运行多种 CUDA kernel 实现，选择在当前 GPU 上最快的那一个。这个过程耗时较长，但确保了 engine 在目标硬件上的最优性能。

## 6.4 PluginConfig 选择策略

`PluginConfig` 定义在 `tensorrt_llm/plugin/plugin.py` 中，控制哪些层使用自定义 plugin 实现。关键的 plugin 选项包括：

- `gpt_attention_plugin`：控制注意力层的实现方式。设为 `"float16"` 时使用自定义的 GPTAttention plugin（融合了 QKV 投影、softmax、attention 计算），性能显著优于原生 TRT 算子组合。
- `gemm_plugin`：矩阵乘法 plugin，针对 LLM 的特定形状进行了优化。
- `nccl_plugin`：多 GPU 通信 plugin，封装了 NCCL 集合通信操作。

一般建议在生产环境中启用 `gpt_attention_plugin`，因为它包含了 Flash Attention、Paged KV Cache 等关键优化。

## 6.5 多 GPU Engine 构建

当模型规模超过单卡显存时，需要进行张量并行（Tensor Parallelism）或流水线并行（Pipeline Parallelism）。在多 GPU 场景下，每个 rank 需要独立的 engine 文件。

```python
# 多 GPU 构建通常使用 mpirun 启动
mpirun -n 4 trtllm-build \
    --checkpoint_dir ./llama-70b-checkpoint \
    --output_dir ./llama-70b-engine \
    --max_batch_size 8 \
    --tp_size 4
```

每个进程根据自己的 rank 编号加载对应的权重分片（如 `rank0.safetensors`、`rank1.safetensors` 等），独立构建 engine 并输出为 `rank0.engine`、`rank1.engine` 等文件。

输出目录的结构如下：

```
llama-70b-engine/
├── config.json           # 统一的配置文件
├── rank0.engine          # GPU 0 的 engine
├── rank1.engine          # GPU 1 的 engine
├── rank2.engine          # GPU 2 的 engine
└── rank3.engine          # GPU 3 的 engine
```

## 6.6 Engine 输出格式

编译生成的 engine 目录包含两类核心文件：

1. **`.engine` 文件**：TensorRT 序列化的推理引擎二进制文件，包含优化后的计算图和选定的 CUDA kernel。该文件与 GPU 架构绑定——在 A100 上编译的 engine 无法在 H100 上运行。

2. **`config.json`**：元数据配置文件，记录了编译时的所有参数，包括模型架构、精度设置、最大形状参数等。Runtime 加载 engine 时需要读取此文件来正确初始化推理环境。

## 6.7 内存优化：Weight Streaming

对于超大模型，即使在编译阶段也可能面临显存不足的问题。TensorRT-LLM 支持 **Weight Streaming** 技术——将部分权重保留在 CPU 内存中，仅在计算需要时流式加载到 GPU。

在 build 时通过 `--weight_streaming` 参数启用此功能：

```python
trtllm-build \
    --checkpoint_dir ./mixtral-8x7b-checkpoint \
    --output_dir ./mixtral-8x7b-engine \
    --weight_streaming
```

启用 weight streaming 后，engine 文件中会包含额外的元数据指示哪些权重支持流式加载。Runtime 阶段可以根据实际可用显存动态决定缓存策略。

## 本章小结

本章完整剖析了 `trtllm-build` 的编译流程。从 CLI 入口的参数解析，到 `BuildConfig` 的配置组织，再到编译三阶段（加载 checkpoint → 构建 TRT Network → 编译为 engine）的详细执行逻辑，我们看到了 TensorRT-LLM 如何将一个 Python 定义的模型转化为高度优化的推理引擎。理解编译流程是进行性能调优的基础——只有知道每个参数如何影响最终的 engine，才能在精度、性能和显存之间找到最佳平衡点。下一章我们将深入 TensorRT 的插件体系，了解那些为 LLM 推理量身定制的 CUDA kernel 是如何被集成到编译流程中的。
