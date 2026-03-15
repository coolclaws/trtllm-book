# 第 14 章：Tensor Parallelism

> "The key insight of Megatron-LM is that certain operations in Transformers are naturally parallelizable across their hidden dimension." —— Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*

当模型参数量达到数百亿甚至数千亿时，单张 GPU 的显存已经无法容纳完整的权重矩阵。Tensor Parallelism（张量并行，简称 TP）是解决这一问题的核心技术之一。它的基本思路非常直观：把一个大矩阵切成若干小块，分配到多张 GPU 上，让每张 GPU 只负责计算其中一部分，最后再通过集合通信把结果汇总。本章将从原理出发，深入 TensorRT-LLM 的源码，逐步揭示 TP 在工业级推理框架中的完整实现。

## 14.1 Tensor Parallelism 的基本原理

Tensor Parallelism 的核心在于对权重矩阵进行切分。假设一个线性层的权重矩阵 $W$ 的形状为 $(H_{in}, H_{out})$，输入 $X$ 的形状为 $(B, S, H_{in})$。我们可以选择两种切分方式：

**列切分（Column Parallel）**：将 $W$ 沿输出维度（列方向）切分为 $N$ 份，每张 GPU 持有 $W_i$，形状为 $(H_{in}, H_{out}/N)$。每张 GPU 接收完整的输入 $X$，独立计算 $Y_i = X \cdot W_i$，得到部分输出。此时各 GPU 上的结果沿最后一个维度拼接即可还原完整输出。

**行切分（Row Parallel）**：将 $W$ 沿输入维度（行方向）切分为 $N$ 份，每张 GPU 持有 $W_i$，形状为 $(H_{in}/N, H_{out})$。此时输入 $X$ 也需要沿隐藏维度切分，每张 GPU 计算 $Y_i = X_i \cdot W_i$，得到的结果需要通过 AllReduce 操作求和才能还原完整输出。

这两种切分方式并非孤立使用，而是巧妙地配合在一起。

## 14.2 Megatron-LM 风格的 TP 策略

Megatron-LM 提出了一种经典的 TP 组合方案，其核心思想是：**ColumnLinear 之后接 RowLinear，中间不需要额外的通信**。

在一个标准的 Transformer MLP 层中，有两个连续的线性变换：

```
MLP(x) = GeLU(x · W1) · W2
```

Megatron-LM 的做法是：

1. 对 $W_1$ 进行列切分（ColumnLinear），每张 GPU 计算 $GeLU(X \cdot W_{1,i})$，得到部分激活值
2. 对 $W_2$ 进行行切分（RowLinear），每张 GPU 用自己持有的部分激活值乘以 $W_{2,i}$
3. 在 RowLinear 的输出处执行一次 AllReduce，将所有 GPU 的部分和汇总

这样一个完整的 MLP 块只需要一次 AllReduce 通信，大幅降低了通信开销。TensorRT-LLM 完全沿用了这一设计模式。

## 14.3 TRT-LLM 中的 ColumnLinear 与 RowLinear

TensorRT-LLM 在 `tensorrt_llm/layers/linear.py` 中实现了 `ColumnLinear` 和 `RowLinear` 两个核心类。让我们看看关键实现。

```python
# tensorrt_llm/layers/linear.py

class ColumnLinear(Linear):
    def __init__(self, in_features, out_features, bias=True,
                 dtype=None, tp_group=None, tp_size=1,
                 gather_output=True):
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.gather_output = gather_output
        # 每个 rank 只分配 out_features / tp_size 列
        super().__init__(in_features, out_features // tp_size,
                         bias=bias, dtype=dtype)

    def forward(self, x):
        y = matmul(x, self.weight.value)
        if self.bias is not None:
            y = y + self.bias.value
        if self.gather_output and self.tp_size > 1:
            y = allgather(y, self.tp_group)
        return y
```

`ColumnLinear` 的关键在于：它将 `out_features` 除以 `tp_size`，让每张 GPU 只存储和计算输出维度的一部分。如果 `gather_output=True`，则在输出时通过 AllGather 收集所有 GPU 的结果；如果为 `False`（比如在 MLP 中与 RowLinear 配合时），则保持切分状态直接传递给下一层。

```python
class RowLinear(Linear):
    def __init__(self, in_features, out_features, bias=True,
                 dtype=None, tp_group=None, tp_size=1):
        self.tp_size = tp_size
        self.tp_group = tp_group
        # 每个 rank 只分配 in_features / tp_size 行
        super().__init__(in_features // tp_size, out_features,
                         bias=bias, dtype=dtype)

    def forward(self, x):
        y = matmul(x, self.weight.value)
        if self.tp_size > 1:
            y = allreduce(y, self.tp_group)
        if self.bias is not None:
            y = y + self.bias.value
        return y
```

`RowLinear` 将 `in_features` 除以 `tp_size`，期望接收已切分的输入。计算完成后执行 AllReduce 求和，恢复完整的输出。注意 bias 是在 AllReduce 之后加的——因为 bias 不需要被切分，只在汇总之后加一次即可。

## 14.4 Attention 层的 TP 切分策略

在 Multi-Head Attention 中，TP 的切分天然适配 multi-head 结构。具体实现位于 `tensorrt_llm/layers/attention.py`：

- **QKV 投影**：使用 `ColumnLinear`，将 Q、K、V 的输出按 head 数量切分。如果模型有 32 个 head，`tp_size=4`，则每张 GPU 处理 8 个 head
- **Output 投影**：使用 `RowLinear`，接收各 GPU 上部分 head 的 attention 输出，执行矩阵乘法后 AllReduce 汇总

```python
# tensorrt_llm/layers/attention.py (简化)

class Attention(Module):
    def __init__(self, hidden_size, num_attention_heads, tp_group, tp_size):
        self.qkv = ColumnLinear(
            hidden_size,
            3 * hidden_size,  # Q, K, V 合并
            tp_group=tp_group,
            tp_size=tp_size,
            gather_output=False  # 不收集，直接传给 attention 计算
        )
        self.dense = RowLinear(
            hidden_size,
            hidden_size,
            tp_group=tp_group,
            tp_size=tp_size
        )
```

对于使用 Grouped Query Attention（GQA）的模型（如 LLaMA-2 70B），K 和 V 的 head 数量少于 Q。TRT-LLM 会根据 `num_kv_heads` 和 `tp_size` 的关系决定是否对 KV 进行切分，还是在每个 rank 上复制完整的 KV heads。

## 14.5 AllReduce 通信实现

AllReduce 是 TP 中最关键的通信原语。TensorRT-LLM 提供了基于 NCCL 的自定义 plugin 实现，源码位于 `cpp/tensorrt_llm/plugins/ncclPlugin/`。

```cpp
// cpp/tensorrt_llm/plugins/ncclPlugin/allreducePlugin.cpp (简化)

int AllreducePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept
{
    // 获取当前 rank 所在的 NCCL communicator
    auto comm = getComm(mGroup);
    // 执行 AllReduce
    NCCLCHECK(ncclAllReduce(
        inputs[0], outputs[0], size,
        ncclFloat16,  // 数据类型
        ncclSum,      // 规约操作
        comm, stream));
    return 0;
}
```

除了标准的 NCCL AllReduce，TRT-LLM 还实现了**自定义 AllReduce kernel**（`cpp/tensorrt_llm/kernels/customAllReduceKernels/`），利用 NVLink 的高带宽特性进行优化。当检测到 GPU 之间通过 NVLink 连接时，自定义 kernel 可以跳过 NCCL 的协议层，直接通过 GPU 间的点对点内存访问实现更低延迟的归约操作。

## 14.6 NVLink vs PCIe：带宽对 TP 性能的影响

TP 的性能瓶颈几乎完全取决于 GPU 间的互联带宽。以下是不同互联方式的带宽对比：

| 互联方式 | 单向带宽 | 典型场景 |
|---------|---------|---------|
| PCIe Gen4 x16 | ~32 GB/s | 消费级 / 早期服务器 |
| PCIe Gen5 x16 | ~64 GB/s | 新一代服务器 |
| NVLink 3.0 (A100) | ~600 GB/s | 数据中心 |
| NVLink 4.0 (H100) | ~900 GB/s | 高端数据中心 |

在实际推理中，一次 AllReduce 需要传输的数据量约为 $2 \times B \times S \times H \times \text{sizeof(dtype)}$（ring AllReduce 的通信量）。对于一个 hidden_size=8192、batch_size=32、seq_len=1 的 decode 步骤，传输量约为 $2 \times 32 \times 1 \times 8192 \times 2 = 1\text{MB}$。在 NVLink 上这几乎可以忽略不计，但在 PCIe 上延迟会显著增加。

**经验法则**：在 PCIe 互联环境下，`tp_size` 一般不建议超过 2；在 NVLink 互联环境下，`tp_size` 可以设置为 4 或 8。

## 14.7 多 GPU Engine 构建与配置

使用 TP 时，每个 rank 需要独立构建一个 TensorRT engine。TRT-LLM 通过 `Mapping` 类（`tensorrt_llm/mapping.py`）来管理并行拓扑：

```python
from tensorrt_llm.mapping import Mapping

# 4 路 Tensor Parallelism
mapping = Mapping(
    world_size=4,
    rank=0,         # 当前进程的 rank
    tp_size=4,
    pp_size=1
)
```

构建 engine 的典型流程如下：

```python
# 为每个 rank 构建独立的 engine
for rank in range(tp_size):
    mapping = Mapping(world_size=tp_size, rank=rank, tp_size=tp_size)
    model = LLaMAForCausalLM.from_hugging_face(
        model_dir, mapping=mapping, dtype='float16'
    )
    engine = build(model, build_config)
    engine.save(f'engine_rank{rank}')
```

在实际部署中，可以通过 `mpirun` 启动多个进程来并行构建：

```bash
mpirun -n 4 python build.py --tp_size 4
```

每个进程通过 MPI 获取自己的 rank，加载对应的权重切片，独立完成 engine 构建。运行时同样以多进程方式启动，各 rank 加载自己的 engine，通过 NCCL 完成推理过程中的 AllReduce 通信。

## 14.8 tp_size 选择的实践建议

选择合适的 `tp_size` 需要综合考虑以下因素：

1. **显存约束**：模型参数在 FP16 下的大小除以单卡可用显存，得到最小 `tp_size`
2. **head 数量整除**：`tp_size` 必须能整除 `num_attention_heads`（以及 GQA 中的 `num_kv_heads`）
3. **互联拓扑**：优先在 NVLink 域内进行 TP，避免跨 NVLink 域或跨节点 TP
4. **延迟 vs 吞吐**：TP 能降低延迟（计算并行化），但通信开销会降低吞吐。在吞吐优先场景下可以用较小的 `tp_size` 配合较大的 batch

一个典型的配置示例：LLaMA-2 70B（FP16 约 140GB）在 8 卡 A100 80GB 上可以选择 `tp_size=4` 或 `tp_size=8`。前者每卡约 35GB 权重，留有足够的 KV cache 空间；后者每卡约 17.5GB，可支持更大的 batch size。

## 本章小结

本章深入分析了 TensorRT-LLM 中 Tensor Parallelism 的完整实现。我们从矩阵切分的基本原理出发，理解了 Megatron-LM 风格的 ColumnLinear + RowLinear 配合策略如何将每个 Transformer 块的通信次数降到最低。在源码层面，`tensorrt_llm/layers/linear.py` 中的 `ColumnLinear` 和 `RowLinear` 清晰地体现了列切分与行切分的实现逻辑，而 `cpp/tensorrt_llm/plugins/ncclPlugin/` 下的 NCCL plugin 和自定义 AllReduce kernel 则展示了通信层的工程优化。我们还讨论了 NVLink 与 PCIe 带宽差异对 TP 性能的决定性影响，以及在实际部署中如何根据硬件拓扑和模型规模选择合适的 `tp_size`。理解了 TP 之后，下一章我们将探讨另一种并行策略——Pipeline Parallelism，它从完全不同的维度切分模型，与 TP 组合使用可以进一步扩展推理规模。
