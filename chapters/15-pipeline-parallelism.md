# 第 15 章：Pipeline Parallelism

> "Pipeline parallelism partitions the layers of a model across devices, and uses micro-batching to hide the pipeline bubble." —— Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*

如果说 Tensor Parallelism 是在"宽度"方向切分模型——将每一层的权重矩阵拆开分给多张 GPU，那么 Pipeline Parallelism（流水线并行，简称 PP）则是在"深度"方向切分——将模型的不同层分配到不同的 GPU 上。PP 的优势在于通信量远小于 TP（只需在相邻 stage 之间传输激活值），但它引入了一个独特的挑战：流水线气泡（pipeline bubble）。本章将从 PP 的基本原理出发，深入 TensorRT-LLM 的源码实现，分析层到 rank 的映射机制、Send/Recv 通信算子，以及 PP 与 TP 的 2D 并行组合策略。

## 15.1 Pipeline Parallelism 的基本原理

一个 Transformer 模型通常由若干层堆叠而成，例如 LLaMA-2 70B 有 80 层 Transformer block。PP 将这些层均匀划分为若干 stage，每个 stage 分配到一张（或一组）GPU 上：

```
Stage 0 (GPU 0): Embedding + Layer 0~19
Stage 1 (GPU 1): Layer 20~39
Stage 2 (GPU 2): Layer 40~59
Stage 3 (GPU 3): Layer 60~79 + LM Head
```

推理时，数据从 Stage 0 开始，逐 stage 向后传递。每两个 stage 之间需要通过 Send/Recv 操作传递中间激活值（hidden states）。与 TP 的 AllReduce 不同，PP 的通信是**点对点**的，通信量为 $B \times S \times H \times \text{sizeof(dtype)}$，通常远小于 TP 的全对全通信。

但 PP 有一个固有问题：当 Stage 0 在处理数据时，Stage 1~3 处于空闲状态；当 Stage 3 在工作时，Stage 0~2 又在等待。这种空闲时间称为**流水线气泡（pipeline bubble）**。

## 15.2 Micro-batch 流水线调度

降低 bubble 比例的经典方法是引入 micro-batch。将一个大 batch 拆分为 $M$ 个 micro-batch，让多个 micro-batch 在流水线中交错执行：

```
时间 →
GPU 0: [mb0] [mb1] [mb2] [mb3]  idle
GPU 1:  idle [mb0] [mb1] [mb2] [mb3]  idle
GPU 2:  idle  idle [mb0] [mb1] [mb2] [mb3]  idle
GPU 3:  idle  idle  idle [mb0] [mb1] [mb2] [mb3]
```

bubble 比例的计算公式为：

$$\text{bubble ratio} = \frac{P - 1}{M + P - 1}$$

其中 $P$ 是 pipeline stage 数量，$M$ 是 micro-batch 数量。当 $M \gg P$ 时，bubble 比例趋近于 0。在推理场景中，由于 decode 阶段每步只产生一个 token，micro-batch 的概念更多地体现在 context phase（prefill 阶段）中对长序列的分块处理。

对于推理任务而言，PP 的 bubble 主要出现在 prefill 阶段的开始和结束。在 decode 阶段，由于每个 step 的计算量很小，流水线的启动开销会成为更突出的问题。因此在实际部署中，纯推理场景下 PP 的使用需要更加谨慎。

## 15.3 TRT-LLM 中的 Mapping 类

TensorRT-LLM 使用 `Mapping` 类来管理所有并行维度的拓扑映射，源码位于 `tensorrt_llm/mapping.py`：

```python
# tensorrt_llm/mapping.py

class Mapping:
    def __init__(self, world_size=1, rank=0, tp_size=1, pp_size=1, cp_size=1):
        self.world_size = world_size
        self.rank = rank
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size

        # 计算当前 rank 所属的 TP group 和 PP group
        self.tp_group = self._get_tp_group()
        self.pp_group = self._get_pp_group()

    @property
    def tp_rank(self):
        """当前进程在 TP group 中的 rank"""
        return self.rank % self.tp_size

    @property
    def pp_rank(self):
        """当前进程在 PP group 中的 rank (即 stage 编号)"""
        return self.rank // self.tp_size

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1
```

`Mapping` 类的核心作用是将全局 rank 映射到具体的并行角色。在 2D 并行（TP + PP）的场景下，rank 的编排方式遵循"TP 优先"原则：相邻的 rank 属于同一个 TP group，跨 TP group 的 rank 属于不同的 PP stage。

例如，`world_size=8, tp_size=4, pp_size=2` 的配置：

```
PP Stage 0: Rank 0, 1, 2, 3  (TP group)
PP Stage 1: Rank 4, 5, 6, 7  (TP group)
```

这种编排保证了 TP 通信发生在物理相邻的 GPU 之间（通常连接在同一 NVLink 域内），而 PP 通信发生在不同 NVLink 域甚至不同节点之间。

## 15.4 层到 Rank 的映射

模型在构建时需要决定哪些层放在哪个 stage。TRT-LLM 中，各模型类通过 `Mapping` 对象来确定当前 rank 需要构建的层范围。以 GPT 模型为例（`tensorrt_llm/models/gpt/model.py`）：

```python
# tensorrt_llm/models/gpt/model.py (简化)

class GPTModel(Module):
    def __init__(self, config):
        self.mapping = config.mapping
        num_layers = config.num_hidden_layers

        # 计算当前 PP stage 负责的层范围
        layers_per_stage = num_layers // self.mapping.pp_size
        first_layer = self.mapping.pp_rank * layers_per_stage
        last_layer = first_layer + layers_per_stage

        # 只构建当前 stage 的层
        self.layers = ModuleList([
            GPTDecoderLayer(config, layer_idx=i)
            for i in range(first_layer, last_layer)
        ])

        # Embedding 只在第一个 stage
        if self.mapping.is_first_pp_rank():
            self.embedding = Embedding(config)

        # LM Head 只在最后一个 stage
        if self.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(config.hidden_size, config.vocab_size)
```

这段代码清晰地展示了 PP 的切分逻辑：每个 stage 只实例化属于自己的层，Embedding 仅在第一个 stage 创建，LM Head 仅在最后一个 stage 创建。这意味着每个 rank 构建的 TensorRT engine 只包含模型的一部分，显存占用相应降低。

## 15.5 Send/Recv 通信算子

PP 的 stage 之间通过 Send 和 Recv 操作传递激活值。TRT-LLM 将这些操作封装为 TensorRT plugin，定义在 `cpp/tensorrt_llm/plugins/ncclPlugin/` 中：

```cpp
// cpp/tensorrt_llm/plugins/ncclPlugin/sendPlugin.cpp (简化)

int SendPlugin::enqueue(...) {
    auto comm = getComm(mGroup);
    int peer = mTgtRank;  // 目标 rank
    NCCLCHECK(ncclSend(
        inputs[0], size, ncclFloat16,
        peer, comm, stream));
    return 0;
}
```

```cpp
// cpp/tensorrt_llm/plugins/ncclPlugin/recvPlugin.cpp (简化)

int RecvPlugin::enqueue(...) {
    auto comm = getComm(mGroup);
    int peer = mSrcRank;  // 源 rank
    NCCLCHECK(ncclRecv(
        outputs[0], size, ncclFloat16,
        peer, comm, stream));
    return 0;
}
```

在模型的 forward 函数中，这些通信操作被插入在 stage 的边界处：

```python
def forward(self, hidden_states, ...):
    # 如果不是第一个 stage，从上一个 stage 接收 hidden states
    if not self.mapping.is_first_pp_rank():
        hidden_states = recv(hidden_states, self.mapping.prev_pp_rank())

    # 执行当前 stage 的层计算
    for layer in self.layers:
        hidden_states = layer(hidden_states, ...)

    # 如果不是最后一个 stage，将 hidden states 发送给下一个 stage
    if not self.mapping.is_last_pp_rank():
        send(hidden_states, self.mapping.next_pp_rank())

    return hidden_states
```

Send 和 Recv 操作使用 NCCL 的点对点通信 API，底层会自动选择最优的传输路径（NVLink、PCIe、或跨节点的 RDMA）。

## 15.6 PP 与 TP 的 2D 并行组合

在实际部署中，PP 和 TP 常常组合使用，形成 2D 并行策略。这种组合的核心原则是：

- **TP 放在 NVLink 域内**：TP 的 AllReduce 通信频繁且对延迟敏感，需要高带宽低延迟的互联
- **PP 放在 NVLink 域之间或节点之间**：PP 的 Send/Recv 通信量相对较小，对延迟的容忍度更高

以 8 卡 H100 服务器部署 LLaMA-2 70B 为例：

```bash
# 方案 A：纯 TP
# tp_size=8, pp_size=1
# 每层参数分到 8 卡，AllReduce 通信量大但走 NVLink

# 方案 B：2D 并行
# tp_size=4, pp_size=2
# 前 40 层在 GPU 0-3 (TP group), 后 40 层在 GPU 4-7 (TP group)
# TP 通信走 NVLink，PP 通信走 NVLink 或 PCIe

mpirun -n 8 python run.py --tp_size 4 --pp_size 2
```

方案 B 的优势在于：每个 TP group 只有 4 张卡，AllReduce 的通信代价更低；同时 PP 的通信量远小于 TP，即使走较慢的链路也不会成为瓶颈。

## 15.7 何时选择 PP vs TP

PP 和 TP 各有适用场景，选择时需要考虑以下因素：

| 维度 | Tensor Parallelism | Pipeline Parallelism |
|-----|-------------------|---------------------|
| 通信模式 | AllReduce（全对全） | Send/Recv（点对点） |
| 通信频率 | 每层 1~2 次 | 每 stage 1 次 |
| 通信量 | 较大（与 hidden_size 成正比） | 较小（仅传激活值） |
| 延迟影响 | 降低计算延迟 | 增加流水线延迟 |
| 显存节省 | 切分权重和激活 | 切分权重，激活不共享 |
| 适合场景 | NVLink 高带宽互联 | 跨节点或低带宽互联 |

**推荐策略**：

1. 如果所有 GPU 通过 NVLink 全互联，优先使用纯 TP
2. 如果 GPU 跨 NVLink 域或跨节点，使用 TP（域内） + PP（域间）
3. 纯 PP 在推理场景下通常不是最优选择，因为 bubble 开销难以消除
4. `pp_size` 不宜过大，一般 2~4 即可；更大的 PP 度会导致 bubble 比例显著上升

## 15.8 GPT 模型流水线切分示例

下面用一个完整的示例展示如何为 GPT 风格模型配置 2D 并行，并在 `tensorrt_llm/builder.py` 中完成构建：

```python
from tensorrt_llm import Mapping, BuildConfig
from tensorrt_llm.models import LLaMAForCausalLM

# 16 卡：4 路 TP x 4 路 PP
world_size = 16
tp_size = 4
pp_size = 4

for rank in range(world_size):
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        tp_size=tp_size,
        pp_size=pp_size
    )
    # pp_rank=0 的 4 个 rank 构建前 20 层
    # pp_rank=1 的 4 个 rank 构建 20~39 层
    # pp_rank=2 的 4 个 rank 构建 40~59 层
    # pp_rank=3 的 4 个 rank 构建 60~79 层
    model = LLaMAForCausalLM.from_hugging_face(
        model_dir, mapping=mapping
    )
    engine = build(model, BuildConfig(max_batch_size=64))
    engine.save(f'engines/rank{rank}/engine.trt')
```

运行时，通过 `mpirun -n 16` 启动所有 rank，框架会自动根据 `Mapping` 信息建立 NCCL communicator，在 TP group 内执行 AllReduce，在 PP stage 之间执行 Send/Recv。

## 本章小结

本章详细分析了 TensorRT-LLM 中 Pipeline Parallelism 的原理与实现。PP 通过将模型按层切分到多个 stage，以点对点通信替代全对全通信，适合在低带宽互联环境下使用。我们深入了 `tensorrt_llm/mapping.py` 中 `Mapping` 类的设计，理解了 TP rank 和 PP rank 的编排逻辑；通过模型源码看到了层到 stage 的映射是如何在模型构造函数中实现的；还分析了 Send/Recv plugin 的底层通信机制。PP 的核心挑战在于 pipeline bubble，micro-batch 技术可以有效缓解这一问题，但在推理的 decode 阶段效果有限。在实际部署中，PP 通常与 TP 组合形成 2D 并行策略，充分利用不同层次互联的带宽特性。下一章我们将进一步扩展视野，探讨跨节点的多机部署方案，解决单台服务器仍然无法容纳超大模型的问题。
