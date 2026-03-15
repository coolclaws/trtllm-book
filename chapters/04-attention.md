# 第 4 章：Attention 实现

> "Attention is all you need, but implementing it efficiently is another story." —— 匿名工程师

Attention 机制是 Transformer 架构的核心，也是大语言模型推理中计算量最大、优化空间最丰富的组件。TensorRT-LLM 在 Attention 的实现上投入了大量工程努力——从 Python 层的灵活配置到 C++ 层的高性能 plugin，再到 KV Cache 的精细管理，形成了一套多层协作的体系。本章将从 `tensorrt_llm/layers/attention.py` 出发，逐层拆解这套实现。

## 4.1 Attention 类的整体结构

`tensorrt_llm/layers/attention.py` 中的 `Attention` 类是所有内置模型共用的注意力层实现。它的构造函数参数非常丰富，因为需要同时支持多种注意力变体和优化选项：

```python
# tensorrt_llm/layers/attention.py（核心参数，简化）
class Attention(Module):
    def __init__(self,
                 local_layer_idx: int,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_kv_heads: Optional[int] = None,    # 支持 GQA/MQA
                 max_position_embeddings: int = 1024,
                 attention_head_size: Optional[int] = None,
                 q_scaling: float = 1.0,
                 position_embedding_type: PositionEmbeddingType = PositionEmbeddingType.learned_absolute,
                 rotary_embedding_base: float = 10000.0,
                 rotary_embedding_scaling: Optional[dict] = None,
                 tp_group: Optional[list] = None,
                 tp_size: int = 1,
                 dtype: Optional[str] = None,
                 dense_bias: bool = True,
                 ...):
```

从参数列表就能看出这个类需要处理的复杂度：注意力头的配置、位置编码的选择、张量并行的设置、量化相关的选项等等。

## 4.2 MHA、MQA 与 GQA

三种注意力模式的区别在于 Query head 和 Key/Value head 的数量关系：

- **MHA（Multi-Head Attention）**：`num_kv_heads == num_attention_heads`，每个 Query head 对应独立的 KV head。GPT-2、早期 LLaMA 使用此模式。
- **MQA（Multi-Query Attention）**：`num_kv_heads == 1`，所有 Query head 共享同一组 KV head。推理时大幅减少 KV Cache 的显存占用。
- **GQA（Grouped-Query Attention）**：`1 < num_kv_heads < num_attention_heads`，每组 Query head 共享一组 KV head。LLaMA-2 70B、Mistral 7B 采用此模式，在质量和效率间取得平衡。

在 TensorRT-LLM 中，这三种模式通过 `num_attention_heads` 和 `num_kv_heads` 两个参数统一处理：

```python
# 在 Attention.__init__ 中
self.num_attention_heads = num_attention_heads
self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads

# QKV 投影的输出维度根据 head 数计算
# Q 的输出维度：num_attention_heads * head_size
# K, V 的输出维度：num_kv_heads * head_size
self.qkv = ColumnLinear(
    hidden_size,
    (num_attention_heads + 2 * self.num_kv_heads) * self.attention_head_size,
    bias=qkv_bias,
    tp_group=tp_group,
    tp_size=tp_size,
)
```

注意 QKV 被合并为一个 `ColumnLinear`，这是 LLM 推理中的常见优化——将三次矩阵乘法合并为一次，减少 kernel launch 开销并提升 GPU 利用率。输出张量在后续会被 split 为 Q、K、V 三部分。

## 4.3 GPTAttention Plugin

TensorRT-LLM 的 Attention 计算核心并非通过标准 TensorRT layer 组合实现，而是通过一个高度定制的 C++ plugin —— `GPTAttention`。这个 plugin 的源码位于：

```
cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.cpp
cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h
cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/
```

GPTAttention plugin 将以下操作融合在一个 kernel 中：

1. Q、K、V 的 reshape 和 transpose
2. RoPE 或 ALiBi 位置编码的应用
3. KV Cache 的读写管理
4. Attention score 的计算（支持 Flash Attention 和 standard attention 两条路径）
5. Softmax
6. 与 Value 的加权求和

这种深度融合避免了中间结果在 GPU 显存之间的反复读写，是性能优化的关键。在 Python 端，plugin 的调用被封装得很简洁：

```python
# tensorrt_llm/layers/attention.py 中调用 GPTAttention plugin（简化）
def forward(self, hidden_states, attention_mask=None, past_key_value=None,
            use_cache=False, kv_cache_params=None, attention_params=None):
    # 1. QKV 投影
    qkv = self.qkv(hidden_states)

    # 2. 调用 GPTAttention plugin 完成实际的 attention 计算
    context, past_key_value = gpt_attention(
        qkv=qkv,
        past_key_value=past_key_value,
        sequence_length=attention_params.sequence_length,
        host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
        context_lengths=attention_params.context_lengths,
        cache_indirection=kv_cache_params.cache_indirection,
        num_heads=self.num_attention_heads,
        num_kv_heads=self.num_kv_heads,
        head_size=self.attention_head_size,
        q_scaling=self.q_scaling,
        rotary_embedding_dim=self.rotary_embedding_dim,
        position_embedding_type=self.position_embedding_type,
        kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets,
        host_kv_cache_block_offsets=kv_cache_params.host_kv_cache_block_offsets,
        max_context_length=attention_params.max_context_length,
    )

    # 3. 输出投影
    output = self.dense(context)
    return output, past_key_value
```

`gpt_attention` 函数定义在 `tensorrt_llm/functional.py` 中，它负责构造 plugin 实例并将其添加到 TensorRT network。

## 4.4 Flash Attention 集成

Flash Attention 通过 tiling 技术避免了在 HBM 中 materialize 完整的 attention matrix，将显存使用从 O(n^2) 降低到 O(n)，同时利用 GPU SRAM 的高带宽加速计算。

在 TensorRT-LLM 中，Flash Attention 并非作为独立模块存在，而是集成在 GPTAttention plugin 内部。plugin 会根据以下条件选择执行路径：

```cpp
// cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.cpp（逻辑简化）
if (is_context_phase) {
    // Context phase（处理 prompt）：使用 Flash Attention
    // 调用优化的 fused multi-head attention kernel
    mFMHARunner->run(params);
} else {
    // Generation phase（逐 token 生成）：使用 Masked MHA
    // 此时 query 长度为 1，问题退化为 GeMV，使用专门的 kernel
    mDecoderXQARunner->run(params);
}
```

Context phase（首次处理整个 prompt）和 generation phase（逐 token 生成）的计算特征截然不同，因此使用不同的 kernel 是合理的。Context phase 的序列较长，Flash Attention 的 tiling 优势明显；generation phase 每次只有一个 token 的 query，问题退化为矩阵向量乘，需要不同的优化策略。

## 4.5 KV Cache 管理

KV Cache 是自回归推理的核心优化：每个生成步骤只需计算当前 token 的 K 和 V，并将其追加到 cache 中，避免重复计算历史 token 的 KV。TensorRT-LLM 支持两种 KV Cache 管理方式：

**Continuous KV Cache**：为每个请求预分配一块连续的显存空间，大小为 `max_seq_len * num_kv_heads * head_size`。优点是实现简单，kernel 访问模式规整；缺点是当不同请求的序列长度差异较大时会造成显存浪费。

**Paged KV Cache**：借鉴操作系统虚拟内存的分页机制，将 KV Cache 切分为固定大小的 block（通常每个 block 包含 64 或 128 个 token 的 KV），通过 block table 进行间接寻址。

```python
# Paged KV Cache 的关键参数
# kv_cache_block_offsets: 形状为 [batch_size, max_blocks_per_seq] 的索引表
# 每个元素指向 KV Cache pool 中的一个 block
# 这使得不同请求可以共享物理 block，甚至支持 copy-on-write 的 prefix caching
```

Paged KV Cache 与 vLLM 提出的 PagedAttention 思想一致。在 TensorRT-LLM 的 runtime 中，`cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` 负责 block 的分配和回收，实现了一个完整的显存管理器。

## 4.6 RoPE 位置编码

Rotary Position Embedding（RoPE）是目前最主流的位置编码方式，LLaMA、Mistral、Qwen 等模型均采用。其核心思想是通过旋转变换将位置信息注入 Q 和 K 向量。

在 TensorRT-LLM 中，RoPE 的计算被融合在 GPTAttention plugin 内部，对应的 CUDA kernel 位于：

```
cpp/tensorrt_llm/kernels/ropeKernels/
```

配置 RoPE 时需要注意几个参数：

```python
# 标准 RoPE
position_embedding_type = PositionEmbeddingType.rope_gpt_neox  # LLaMA 风格
rotary_embedding_base = 10000.0   # theta 基数
rotary_embedding_dim = head_size  # 旋转维度，通常等于 head_size

# 扩展上下文长度的 RoPE 变体
rotary_embedding_scaling = {
    "type": "dynamic",       # 支持 linear / dynamic / yarn 等
    "factor": 2.0,           # 缩放因子
}
```

`rope_gpt_neox` 和 `rope_gpt_j` 是两种不同的 RoPE 实现方式，区别在于维度的交错模式。GPT-NeoX 风格将前半部分和后半部分配对旋转，GPT-J 风格将相邻维度配对。LLaMA 系列使用 GPT-NeoX 风格。

## 4.7 ALiBi 位置编码

ALiBi（Attention with Linear Biases）是另一种位置编码方案，被 BLOOM、MPT 等模型使用。它不修改 Q/K 向量，而是直接在 attention score 上加一个与距离成正比的偏置：

```python
# ALiBi 的核心思想（伪代码）
# 对于 head h，attention score 加上：-m_h * |i - j|
# 其中 m_h 是每个 head 不同的斜率，i 和 j 是 token 位置
position_embedding_type = PositionEmbeddingType.alibi
```

ALiBi 的优势是天然支持外推到训练时未见过的序列长度，且计算开销很小。在 GPTAttention plugin 中，ALiBi 偏置的计算同样被融合在 attention kernel 内部。

## 4.8 Attention Mask 处理

LLM 推理中的 attention mask 主要有两种场景：

1. **Causal mask**：标准的自回归掩码，确保每个 token 只能关注它之前的 token。这是绝大多数场景的默认选项，在 GPTAttention plugin 中通过简单的位置比较实现，无需显式传入 mask 张量。

2. **Custom mask**：用于处理如 padding（一个 batch 中不同请求长度不同时的填充位置）等场景。TensorRT-LLM 通过 `context_lengths` 参数告知 plugin 每个请求的实际长度，plugin 在计算时会自动忽略 padding 位置。

```python
# 在 inflight batching 场景中，不同请求混合在同一 batch 中
# context_lengths 告诉 plugin 每个请求的有效长度
# 这比传入一个完整的 mask 矩阵更高效
attention_params = AttentionParams(
    context_lengths=context_lengths,       # [batch_size]
    sequence_length=sequence_length,       # 当前 step 的序列长度
    max_context_length=max_context_length, # batch 中的最大长度
)
```

## 本章小结

本章深入分析了 TensorRT-LLM 的 Attention 实现。`Attention` 类（`tensorrt_llm/layers/attention.py`）通过 `num_attention_heads` 和 `num_kv_heads` 的配置统一支持 MHA、MQA、GQA 三种模式。核心计算由 GPTAttention plugin（`cpp/tensorrt_llm/plugins/gptAttentionPlugin/`）完成，它将 QKV reshape、位置编码、KV Cache 读写和 attention 计算深度融合为一个高性能 kernel。KV Cache 支持 continuous 和 paged 两种管理方式，后者通过分页机制显著提升了显存利用率。位置编码方面，RoPE 和 ALiBi 均在 plugin 内部实现，避免了额外的 kernel launch 开销。下一章我们将从单个 Attention 层上升到完整模型的维度，看看 TensorRT-LLM 是如何组织和管理其丰富的内置模型库的。
