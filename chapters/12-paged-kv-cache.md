# 第 12 章：Paged KV Cache

> "计算机科学中的每一个问题都可以通过增加一层间接性来解决。" —— David Wheeler

KV Cache 是大语言模型推理中最重要的优化手段之一，而 Paged KV Cache 则是让这一优化在高并发场景下真正可扩展的关键技术。本章将从 KV Cache 的基本原理出发，深入分析 TensorRT-LLM 中 Paged KV Cache 的 C++ 实现，并与 vLLM 的方案进行对比。

## 12.1 KV Cache 在 LLM 推理中的作用

在 Transformer 的自回归生成过程中，每生成一个新 token 都需要对之前所有 token 执行注意力计算。朴素实现意味着生成第 N 个 token 时，需要重新计算前 N-1 个 token 的 Key 和 Value——这是极其浪费的，因为这些值在之前的步骤中已经计算过。

KV Cache 的核心思想是：将每一步计算出的 Key 和 Value 向量缓存起来，后续步骤直接读取缓存而非重新计算。这将生成阶段每步的计算复杂度从 O(N) 降低到 O(1)（仅计算新 token 的 K、V），代价是需要 O(N) 的 GPU 显存来存储缓存。

对于一个典型的 70B 参数模型（80 层、8 个 KV head、128 维 head size、FP16 精度），单个序列 2048 个 token 的 KV Cache 占用：

```
80 层 × 2(K+V) × 8 heads × 128 dim × 2048 tokens × 2 bytes(FP16)
= 80 × 2 × 8 × 128 × 2048 × 2
≈ 6.4 GB
```

当需要同时处理数十甚至上百个并发请求时，KV Cache 的显存占用迅速成为瓶颈。

## 12.2 连续 KV Cache 的局限

最简单的 KV Cache 实现是**连续分配**——为每个序列预分配一段连续的 GPU 内存，大小按最大可能的序列长度计算：

```python
# 连续 KV Cache 的简化示意
kv_cache = torch.zeros(
    num_layers, 2, batch_size, num_kv_heads,
    max_seq_len, head_dim,
    dtype=torch.float16, device='cuda'
)
```

这种方式有两个严重问题：

**内存浪费**：每个序列都按 `max_seq_len` 分配，但实际生成的长度通常远小于上限。如果 `max_seq_len=4096` 而平均生成长度为 500，则约 88% 的预分配内存是浪费的。

**内存碎片**：不同序列的生命周期不同——短序列完成后释放的内存可能无法被长序列使用，导致虽然总空闲显存足够，但找不到足够大的连续块来分配。

## 12.3 Paged KV Cache 的设计思想

Paged KV Cache 借鉴了操作系统虚拟内存的分页机制：将 KV Cache 划分为固定大小的 **block**，每个 block 存储固定数量 token 的 KV 数据。序列的 KV Cache 由一个 block 链表组成，block 之间不需要物理上连续。

```
逻辑视图（序列 A，10 个 token，block_size=4）:
[Token 0-3] → [Token 4-7] → [Token 8-9, 空, 空]
   Block 5       Block 12       Block 3

逻辑视图（序列 B，6 个 token）:
[Token 0-3] → [Token 4-5, 空, 空]
   Block 8       Block 1
```

这种设计解决了连续分配的两个问题：内存浪费被限制在每个序列最后一个 block 的内部碎片（平均浪费 `block_size/2` 个 token 的空间）；block 可以在任意位置分配，不需要连续内存，彻底消除了外部碎片。

## 12.4 TRT-LLM 的实现：KvCacheManager

TRT-LLM 的 Paged KV Cache 核心实现在 `cpp/tensorrt_llm/runtime/kvCacheManager.h` 和对应的 `.cpp` 文件中：

```cpp
class KvCacheManager {
public:
    KvCacheManager(SizeType numLayers, SizeType numKvHeads,
                   SizeType headDim, SizeType blockSize,
                   SizeType maxBlocksPerSeq, SizeType maxNumSeqs,
                   nvinfer1::DataType dataType,
                   CudaStreamPtr stream);

    // 为新序列分配 block
    void allocateBlocks(SizeType seqIdx, SizeType numTokens);

    // 追加新 token 时可能需要分配新 block
    void appendToken(SizeType seqIdx);

    // 释放序列的所有 block
    void releaseBlocks(SizeType seqIdx);

    // 获取 block 指针表（传给 attention kernel）
    ITensor::SharedPtr getBlockPointersOfBatch() const;

    // 查询可用 block 数
    SizeType numFreeBlocks() const;

private:
    // Block 池：所有可用的物理 block
    std::vector<KvCacheBlock> mFreeBlocks;

    // 每个序列的 block 链表
    std::vector<std::vector<KvCacheBlock*>> mSequenceBlockLists;

    // 预分配的 KV Cache 物理内存
    IBufferPtr mKvCachePool;
};
```

### Block 的分配与释放

当一个新请求到达时，`allocateBlocks()` 根据输入 prompt 的长度计算需要的 block 数量，从 `mFreeBlocks` 池中取出相应数量的 block，构成该序列的 block 链表。计算公式为：

```cpp
SizeType numBlocks = (numTokens + mBlockSize - 1) / mBlockSize;
```

在 Generation Phase 中，每生成一个新 token，`appendToken()` 检查当前最后一个 block 是否还有空位：如果有，直接写入；如果已满，从 free pool 中分配一个新 block 并追加到链表尾部。

当序列生成完毕，`releaseBlocks()` 将该序列的所有 block 归还到 free pool。这些 block 可以立即被其他新请求复用，无需任何 `cudaMalloc` 或 `cudaFree`。

### Block 指针表

Attention kernel 需要知道每个序列的 KV Cache 数据存储在哪些物理 block 中。`getBlockPointersOfBatch()` 返回一个二维表，行代表序列，列代表该序列的第 i 个逻辑 block 对应的物理地址：

```cpp
// block_pointers[batch_idx][block_idx] = 物理 block 的 GPU 地址
// Attention kernel 通过这个表来间接寻址 KV Cache
```

这个间接寻址表是 Paged KV Cache 的核心数据结构，它解耦了逻辑上连续的 token 序列和物理上分散的 GPU 内存块。

## 12.5 KvCacheConfig 配置参数

用户通过 `KvCacheConfig` 来控制 Paged KV Cache 的行为：

```python
from tensorrt_llm import KvCacheConfig

kv_cache_config = KvCacheConfig(
    max_tokens_in_paged_kv_cache=81920,   # KV Cache 最大容量（token 数）
    kv_cache_free_gpu_mem_fraction=0.85,   # 使用 85% 的空闲 GPU 显存
    enable_block_reuse=True,               # 启用 block 复用
)
```

`max_tokens_in_paged_kv_cache` 直接决定了 KV Cache 池的大小。如果设置过小，并发请求数会受限；如果设置过大，可能挤占模型权重和计算所需的显存。实践中更常用 `kv_cache_free_gpu_mem_fraction`——让 TRT-LLM 自动计算加载模型后剩余显存的指定比例用于 KV Cache。

`enable_block_reuse` 是一个高级优化选项。当多个请求共享相同的前缀（例如使用相同的 system prompt）时，这些请求可以共享前缀部分的 KV Cache block，避免重复计算和存储。这在 RAG（Retrieval-Augmented Generation）等场景中尤其有效。

## 12.6 Block 大小选择策略

Block 大小是影响 Paged KV Cache 性能的重要超参数。TRT-LLM 默认的 block 大小通常为 64 或 128 个 token。选择需要权衡以下因素：

**较大的 block**（如 128）：内部碎片更大（平均浪费 64 token 的空间），但 block 指针表更小，attention kernel 的间接寻址开销更低，内存访问更连续。

**较小的 block**（如 32）：内部碎片更小，内存利用率更高，但 block 指针表更大，间接寻址更频繁，可能影响 attention kernel 的性能。

在实践中，block 大小还需要与 attention kernel 的实现对齐。TRT-LLM 的 flash attention 和 XQA kernel 对特定的 block 大小有优化路径，偏离这些预设值可能导致性能回退。

## 12.7 KV Cache 量化：FP8 KV Cache

为了进一步降低 KV Cache 的显存占用，TRT-LLM 支持 FP8 KV Cache 量化。原理是将原本以 FP16 存储的 K、V 向量量化为 FP8（E4M3）格式，显存占用直接减半：

```python
# 构建 engine 时启用 FP8 KV Cache
builder_config = BuilderConfig(
    # ...
    use_fp8_kv_cache=True,
)
```

FP8 KV Cache 的量化在写入缓存时进行，反量化在 attention kernel 读取时进行。TRT-LLM 使用 per-tensor 或 per-channel 的缩放因子来最小化量化误差：

```cpp
// 简化示意：FP8 量化写入
fp8_value = static_cast<__nv_fp8_e4m3>(fp16_value / scale);

// 反量化读取
fp16_value = static_cast<half>(fp8_value) * scale;
```

实验表明，FP8 KV Cache 在大多数模型上的精度损失微乎其微（perplexity 变化通常小于 0.1%），但显存节省是实实在在的——这意味着同样的 GPU 可以支持近两倍的并发序列。

## 12.8 与 vLLM PagedAttention 的对比

vLLM 是最早在开源社区推广 Paged KV Cache（PagedAttention）概念的框架。对比两者的实现有助于理解 TRT-LLM 的设计选择：

| 维度 | vLLM | TRT-LLM |
|------|------|---------|
| 实现语言 | Python + CUDA kernel | C++ + CUDA kernel |
| Block 管理 | Python 层的 BlockAllocator | C++ 层的 KvCacheManager |
| Attention kernel | 自研 PagedAttention CUDA kernel | 集成 Flash Attention + 自研 XQA kernel |
| 量化支持 | FP8（后期版本） | FP8、INT8 |
| Block 复用 | 支持（prefix caching） | 支持（enable_block_reuse） |
| 调度集成 | Python 调度器直接管理 | C++ Batch Manager 管理 |

vLLM 的优势在于 Python 实现的灵活性和可扩展性——研究人员可以方便地修改调度策略和 block 管理逻辑。TRT-LLM 的优势则在于 C++ 实现的性能——block 管理的开销更低，与 TensorRT engine 的集成更紧密，且避免了 Python GIL 对高并发场景的影响。

两者在设计理念上是趋同的，核心区别在于工程实现的语言和优化侧重点。vLLM 更适合快速原型开发和学术研究，TRT-LLM 更适合对延迟敏感的生产环境部署。

## 本章小结

本章系统分析了 Paged KV Cache 的设计原理与 TRT-LLM 的具体实现。KV Cache 通过缓存历史 token 的 Key 和 Value 避免重复计算，是 LLM 推理的基础优化。传统的连续分配方式存在严重的内存浪费和碎片问题，Paged KV Cache 借鉴操作系统虚拟内存分页思想，将 KV Cache 划分为固定大小的 block 进行动态管理。TRT-LLM 的 `KvCacheManager`（位于 `cpp/tensorrt_llm/runtime/kvCacheManager.h`）实现了 block 的分配、释放、复用和指针表管理。FP8 KV Cache 量化将显存占用减半且精度损失极小。Block 大小的选择需要在内存利用率和 attention kernel 性能之间取得平衡。与 vLLM 相比，TRT-LLM 的 C++ 实现在性能上更具优势，而 vLLM 在灵活性上更胜一筹。
