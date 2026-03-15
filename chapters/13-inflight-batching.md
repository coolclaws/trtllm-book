# 第 13 章：In-flight Batching

> "不要让最快的人等最慢的人，让每个人按自己的速度前进。" —— 并发系统设计的第一原则

在传统的 LLM 推理服务中，一个 batch 内的所有请求必须同时开始、同时结束。这种 static batching 方式在面对长度差异巨大的请求时，造成了严重的 GPU 资源浪费。In-flight Batching（也称 continuous batching）从根本上改变了这一范式。本章将深入分析 TRT-LLM 中 In-flight Batching 的实现，包括调度策略、请求状态机、与 Paged KV Cache 的协同，以及 `GptManager` 的架构设计。

## 13.1 Static Batching 的问题

在 static batching 中，服务端收集一批请求，一起送入模型推理，等所有请求都生成完毕后再统一返回结果：

```
时间轴 →
请求 A (生成 20 tokens): [======]
请求 B (生成 100 tokens): [==============================]
请求 C (生成 15 tokens): [=====]
请求 D (生成 80 tokens):  [========================]

GPU 实际执行:             [==============================]
                          ↑ A,C 早已完成，但 GPU 仍在为它们保留资源
```

问题显而易见：请求 A 和 C 在第 20 步和第 15 步就已经生成了 EOS token，但它们的 KV Cache 空间一直被占用到第 100 步（请求 B 完成时）才释放。更糟糕的是，在请求 A 和 C 完成后，它们对应的 batch 槽位处于空转状态——GPU 在为这些已完成的请求执行无意义的 padding 计算。

在真实的在线服务中，请求长度的差异可能非常大（从几个 token 到几千个 token），static batching 的 GPU 利用率通常只有 30-50%。

## 13.2 In-flight Batching 的核心思想

In-flight Batching 的核心思想是：在 **iteration 级别**（而非 batch 级别）进行调度。每完成一步 token 生成，调度器都有机会：

1. 将已完成的请求移出 batch，释放其资源
2. 将等待队列中的新请求插入 batch

```
时间轴 →
请求 A (20 tokens):  [======]
请求 C (15 tokens):  [=====]
请求 E (新请求):           [===========]  ← 在 C 完成后立即加入
请求 B (100 tokens): [==============================]
请求 F (新请求):            [========]    ← 在 A 完成后立即加入
请求 D (80 tokens):  [========================]
请求 G (新请求):                          [====] ← 在 D 完成后加入
```

这种方式使得 GPU 的 batch 槽位始终被有效请求占据，利用率可以提升到 80-95%。TRT-LLM 的文档中将这种技术也称为 "continuous batching"。

## 13.3 请求状态机

在 TRT-LLM 的实现中，每个请求在其生命周期内经历以下状态转换：

```
                    ┌──────────────┐
  新请求到达 ──────→│ CONTEXT_INIT │
                    └──────┬───────┘
                           │ Context Phase 完成
                           ▼
                    ┌──────────────┐
                    │  GENERATION  │←──── 每步生成一个 token
                    └──────┬───────┘      并循环回此状态
                           │ 满足停止条件
                           ▼
                    ┌──────────────┐
                    │   FINISHED   │
                    └──────────────┘
```

**CONTEXT_INIT**：请求刚被调度器接受，需要执行 Context Phase——一次性处理完整的输入 prompt，填充 KV Cache。这个阶段的计算量与输入长度成正比，是计算密集型操作。

**GENERATION**：Context Phase 完成后进入生成阶段。每个 iteration 生成一个 token，更新 KV Cache。这个阶段是访存密集型操作。

**FINISHED**：当生成的 token 满足停止条件时，请求标记为完成。停止条件包括：生成了 EOS token、达到 `max_new_tokens` 上限、命中用户指定的 stop words。

In-flight Batching 的关键在于：处于 CONTEXT_INIT 和 GENERATION 状态的请求可以在同一个 batch 中共存。这意味着在一次 GPU iteration 中，部分请求在执行 Context Phase，其余请求在执行 Generation Phase。

## 13.4 TRT-LLM 的调度实现

TRT-LLM 的 In-flight Batching 实现主要位于 `cpp/tensorrt_llm/batch_manager/` 目录下。核心组件包括调度器（Scheduler）、请求管理和 batch 组装逻辑。

### SchedulerPolicy

调度器支持两种策略，通过 `SchedulerPolicy` 枚举控制：

```cpp
// cpp/include/tensorrt_llm/batch_manager/schedulerPolicy.h
enum class SchedulerPolicy {
    kMAX_UTILIZATION,
    kGUARANTEED_NO_EVICT
};
```

**MAX_UTILIZATION**：最大化 GPU 利用率。调度器会尽可能多地将等待队列中的请求加入当前 batch，即使这意味着当 GPU 显存不足时，某些已在运行的请求需要被"驱逐"（暂停并释放其 KV Cache）。被驱逐的请求稍后会被重新调度，代价是需要重新执行 Context Phase。

```python
# MAX_UTILIZATION 的调度伪代码
def schedule(waiting_queue, active_requests, free_blocks):
    # 尝试加入尽可能多的新请求
    for req in waiting_queue:
        blocks_needed = estimate_blocks(req)
        if blocks_needed <= free_blocks:
            activate(req)
            free_blocks -= blocks_needed
        else:
            # 如果显存不足，考虑驱逐低优先级请求
            evictable = find_evictable(active_requests)
            if evictable:
                evict(evictable)
                free_blocks += evictable.blocks
                activate(req)
```

**GUARANTEED_NO_EVICT**：保证已接受的请求不会被驱逐。调度器只有在确认有足够资源容纳新请求的完整生成过程后才会接受它。这意味着更保守的并发度，但已运行的请求不会被中断，延迟更加可预测。

```python
# GUARANTEED_NO_EVICT 的调度伪代码
def schedule(waiting_queue, active_requests, free_blocks):
    for req in waiting_queue:
        # 估算请求整个生命周期需要的最大 block 数
        max_blocks = estimate_max_blocks(req)
        if max_blocks <= free_blocks:
            activate(req)
            free_blocks -= max_blocks
        else:
            break  # 资源不足，停止接受新请求
```

选择哪种策略取决于业务需求：在线聊天服务通常选择 `GUARANTEED_NO_EVICT` 以保证稳定的首 token 延迟（TTFT）；批量处理场景更适合 `MAX_UTILIZATION` 以最大化吞吐。

### Iteration 级别调度

In-flight Batching 的核心循环在 batch manager 的主执行线程中：

```cpp
// 简化的主循环示意
while (running) {
    // 1. 检查已完成的请求，释放资源
    for (auto& req : active_requests) {
        if (req.isFinished()) {
            releaseKvCacheBlocks(req);
            moveToCompleted(req);
        }
    }

    // 2. 从等待队列中调度新请求
    auto new_requests = scheduler.schedule(
        waiting_queue, active_requests, kvCacheManager.numFreeBlocks());

    // 3. 为新请求执行 Context Phase
    for (auto& req : new_requests) {
        kvCacheManager.allocateBlocks(req);
        executeContextPhase(req);
        req.setState(RequestState::GENERATION);
    }

    // 4. 为所有 GENERATION 状态的请求组装 batch 并执行一步
    auto batch = assembleBatch(active_requests);
    executeGenerationStep(batch);

    // 5. 处理采样结果，更新请求状态
    for (auto& req : batch) {
        auto token = sampleToken(req);
        req.appendToken(token);
        kvCacheManager.appendToken(req);
        if (checkStopCondition(req, token)) {
            req.setState(RequestState::FINISHED);
        }
    }
}
```

每一次循环就是一个 iteration。在每个 iteration 中，调度器都有机会加入新请求和释放已完成请求的资源，从而实现 batch 的动态调整。

## 13.5 与 Paged KV Cache 的协同

In-flight Batching 和 Paged KV Cache 是天然互补的技术。没有 Paged KV Cache，In-flight Batching 的效果会大打折扣——因为连续分配的 KV Cache 无法灵活地分配和回收。

具体的协同体现在：

**动态分配**：新请求加入 batch 时，`KvCacheManager` 从 block pool 中分配所需的 block。由于 block 是固定大小的小单元，分配操作是 O(1) 的。

**逐步增长**：在 Generation Phase 中，每生成一个 token，可能需要一个新的 block。`KvCacheManager.appendToken()` 在当前 block 填满时自动分配新 block。

**即时回收**：请求完成后，其所有 block 立即归还到 free pool，可以被下一个 iteration 中加入的新请求使用。不需要等待整个 batch 完成。

```cpp
// 请求完成时的资源回收
void onRequestFinished(Request& req) {
    // 释放 KV Cache blocks -> 即时可被新请求使用
    kvCacheManager.releaseBlocks(req.sequenceIndex());

    // 将结果放入输出队列
    outputQueue.push(req.getResult());
}
```

如果使用连续 KV Cache，新请求必须等到一整段连续内存被释放后才能被分配——这可能需要等待多个请求同时完成，严重限制了 In-flight Batching 的灵活性。

## 13.6 GptManager：批处理管理器

`GptManager` 是 TRT-LLM 中 In-flight Batching 的顶层管理类，定义在 `cpp/include/tensorrt_llm/batch_manager/GptManager.h` 中：

```cpp
class GptManager {
public:
    GptManager(std::filesystem::path const& trtEnginePath,
               TrtGptModelType modelType,
               GetInferenceRequestsCallback getInferenceRequestsCb,
               SendResponseCallback sendResponseCb,
               std::optional<SchedulerPolicy> schedulerPolicy,
               std::optional<KvCacheConfig> kvCacheConfig);

    void shutdown();

private:
    // 主执行线程
    void executionLoop();

    std::unique_ptr<TrtGptModel> mModel;
    std::unique_ptr<Scheduler> mScheduler;
    std::unique_ptr<KvCacheManager> mKvCacheManager;
    std::thread mExecutionThread;
};
```

`GptManager` 采用回调模式与外部系统交互。`GetInferenceRequestsCallback` 用于从请求队列获取新的推理请求，`SendResponseCallback` 用于将生成结果发送回客户端。这种设计使得 `GptManager` 可以灵活集成到各种服务框架中——无论是 gRPC 服务、HTTP 服务还是 Triton Inference Server。

`executionLoop()` 在独立线程中运行上述的 iteration 级别调度循环。它不断从回调获取新请求、执行推理、通过回调发送结果。

## 13.7 请求队列管理与停止条件

TRT-LLM 的请求队列支持优先级排序。在 `MAX_UTILIZATION` 策略下，调度器在需要驱逐请求时会优先驱逐低优先级的请求：

```cpp
struct InferenceRequest {
    IdType requestId;
    std::vector<TokenIdType> inputTokens;
    SamplingConfig samplingConfig;
    SizeType maxNewTokens;
    std::optional<PriorityType> priority;
    std::optional<std::vector<std::vector<TokenIdType>>> stopWords;
};
```

停止条件的检测在每个 iteration 结束后进行：

```cpp
bool checkStopCondition(Request const& req, TokenIdType newToken) {
    // 1. 检查 EOS token
    if (newToken == req.endId()) return true;

    // 2. 检查最大生成长度
    if (req.generatedLength() >= req.maxNewTokens()) return true;

    // 3. 检查 stop words
    if (req.hasStopWords()) {
        auto const& generated = req.generatedTokens();
        for (auto const& stopWord : req.stopWords()) {
            if (endsWith(generated, stopWord)) return true;
        }
    }

    return false;
}
```

Stop words 的检测需要匹配 token 序列的后缀，而非单个 token。例如，stop word `"\n\nHuman:"` 可能被 tokenizer 编码为多个 token，需要检查最近生成的若干 token 是否恰好构成这个序列。

## 13.8 性能收益量化

In-flight Batching 带来的性能提升取决于工作负载的特征。以下是一些典型场景的对比数据（以相对吞吐量表示）：

| 场景 | Static Batching | In-flight Batching | 提升 |
|------|----------------|-------------------|------|
| 输出长度均匀（偏差 <10%） | 1.0x | 1.1x | ~10% |
| 输出长度中等差异（偏差 ~50%） | 1.0x | 1.8x | ~80% |
| 输出长度差异大（偏差 >100%） | 1.0x | 2.5x | ~150% |
| 混合短查询与长生成 | 1.0x | 3.0x+ | ~200%+ |

当请求长度差异越大时，In-flight Batching 的优势越明显。在极端情况下（如同时处理简单问答和长文章生成），吞吐量提升可达 3 倍以上。

## 本章小结

本章深入分析了 TensorRT-LLM 的 In-flight Batching 实现。传统的 static batching 因短序列等待长序列而造成严重的 GPU 资源浪费，In-flight Batching 通过 iteration 级别的调度——在每一步都可以加入新请求、移除已完成请求——从根本上解决了这一问题。TRT-LLM 的实现位于 `cpp/tensorrt_llm/batch_manager/` 目录下，核心包括两种调度策略（`MAX_UTILIZATION` 追求最大吞吐、`GUARANTEED_NO_EVICT` 保证延迟稳定性）和清晰的请求状态机（CONTEXT_INIT → GENERATION → FINISHED）。In-flight Batching 与 Paged KV Cache 天然协同——block 的即时分配与回收使得 batch 的动态调整得以高效实现。`GptManager` 作为顶层管理类，通过回调模式灵活集成到各种服务框架中。在请求长度差异较大的真实场景中，In-flight Batching 可以带来数倍的吞吐量提升。
