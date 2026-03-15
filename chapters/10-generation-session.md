# 第 10 章：GenerationSession 与 Runner

> "好的抽象不是隐藏复杂性，而是让复杂性在需要时可触及。" —— 改编自 Joel Spolsky

在前面的章节中，我们已经深入分析了 TensorRT-LLM 的模型定义、量化策略和引擎构建流程。当一个 TensorRT engine 文件编译完成后，接下来的核心问题是：如何高效地加载并运行它？本章将围绕 `GenerationSession`、`ModelRunner` 以及 `ModelRunnerCpp` 这三个关键类，逐层剖析 TensorRT-LLM 的 Python 推理运行时。

## 10.1 GenerationSession：推理会话的基石

`GenerationSession` 定义在 `tensorrt_llm/runtime/generation.py` 中，是 TensorRT-LLM Python 运行时最核心的类。它直接管理 TensorRT engine 的加载、GPU buffer 的分配以及推理的逐步执行。

### Session 初始化流程

创建一个 `GenerationSession` 实例时，框架会执行以下关键步骤：

```python
class GenerationSession:
    def __init__(self, model_config, engine_buffer, mapping, debug_mode=False):
        # 1. 反序列化 TensorRT engine
        self.runtime = trt.Runtime(TRT_LLM_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        self.context = self.engine.create_execution_context()

        # 2. 根据 model_config 确定模型参数
        self.vocab_size = model_config.vocab_size
        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.hidden_size = model_config.hidden_size
        self.max_batch_size = model_config.max_batch_size

        # 3. 分配推理所需的 GPU buffer
        self._setup_buffers()
```

`_setup_buffers()` 方法负责预分配所有推理过程中需要的 GPU 内存，包括输入 token IDs、位置编码、注意力掩码、KV Cache 缓冲区以及输出 logits。预分配策略避免了推理过程中频繁调用 `cudaMalloc`，这对延迟敏感的在线推理至关重要。

engine buffer 的来源通常是从磁盘读取序列化后的 `.engine` 文件：

```python
with open(engine_path, 'rb') as f:
    engine_buffer = f.read()

session = GenerationSession(
    model_config=model_config,
    engine_buffer=engine_buffer,
    mapping=Mapping(world_size=tp_size, rank=rank, tp_size=tp_size)
)
```

其中 `Mapping` 对象描述了张量并行的拓扑结构——在多 GPU 推理场景下，每张卡上的 `GenerationSession` 只加载自己那份分片后的 engine。

### generate() 方法详解

`generate()` 是推理的入口方法，它的执行分为两个截然不同的阶段：

**Context Phase（预填充阶段）**：一次性处理完整的输入 prompt。这个阶段的计算特征是高度并行化的矩阵运算——输入 prompt 中的所有 token 同时参与注意力计算，生成各层的 KV Cache。

```python
def generate(self, input_ids, sampling_config, ...):
    # Context phase: 处理整个输入序列
    batch_size, input_length = input_ids.shape
    self._set_context_phase_tensors(input_ids, input_length)
    self.context.execute_v2(self.buffer_dict)
    # KV Cache 已填充完毕

    # Generation phase: 逐 token 生成
    for step in range(max_new_tokens):
        self._set_generation_phase_tensors(step)
        self.context.execute_v2(self.buffer_dict)
        next_tokens = self._sample(logits, sampling_config)
        if self._check_stop_conditions(next_tokens):
            break
```

**Generation Phase（生成阶段）**：逐 token 自回归生成。每一步只输入上一步生成的单个 token（或 beam search 下的多个候选），执行一次完整的 Transformer 前向传播，输出下一个 token 的 logits。这个阶段的特征是访存密集型——每一步都要读取完整的 KV Cache，但计算量相对较小。

两个阶段之间 KV Cache 的连续性是整个生成过程正确工作的关键。Context Phase 写入的 KV Cache 在 Generation Phase 中被持续读取和追加。

## 10.2 采样参数与解码策略

`generate()` 方法接收一个 `SamplingConfig` 对象来控制 token 的采样行为：

```python
from tensorrt_llm.runtime import SamplingConfig

sampling_config = SamplingConfig(
    end_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
    max_new_tokens=512,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    num_beams=1,           # 1 表示 greedy/sampling, >1 表示 beam search
    repetition_penalty=1.1,
)
```

**Temperature** 控制 logits 分布的平滑程度。当 `temperature=1.0` 时保持原始分布；低于 1.0 使分布更尖锐（趋向 greedy）；高于 1.0 使分布更平坦（增加随机性）。在 TRT-LLM 中，temperature 缩放在 GPU kernel 中完成，避免了额外的内存往返：

```cpp
// cpp/tensorrt_llm/kernels/samplingTopKKernels.cu
// logit[i] = logit[i] / temperature
```

**Top-K 与 Top-P** 是两种互补的截断采样策略。Top-K 保留概率最高的 K 个 token；Top-P（nucleus sampling）保留累积概率达到 P 的最小 token 集合。TRT-LLM 支持两者同时生效——先做 Top-K 截断，再在剩余候选中做 Top-P 筛选。

**Beam Search** 则是一种确定性更高的解码策略，维护 `num_beams` 个候选序列并在每一步扩展得分最高的组合。Beam Search 在 TRT-LLM 中需要额外的 KV Cache 管理逻辑，因为多个 beam 可能共享前缀的 KV Cache。

## 10.3 ModelRunner：面向用户的高层封装

直接使用 `GenerationSession` 需要手动管理大量细节——加载配置文件、反序列化 engine、处理张量并行等。`ModelRunner`（定义在 `tensorrt_llm/runtime/model_runner.py`）提供了更简洁的接口：

```python
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir(
    engine_dir="/path/to/engines",
    rank=0,
)

outputs = runner.generate(
    batch_input_ids=[input_ids],
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)
```

`from_dir()` 是最常用的工厂方法，它会自动完成以下步骤：读取 `engine_dir` 下的 `config.json` 获取模型配置；根据当前 rank 加载对应的 engine 文件；实例化 `GenerationSession`。

`ModelRunner.generate()` 内部会处理输入的 padding、attention mask 的构造、采样配置的组装，最终调用底层 `GenerationSession.generate()`。对于大多数离线推理场景，`ModelRunner` 是推荐的入口。

## 10.4 ModelRunnerCpp：C++ Executor 的 Python 桥梁

`ModelRunnerCpp` 是 TRT-LLM 推荐的高性能推理接口，它不再使用纯 Python 的 `GenerationSession`，而是通过 pybind11 绑定调用 C++ 实现的 `Executor` API：

```python
from tensorrt_llm.runtime import ModelRunnerCpp

runner = ModelRunnerCpp.from_dir(
    engine_dir="/path/to/engines",
    max_batch_size=64,
    max_input_len=1024,
    max_output_len=512,
    max_beam_width=1,
)
```

`ModelRunnerCpp` 的核心优势在于：它能利用 C++ 层的 In-flight Batching 和 Paged KV Cache，而这些功能在纯 Python 的 `GenerationSession` 中是不可用的。底层的 `Executor` 对象管理着一个请求队列，支持异步提交和结果获取：

```python
# 提交请求（非阻塞）
request_id = runner.enqueue_request(input_ids, sampling_config)

# 获取结果（可轮询或回调）
result = runner.await_response(request_id)
```

## 10.5 Batch Manager 的 Python 接口

TRT-LLM 通过 `tensorrt_llm/batch_manager/` 模块暴露了 Batch Manager 的 Python 绑定。开发者可以注册回调函数来控制请求的调度逻辑：

```python
from tensorrt_llm.batch_manager import BatchManager, SchedulerPolicy

batch_manager = BatchManager(
    engine_dir=engine_dir,
    scheduler_policy=SchedulerPolicy.MAX_UTILIZATION,
    max_num_sequences=128,
)
```

`SchedulerPolicy.MAX_UTILIZATION` 策略会尽可能多地将请求塞入当前 batch，以最大化 GPU 利用率；而 `GUARANTEED_NO_EVICT` 则保证已接受的请求不会因资源不足被驱逐。选择哪种策略取决于应用场景对吞吐量和延迟稳定性的权衡。

## 10.6 Streaming 生成模式

对于交互式应用（如聊天机器人），等待全部 token 生成完毕再返回是不可接受的。TRT-LLM 支持 streaming 模式，每生成一个（或几个）token 就立即返回给调用方：

```python
# 使用 ModelRunnerCpp 的 streaming 模式
for partial_output in runner.generate(
    batch_input_ids=[input_ids],
    max_new_tokens=256,
    streaming=True,
):
    new_token = partial_output.token_ids[0][-1]
    print(tokenizer.decode(new_token), end="", flush=True)
```

Streaming 模式的实现依赖于 C++ 层的回调机制——每完成一步 generation iteration，`Executor` 就会通过回调将中间结果推送到 Python 层的队列中。这种设计避免了 Python 层频繁轮询 C++ 层状态带来的开销。

在底层实现上，streaming 并不改变计算逻辑，只改变结果的传递方式。每一步 iteration 的 logits 采样、token 选择、KV Cache 更新等操作与非 streaming 模式完全一致。区别仅在于：非 streaming 模式在所有 token 生成完毕后一次性返回完整序列，而 streaming 模式在每一步都触发一次结果回传。

## 本章小结

本章我们从底层到高层，系统地分析了 TensorRT-LLM 的 Python 推理运行时。`GenerationSession` 是最底层的推理会话抽象，直接管理 TensorRT engine 的执行和 buffer 的分配，其 `generate()` 方法清晰地分为 Context Phase 和 Generation Phase 两个阶段。`ModelRunner` 在此之上提供了面向用户的简洁接口，自动处理配置加载和输入预处理。`ModelRunnerCpp` 则是性能最优的选择，通过 pybind11 桥接 C++ Executor，支持 In-flight Batching、Paged KV Cache 和 streaming 生成等高级特性。采样参数（temperature、top_k、top_p、beam search）控制着从 logits 到 token 的转换过程，不同的参数组合适用于不同的应用场景。理解这些运行时组件的分层设计，是有效使用和调优 TensorRT-LLM 的基础。
