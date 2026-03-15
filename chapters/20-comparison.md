# 第 20 章：与 vLLM/SGLang 选型对比

> "没有最好的工具，只有最适合的工具。工程决策的本质是在约束条件下寻找最优解。"

经过前面 19 章对 TensorRT-LLM 源码的深入分析，读者应该已经对其架构、原理和能力有了全面的认识。但在真实的技术选型中，TensorRT-LLM 不是唯一的选择。vLLM 和 SGLang 是当前最活跃的两个开源 LLM 推理框架，各有其独特的设计哲学和适用场景。本章将从多个维度进行系统性的对比分析，帮助读者在不同场景下做出理性的技术决策。

## 20.1 编译型 vs 解释型：根本哲学差异

TensorRT-LLM 和 vLLM 的根本差异，类似于 C++ 和 Python 的差异——前者是编译型推理框架，后者是解释型推理框架。

**TensorRT-LLM 的编译型路线：**

```
模型权重 + 配置
      │
      ▼
 trtllm-build（编译阶段）
      │  - 算子融合
      │  - 内存布局优化
      │  - Kernel 自动调优
      ▼
 TensorRT Engine（二进制文件）
      │
      ▼
  Runtime 加载执行
```

编译阶段可能耗时数分钟到数十分钟，但生成的引擎文件是针对特定 GPU 架构深度优化的。每次修改模型配置（如 batch size 上限、序列长度、量化精度）都需要重新编译。

**vLLM 的解释型路线：**

```python
# vLLM 的启动过程——直接加载模型权重
from vllm import LLM

# 无需预编译，直接从 HuggingFace 权重启动
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,
    dtype="float16",
)
```

vLLM 在运行时动态编译 CUDA kernel（通过 PyTorch 的 JIT 和自定义 CUDA 扩展），无需离线编译步骤。这种方式牺牲了一部分极致性能，换取了显著的灵活性。

## 20.2 核心差异维度详细对比

以下从八个关键维度进行深入对比：

### 部署复杂度

TensorRT-LLM 的部署链路较长：安装依赖 -> 转换权重 -> 构建引擎 -> 配置 Triton -> 启动服务。中间任何一个环节出错都可能导致部署失败。相关的代码路径包括 `tensorrt_llm/commands/build.py` 中的构建逻辑和 `tensorrt_llm/models/` 下各模型的转换脚本。

```bash
# TRT-LLM 部署流程（简化）
# Step 1: 转换权重
python convert_checkpoint.py \
    --model_dir llama-7b-hf \
    --output_dir converted/ \
    --dtype float16

# Step 2: 构建引擎
trtllm-build \
    --checkpoint_dir converted/ \
    --output_dir engines/ \
    --gemm_plugin float16 \
    --max_batch_size 256 \
    --max_input_len 2048 \
    --max_seq_len 4096

# Step 3: 启动服务
trtllm-serve serve --engine_dir engines/ --tokenizer_dir llama-7b-hf
```

```bash
# vLLM 部署流程
vllm serve meta-llama/Llama-2-7b-hf --dtype float16
# 一行命令，完成
```

### 灵活性

vLLM 的灵活性体现在运行时可调参数远多于 TensorRT-LLM。例如，vLLM 可以在运行时动态调整 `max_num_seqs`（最大并发序列数），而 TensorRT-LLM 的 `max_batch_size` 是在引擎构建时确定的。

### 性能

这是 TensorRT-LLM 最大的优势领域。通过离线编译优化，TensorRT-LLM 在以下方面具有结构性优势：

- **算子融合**：将多个小 kernel 融合为一个大 kernel，减少 launch overhead
- **内存优化**：编译器可以全局分析内存使用模式，优化显存分配
- **Kernel 自动调优**：针对具体矩阵尺寸选择最优的 GEMM 实现
- **FP8 深度优化**：在 Hopper 架构上充分利用 FP8 Tensor Core

### 模型支持

```
模型             | TRT-LLM | vLLM  | SGLang
-----------------|---------|-------|-------
LLaMA/LLaMA-2/3 |    ✓    |   ✓   |   ✓
Mistral/Mixtral  |    ✓    |   ✓   |   ✓
GPT-2/GPT-J      |    ✓    |   ✓   |   ✓
ChatGLM           |    ✓    |   ✓   |   △
Qwen/Qwen-2      |    ✓    |   ✓   |   ✓
Falcon            |    ✓    |   ✓   |   △
Phi-3             |    ✓    |   ✓   |   ✓
DeepSeek-V2/V3   |    ✓    |   ✓   |   ✓
自定义模型        |   困难   | 较容易 | 较容易
```

vLLM 和 SGLang 由于采用 PyTorch 生态，添加新模型的门槛较低——本质上只需要实现一个 `nn.Module`。而 TensorRT-LLM 需要在 `tensorrt_llm/models/` 下用 TensorRT 的 Python API 重新定义模型的计算图，学习曲线更陡峭。

## 20.3 vLLM 的核心优势

**快速迭代**：vLLM 的发版节奏极快，通常每 1-2 周一个版本。社区贡献者数量庞大，新特性从提出到合并的周期短。

**社区活跃**：GitHub star 数量、issue 响应速度、Discord 社区活跃度都远超 TensorRT-LLM。当你遇到问题时，更容易找到相似的 issue 或者获得社区帮助。

**部署简单**：一行命令启动，支持直接从 HuggingFace Hub 加载模型，无需预编译步骤。对于快速原型验证和开发环境极为友好。

**OpenAI 兼容性好**：vLLM 的 OpenAI 兼容 API（`vllm.entrypoints.openai.api_server`）经过了大量真实场景验证，兼容性极佳。

```python
# vLLM 的离线推理——简洁直观
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

outputs = llm.generate(["What is deep learning?"], params)
for output in outputs:
    print(output.outputs[0].text)
```

## 20.4 TensorRT-LLM 的核心优势

**极致性能**：在高并发、大 batch size 场景下，TensorRT-LLM 的吞吐量优势显著。对于推理成本敏感的生产环境，每提升 10% 的吞吐量都意味着真金白银的节省。

**NVIDIA 官方优化**：TensorRT-LLM 团队可以第一时间获得新 GPU 架构的底层优化支持。例如 H100 的 FP8 支持、Blackwell 架构的新特性等，TensorRT-LLM 通常是最早支持的推理框架。

**FP8 支持成熟**：TensorRT-LLM 对 FP8 量化的支持是最完善的，包括 per-tensor 和 per-channel 两种校准模式。这在 H100 上是一个非常重要的性能杠杆。

**与 NVIDIA 生态深度集成**：Triton Inference Server、NVIDIA NIM、TensorRT 优化器等 NVIDIA 工具链与 TensorRT-LLM 的集成是原生的、无缝的。

## 20.5 SGLang 的独特定位

SGLang（Structured Generation Language）是 UC Berkeley 团队推出的推理框架，它的核心创新在于两个方面：

**RadixAttention**：SGLang 使用基数树（Radix Tree）来管理 KV cache 的复用。当多个请求共享相同的 prefix（例如相同的 system prompt）时，SGLang 可以高效地复用这部分 KV cache，避免重复计算：

```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))
```

在上面的代码中，如果多个请求使用相同的 system prompt，RadixAttention 会自动复用已缓存的 KV 数据。

**结构化生成优化**：SGLang 对 JSON schema 约束生成、正则表达式约束生成等结构化输出场景做了专门的优化。在需要模型输出严格 JSON 格式的应用中（如 function calling、数据提取），SGLang 的性能优势明显。

## 20.6 选型决策树

根据不同场景，以下决策树可以帮助快速做出选择：

```
开始
  │
  ├── 是否需要极致推理性能？
  │     ├── 是 → 生产环境是否使用 NVIDIA GPU？
  │     │         ├── 是 → TensorRT-LLM
  │     │         └── 否 → vLLM（支持 AMD ROCm）
  │     └── 否 ↓
  │
  ├── 是否需要快速迭代和原型验证？
  │     ├── 是 → vLLM
  │     └── 否 ↓
  │
  ├── 是否有大量结构化生成需求？
  │     ├── 是 → SGLang
  │     └── 否 ↓
  │
  ├── 是否需要支持自定义/非主流模型？
  │     ├── 是 → vLLM
  │     └── 否 ↓
  │
  ├── 是否已有 NVIDIA Triton 基础设施？
  │     ├── 是 → TensorRT-LLM
  │     └── 否 → vLLM（默认推荐）
```

需要强调的是，这个决策树是简化的指导，实际选型还需要考虑团队的技术栈背景、运维能力和长期维护成本。

## 20.7 混合部署策略

在实践中，越来越多的团队采用混合部署策略：

**开发阶段用 vLLM**：快速加载模型、调试 prompt、验证效果。无需等待引擎编译，修改参数即时生效。

**生产阶段用 TensorRT-LLM**：当模型和配置稳定后，使用 TensorRT-LLM 构建优化引擎，部署到生产环境以获得最佳性价比。

```python
# 统一的推理接口抽象
from abc import ABC, abstractmethod

class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, prompt: str, params: dict) -> str:
        pass

class VLLMBackend(LLMBackend):
    """开发环境使用"""
    def __init__(self, model_name: str):
        from vllm import LLM
        self.llm = LLM(model=model_name)

    async def generate(self, prompt, params):
        from vllm import SamplingParams
        output = self.llm.generate([prompt], SamplingParams(**params))
        return output[0].outputs[0].text

class TRTLLMBackend(LLMBackend):
    """生产环境使用"""
    def __init__(self, engine_dir: str):
        import tensorrt_llm
        self.executor = tensorrt_llm.Executor(
            engine_dir,
            tensorrt_llm.ModelType.DECODER_ONLY,
            tensorrt_llm.ExecutorConfig(max_beam_width=1)
        )

    async def generate(self, prompt, params):
        request = tensorrt_llm.Request(
            input_token_ids=self.tokenize(prompt),
            max_tokens=params.get("max_tokens", 256),
        )
        req_id = self.executor.enqueue_request(request)
        responses = self.executor.await_responses(req_id)
        return self.detokenize(responses[-1].output_token_ids)

# 通过环境变量切换后端
import os
backend_type = os.getenv("LLM_BACKEND", "vllm")
if backend_type == "trtllm":
    backend = TRTLLMBackend("/engines/llama-7b")
else:
    backend = VLLMBackend("meta-llama/Llama-2-7b-hf")
```

这种抽象层的好处是上层业务代码完全不需要感知底层推理框架的差异。

## 20.8 未来趋势：融合与互通

LLM 推理框架的生态正在走向融合：

**vLLM 集成 TensorRT 后端**：vLLM 已经在实验性地支持将 TensorRT 作为执行后端。这意味着用户可以享受 vLLM 的易用性和 TensorRT 的高性能。在 `vllm/engine/` 相关代码中已经可以看到对 TensorRT 后端的适配工作。

**TensorRT-LLM 简化部署流程**：NVIDIA 也意识到了部署复杂度的问题，`trtllm-serve` 命令和 NVIDIA NIM 容器都在朝着一键部署的方向努力。

**硬件抽象层的统一**：随着 AMD、Intel 等厂商的 AI 加速器逐渐成熟，框架之间对硬件后端的抽象正在趋同。OpenAI Triton（编程语言，非推理服务器）作为 kernel 开发的中间层，正在被多个框架采用。

**标准化推理 API**：OpenAI API 格式已成为事实标准，所有主流框架都在提供兼容层。未来可能会出现更正式的行业标准。

这些趋势表明，选择任何一个框架都不是终身承诺。保持架构的灵活性，做好后端抽象，让你在需要时可以平滑地切换推理框架，这才是最重要的工程智慧。

## 本章小结

本章从编译型 vs 解释型的根本哲学差异出发，对 TensorRT-LLM、vLLM 和 SGLang 三大 LLM 推理框架进行了系统性对比。TensorRT-LLM 的优势在于极致性能和 NVIDIA 生态集成；vLLM 的优势在于部署简单、社区活跃和快速迭代；SGLang 则在 KV cache 复用和结构化生成方面有独特创新。在实际选型中，建议采用混合部署策略——开发用 vLLM，生产用 TensorRT-LLM——并通过统一的后端抽象层隔离底层差异。技术选型没有标准答案，关键在于理解自己的需求和约束，做出最适合当前阶段的选择。
