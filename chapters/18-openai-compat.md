# 第 18 章：OpenAI API 兼容

> "最好的 API 迁移是用户完全感知不到的迁移。" —— 某大型语言模型平台架构师

当企业决定将推理后端从 OpenAI API 切换到自托管的 TensorRT-LLM 时，最大的阻力往往不是性能或成本，而是已有代码的兼容性。数以万计的应用程序已经围绕 OpenAI 的 `/v1/chat/completions` 接口构建，任何不兼容的改动都意味着巨大的迁移成本。本章讨论 TensorRT-LLM 生态中的 OpenAI API 兼容方案，让你的自托管推理服务成为 OpenAI 的 drop-in 替换。

## 18.1 为什么需要 OpenAI 兼容 API

OpenAI 的 Chat Completions API 已经成为 LLM 领域的事实标准接口。几乎所有主流框架和工具——LangChain、LlamaIndex、AutoGen、Cursor——都以 OpenAI SDK 作为默认的 LLM 调用方式。提供兼容接口的好处显而易见：

1. **零代码迁移**：只需修改 `base_url` 和 `api_key`，无需改动业务逻辑
2. **工具链复用**：所有基于 OpenAI SDK 构建的中间件直接可用
3. **渐进式迁移**：可以逐步将流量从 OpenAI 切换到自托管服务
4. **多后端统一**：同一套客户端代码可以对接不同的推理后端

```python
# 用户代码无需任何改动，仅修改连接配置
from openai import OpenAI

# 原来连接 OpenAI
# client = OpenAI(api_key="sk-xxx")

# 现在连接自托管 TensorRT-LLM 服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused"  # 自托管通常不需要真实 key
)

response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
    max_tokens=256,
    stream=True
)
```

## 18.2 TRT-LLM 的兼容方案架构

TensorRT-LLM 提供 OpenAI 兼容 API 的方式经历了几个阶段的演进。当前推荐的架构是在 Triton Backend 之上构建一个轻量级的 HTTP 前端服务：

```
                    ┌─────────────────────────┐
                    │   OpenAI Compatible API  │
                    │   (FastAPI / Flask)       │
                    │   /v1/chat/completions   │
                    │   /v1/completions         │
                    │   /v1/models              │
                    └────────────┬────────────┘
                                 │ gRPC / HTTP
                    ┌────────────▼────────────┐
                    │   Triton Inference Server │
                    │   tensorrtllm_backend     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   TensorRT-LLM Engine     │
                    └─────────────────────────┘
```

在 TensorRT-LLM 的代码仓库中，`tensorrt_llm/serve/` 目录包含了 OpenAI 兼容服务的实现。其核心入口通过 `trtllm-serve` 命令暴露：

```bash
# 使用 trtllm-serve 启动 OpenAI 兼容服务
trtllm-serve serve \
    --model_dir /engines/llama-3-8b \
    --tokenizer_dir /models/llama-3-8b \
    --host 0.0.0.0 \
    --port 8000
```

## 18.3 serve 模块源码分析

`tensorrt_llm/serve/` 目录下的关键文件结构如下：

```
tensorrt_llm/serve/
├── __init__.py
├── openai_server.py       # FastAPI 应用主体
├── openai_protocol.py     # 请求/响应数据模型
├── chat_utils.py          # chat template 处理
└── triton_client.py       # Triton gRPC 客户端封装
```

`openai_protocol.py` 中定义了与 OpenAI API 完全一致的数据结构：

```python
# tensorrt_llm/serve/openai_protocol.py（简化）
from pydantic import BaseModel, Field
from typing import List, Optional, Union

class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: Union[str, List]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
```

## 18.4 /v1/chat/completions 接口实现

核心路由处理逻辑在 `openai_server.py` 中：

```python
# openai_server.py 核心逻辑（简化）
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json, time, uuid

app = FastAPI()

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 1. 应用 chat template，将 messages 转为 prompt
    prompt = apply_chat_template(
        tokenizer=tokenizer,
        messages=request.messages,
    )

    # 2. 构建 TRT-LLM 推理参数
    sampling_params = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens or 1024,
        "stop": request.stop,
    }

    if request.stream:
        return StreamingResponse(
            stream_generator(prompt, sampling_params, request.model),
            media_type="text/event-stream"
        )
    else:
        output = await run_inference(prompt, sampling_params)
        return build_response(output, request.model)
```

这里的关键步骤是 chat template 的应用。不同的模型（LLaMA、Mistral、ChatGLM 等）有不同的对话格式，`apply_chat_template` 函数利用 HuggingFace tokenizer 内置的 Jinja2 模板来处理这种差异。

## 18.5 Streaming 响应（SSE）

Streaming 是 LLM 服务的核心特性，用户期望看到逐 token 输出而非等待完整响应。OpenAI API 使用 Server-Sent Events（SSE）协议实现 streaming：

```python
async def stream_generator(prompt, params, model_name):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    async for token_text in inference_stream(prompt, params):
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": token_text},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # 发送结束标记
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

每个 SSE 事件以 `data: ` 前缀开头，以两个换行符结尾。最后一个事件固定为 `data: [DONE]`，客户端据此判断 streaming 结束。

## 18.6 Chat Template 处理

Chat template 是将多轮对话消息转为模型可理解的纯文本 prompt 的关键环节。以 LLaMA-3 为例：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is TensorRT-LLM?"},
]

# 使用 tokenizer 内置的 chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# 输出类似：
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful assistant.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# What is TensorRT-LLM?<|eot_id|>
# <|start_header_id|>assistant<|end_header_id|>
```

如果模型没有内置 chat template，服务端需要提供一个默认模板或者允许用户通过配置文件指定。

## 18.7 /v1/completions 接口

除了 Chat Completions，OpenAI 兼容层还需要支持传统的 Completions 接口，它直接接受纯文本 prompt 而非消息列表：

```python
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    # 无需 chat template，直接使用 prompt
    output = await run_inference(request.prompt, {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    })
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"text": output, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
    }
```

## 18.8 与 vLLM OpenAI Server 的对比

vLLM 同样提供了 OpenAI 兼容的 API 服务器，且其实现更加成熟。两者的关键差异如下：

| 维度 | TRT-LLM + Triton | vLLM |
|------|-------------------|------|
| 启动方式 | `trtllm-serve` 或自建前端 | `vllm serve model_name` |
| 底层通信 | 前端 -> Triton gRPC -> Backend | 直接内嵌 |
| 兼容完整度 | 核心接口兼容 | 接近完全兼容 |
| Function Calling | 需要额外实现 | 内置支持 |
| 多模态输入 | 部分支持 | 较好支持 |
| 部署复杂度 | 较高（多组件） | 较低（单进程） |

vLLM 的 `vllm.entrypoints.openai.api_server` 模块经过了大量社区贡献者的完善，在边界情况的处理上更加细致。如果 OpenAI 兼容性是首要需求，vLLM 的方案值得认真考虑。

## 18.9 部署架构建议

推荐的生产部署架构是在 Triton + TensorRT-LLM 前面放置一个无状态的 OpenAI 兼容前端：

```bash
# 使用 Docker Compose 编排多组件部署
# docker-compose.yml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
    command: >
      tritonserver --model-repository=/models
      --http-port=8100 --grpc-port=8101 --metrics-port=8102
    volumes:
      - ./model_repo:/models
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  openai-frontend:
    image: custom-openai-frontend:latest
    ports:
      - "8000:8000"
    environment:
      - TRITON_GRPC_URL=triton:8101
      - MODEL_NAME=llama-3-8b
```

这种分离架构的好处是前端服务可以独立扩缩容，且不影响 GPU 资源的分配。

## 本章小结

本章介绍了在 TensorRT-LLM 生态中实现 OpenAI API 兼容的方案与实现细节。通过在 Triton Backend 前端放置一个轻量级的 HTTP 服务，我们可以将 TensorRT-LLM 的高性能推理能力以 OpenAI 兼容的接口暴露出来，实现对现有应用的零代码迁移。Chat template 处理、SSE streaming 和请求参数映射是实现兼容层的三个核心技术点。虽然 TRT-LLM 的 OpenAI 兼容性尚在不断完善中，但对于核心的 `/v1/chat/completions` 和 `/v1/completions` 接口，已经能够满足大多数生产场景的需求。
