# 第 17 章：Triton Backend 集成

> "推理框架的价值，不在于它能跑多快的单次请求，而在于它能否在生产环境中稳定地服务百万用户。" —— NVIDIA Triton 团队

在前面的章节中，我们深入分析了 TensorRT-LLM 的引擎构建、runtime 执行和调度机制。但在真实的生产部署中，我们还需要一个成熟的推理服务框架来处理 HTTP/gRPC 请求路由、模型版本管理、健康检查和监控等横切关注点。NVIDIA Triton Inference Server 正是为此而生的通用推理服务框架，而 `tensorrtllm_backend` 则是连接 TensorRT-LLM 与 Triton 的桥梁。

## 17.1 Triton Inference Server 概述

Triton Inference Server 是 NVIDIA 推出的开源推理服务平台，支持多种推理后端（TensorRT、PyTorch、ONNX Runtime、TensorFlow 等），提供统一的 HTTP/gRPC API。其核心优势包括：

- **多模型并发服务**：在同一个服务实例中加载和管理多个模型
- **动态 batching**：自动将多个请求合并为一个 batch 以提升吞吐
- **模型版本管理**：支持 A/B 测试和灰度发布
- **丰富的监控指标**：内置 Prometheus metrics 端点

Triton 的架构设计采用了 backend 插件机制，每种推理框架对应一个 backend 动态库。对于 TensorRT-LLM，对应的 backend 实现位于 GitHub 仓库 `triton-inference-server/tensorrtllm_backend`。

## 17.2 tensorrtllm_backend 仓库结构

`tensorrtllm_backend` 仓库是 Triton 与 TensorRT-LLM 集成的核心项目，其目录结构如下：

```
tensorrtllm_backend/
├── inflight_batcher_llm/    # 核心 backend 实现
│   ├── src/
│   │   ├── libtensorrtllm.cc   # backend 入口
│   │   ├── model_instance_state.cc
│   │   └── utils.cc
│   └── CMakeLists.txt
├── all_models/               # 模型配置模板
│   └── inflight_batcher_llm/
│       ├── preprocessing/
│       │   ├── 1/model.py
│       │   └── config.pbtxt
│       ├── tensorrt_llm/
│       │   ├── 1/           # 放置引擎文件
│       │   └── config.pbtxt
│       ├── postprocessing/
│       │   ├── 1/model.py
│       │   └── config.pbtxt
│       └── ensemble/
│           └── config.pbtxt
├── scripts/
│   └── launch_triton_server.py
└── tools/
    └── fill_template.py
```

核心的 C++ backend 实现在 `inflight_batcher_llm/src/` 目录下。`libtensorrtllm.cc` 是 Triton backend 接口的入口点，它实现了 Triton 要求的 `TRITONBACKEND_ModelInstanceExecute` 等回调函数：

```cpp
// inflight_batcher_llm/src/libtensorrtllm.cc
extern "C" {

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance,
    TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    ModelInstanceState* model_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
        instance, reinterpret_cast<void**>(&model_state)));

    // 将 Triton 请求转换为 TRT-LLM 的 InferenceRequest
    for (uint32_t r = 0; r < request_count; ++r) {
        auto trtllm_request = model_state->CreateInferenceRequest(requests[r]);
        model_state->EnqueueRequest(std::move(trtllm_request));
    }

    return nullptr;  // success
}

}  // extern "C"
```

这段代码的关键职责是将 Triton 的请求格式转换为 TensorRT-LLM 内部的 `InferenceRequest` 对象，然后交给 Executor 进行调度。

## 17.3 inflight_batcher_llm 架构

`inflight_batcher_llm` 这个名字揭示了其核心特性——支持 inflight batching（即我们在第 12 章讨论的连续批处理）。与传统的静态 batching 不同，inflight batching 允许在一个 batch 正在执行时动态插入新请求或移除已完成的请求。

在 `model_instance_state.cc` 中，backend 维护了与 TensorRT-LLM Executor 的连接：

```cpp
// model_instance_state.cc 核心逻辑（简化）
class ModelInstanceState {
public:
    void Initialize() {
        // 加载 TRT-LLM 引擎
        executor_ = std::make_unique<tle::Executor>(
            engine_path_,
            tle::ModelType::kDECODER_ONLY,
            executor_config_);
    }

    void EnqueueRequest(tle::Request request) {
        auto req_id = executor_->enqueueRequest(std::move(request));
        active_requests_[req_id] = /* tracking info */;
    }

    std::vector<tle::Response> PollResponses() {
        return executor_->awaitResponses(timeout_);
    }

private:
    std::unique_ptr<tle::Executor> executor_;
    std::unordered_map<uint64_t, RequestInfo> active_requests_;
};
```

## 17.4 model_config.pbtxt 配置详解

Triton 使用 Protocol Buffer Text 格式（`.pbtxt`）定义模型配置。TensorRT-LLM backend 的配置文件是整个部署的核心，以下是 `tensorrt_llm` 模型的典型配置：

```
# all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt
name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 256

model_transaction_policy {
  decoupled: true    # 支持 streaming 输出
}

input [
  { name: "input_ids",         data_type: TYPE_INT32, dims: [ -1 ] },
  { name: "input_lengths",     data_type: TYPE_INT32, dims: [ 1 ] },
  { name: "request_output_len", data_type: TYPE_INT32, dims: [ 1 ] },
  { name: "streaming",         data_type: TYPE_BOOL,  dims: [ 1 ] },
  { name: "beam_width",        data_type: TYPE_INT32, dims: [ 1 ] },
  { name: "temperature",       data_type: TYPE_FP32,  dims: [ 1 ] },
  { name: "top_k",             data_type: TYPE_INT32, dims: [ 1 ] },
  { name: "top_p",             data_type: TYPE_FP32,  dims: [ 1 ] }
]

output [
  { name: "output_ids",        data_type: TYPE_INT32, dims: [ -1, -1 ] },
  { name: "sequence_length",   data_type: TYPE_INT32, dims: [ -1 ] }
]

parameters: {
  key: "gpt_model_type"
  value: { string_value: "inflight_fused_batching" }
}
parameters: {
  key: "gpt_model_path"
  value: { string_value: "/engines/llama-7b/fp16/1-gpu" }
}
parameters: {
  key: "batch_scheduler_policy"
  value: { string_value: "max_utilization" }
}
parameters: {
  key: "kv_cache_free_gpu_mem_fraction"
  value: { string_value: "0.85" }
}
```

几个关键配置项值得特别说明：`decoupled: true` 启用了 Triton 的解耦事务模式，允许一个请求产生多个响应（这对 streaming 场景至关重要）；`batch_scheduler_policy` 设为 `max_utilization` 时会尽可能填满 batch，而 `guaranteed_no_evict` 则保证已接受的请求不会被驱逐。

## 17.5 请求路由：Ensemble 模型

LLM 推理通常包含三个阶段：文本预处理（tokenization）、模型推理和后处理（detokenization）。Triton 通过 ensemble 模型将这三个阶段串联起来：

```
# all_models/inflight_batcher_llm/ensemble/config.pbtxt
name: "ensemble"
platform: "ensemble"
max_batch_size: 256

ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: -1
      input_map { key: "query"       value: "query" }
      input_map { key: "request_output_len" value: "request_output_len" }
      output_map { key: "input_ids"     value: "_INPUT_IDS" }
      output_map { key: "input_lengths" value: "_INPUT_LENGTHS" }
    },
    {
      model_name: "tensorrt_llm"
      model_version: -1
      input_map { key: "input_ids"     value: "_INPUT_IDS" }
      input_map { key: "input_lengths" value: "_INPUT_LENGTHS" }
      output_map { key: "output_ids"   value: "_OUTPUT_IDS" }
    },
    {
      model_name: "postprocessing"
      model_version: -1
      input_map { key: "output_ids"    value: "_OUTPUT_IDS" }
      output_map { key: "text_output"  value: "text_output" }
    }
  ]
}
```

请求流经 `preprocessing`（Python backend，负责 tokenization） -> `tensorrt_llm`（C++ backend，执行引擎推理） -> `postprocessing`（Python backend，负责 detokenization）。

## 17.6 Triton Client 调用示例

使用 Triton 的 Python client 调用 TensorRT-LLM 服务非常简洁：

```python
import tritonclient.grpc as grpcclient
import numpy as np

client = grpcclient.InferenceServerClient(url="localhost:8001")

# 构造输入
query = np.array([["What is TensorRT-LLM?"]], dtype=object)
output_len = np.array([[128]], dtype=np.int32)

inputs = [
    grpcclient.InferInput("query", query.shape, "BYTES"),
    grpcclient.InferInput("request_output_len", output_len.shape, "INT32"),
]
inputs[0].set_data_from_numpy(query)
inputs[1].set_data_from_numpy(output_len)

# 同步推理
result = client.infer("ensemble", inputs)
output = result.as_numpy("text_output")
print(output[0].decode("utf-8"))
```

对于 streaming 场景，需要使用异步 gRPC 接口配合回调函数：

```python
def callback(result, error):
    if error:
        print(f"Error: {error}")
    else:
        token = result.as_numpy("text_output")[0].decode("utf-8")
        print(token, end="", flush=True)

client.start_stream(callback=callback)
client.async_stream_infer("ensemble", inputs)
```

## 17.7 健康检查与 Metrics 监控

Triton 默认在 `8000`（HTTP）、`8001`（gRPC）、`8002`（Metrics）三个端口提供服务。健康检查端点包括：

- `GET /v2/health/ready` — 服务是否就绪
- `GET /v2/health/live` — 服务是否存活
- `GET /v2/models/tensorrt_llm/ready` — 特定模型是否就绪

Metrics 端点 `GET /metrics` 返回 Prometheus 格式的指标，关键指标包括：

- `nv_inference_request_success` — 成功请求数
- `nv_inference_request_duration_us` — 请求延迟
- `nv_inference_queue_duration_us` — 队列等待时间
- `nv_gpu_utilization` — GPU 利用率

## 17.8 动态 Batching 配置

虽然 TensorRT-LLM 内部已经实现了 inflight batching，但 Triton 层的动态 batching 配置仍然影响请求进入 backend 前的聚合行为。对于 LLM 场景，通常建议关闭 Triton 层的动态 batching（因为 inflight batching 已经在 backend 层处理），或者将其配置为直通模式：

```
dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

启动 Triton 服务的典型命令如下：

```bash
python3 scripts/launch_triton_server.py \
    --model_repo all_models/inflight_batcher_llm \
    --tensorrt_llm_model_name tensorrt_llm \
    --world_size 1
```

对于多 GPU 张量并行场景，`--world_size` 参数指定使用的 GPU 数量，脚本会自动启动对应数量的 MPI 进程。

## 本章小结

本章详细介绍了 TensorRT-LLM 与 Triton Inference Server 的集成方案。`tensorrtllm_backend` 作为连接二者的桥梁，通过 C++ backend 接口将 Triton 的请求路由能力与 TensorRT-LLM 的高性能推理能力结合在一起。Ensemble 模型配置将 tokenization、推理和 detokenization 串联为完整的端到端流水线。在生产环境中，Triton 提供的健康检查、metrics 监控和多模型管理能力是不可或缺的基础设施。理解这一集成架构，是将 TensorRT-LLM 从实验环境推向生产部署的关键一步。
