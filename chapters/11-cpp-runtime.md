# 第 11 章：C++ 运行时

> "任何足够复杂的 Python 程序都包含一个临时编写的、不完善的 C++ 运行时实现。" —— 改编自 Greenspun 第十定律

上一章我们分析了 TensorRT-LLM 的 Python 推理接口。然而，真正承担高性能推理重任的是底层的 C++ 运行时。本章将深入 `cpp/tensorrt_llm/runtime/` 目录，逐一剖析 `TllmRuntime`、`GptSession`、`Executor` 等核心类的设计与实现，并探讨内存管理、CUDA Stream 控制以及 C++/Python 绑定的工程细节。

## 11.1 目录结构概览

TensorRT-LLM 的 C++ 运行时代码主要组织在以下路径：

```
cpp/
├── include/tensorrt_llm/
│   ├── executor/
│   │   └── executor.h          # Executor API 声明
│   ├── runtime/
│   │   ├── tllmRuntime.h       # TllmRuntime 声明
│   │   ├── gptSession.h        # GptSession 声明
│   │   ├── bufferManager.h     # GPU 内存管理
│   │   └── cudaStream.h        # CUDA Stream 封装
│   └── batch_manager/
│       └── GptManager.h        # 批处理管理器
├── tensorrt_llm/
│   ├── runtime/
│   │   ├── tllmRuntime.cpp     # TllmRuntime 实现
│   │   ├── gptSession.cpp      # GptSession 实现
│   │   └── bufferManager.cpp   # BufferManager 实现
│   └── executor/
│       └── executor.cpp        # Executor 实现
└── pybind/
    └── bindings.cpp            # pybind11 绑定
```

这个目录结构体现了清晰的分层思想：`include/` 下放公开头文件，`tensorrt_llm/` 下放实现文件，`pybind/` 负责将 C++ 接口暴露给 Python。

## 11.2 TllmRuntime：TensorRT Engine 执行封装

`TllmRuntime` 是对 TensorRT `IRuntime` 和 `ICudaEngine` 的封装，定义在 `cpp/include/tensorrt_llm/runtime/tllmRuntime.h` 中：

```cpp
class TllmRuntime {
public:
    TllmRuntime(void const* engineData, std::size_t engineSize,
                nvinfer1::ILogger& logger);

    // 执行推理
    bool executeContext(SizeType contextIndex) const;

    // 获取 engine 的输入输出 tensor 信息
    nvinfer1::ITensor const* getBinding(char const* name,
                                         SizeType contextIndex) const;

    // 设置 tensor 地址
    void setTensorAddress(char const* name, void* data,
                          SizeType contextIndex);

private:
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> mContexts;
};
```

`TllmRuntime` 的设计要点在于它支持多个 `IExecutionContext`。在 TensorRT 中，一个 engine 可以创建多个 execution context，每个 context 维护独立的输入输出绑定。TRT-LLM 利用这一特性，通常创建两个 context——一个用于 Context Phase（处理长序列输入），另一个用于 Generation Phase（逐 token 生成）。两个 context 可以使用不同的 optimization profile，从而在不同的输入形状下都获得最优性能。

`executeContext()` 方法是对 `IExecutionContext::enqueueV3()` 的封装，它将推理任务提交到 CUDA Stream 上异步执行。

## 11.3 GptSession：推理会话管理

`GptSession` 定义在 `cpp/include/tensorrt_llm/runtime/gptSession.h` 中，是比 `TllmRuntime` 更高层的抽象，对应 Python 层的 `GenerationSession`：

```cpp
class GptSession {
public:
    GptSession(GptSessionConfig const& config,
               ModelConfig const& modelConfig,
               WorldConfig const& worldConfig,
               void const* engineData, std::size_t engineSize);

    void generateBatched(GenerationOutput& outputs,
                         GenerationInput const& inputs,
                         SamplingConfig const& samplingConfig);

private:
    std::unique_ptr<TllmRuntime> mRuntime;
    std::unique_ptr<BufferManager> mBufferManager;
    ModelConfig mModelConfig;
    WorldConfig mWorldConfig;  // 张量并行/流水线并行配置
};
```

`GptSession` 负责编排完整的推理流程：管理 KV Cache 的分配与更新、协调 Context Phase 和 Generation Phase 的切换、处理 beam search 的分支与合并。它持有一个 `TllmRuntime` 实例来执行实际的 TensorRT 推理，以及一个 `BufferManager` 来管理所有 GPU 内存。

`generateBatched()` 方法接受一个 batch 的输入，完成从预填充到生成完毕的全部流程。这个方法的内部循环与上一章 Python 层 `generate()` 的逻辑对应，但效率更高——所有操作都在 C++ 中完成，没有 Python GIL 的开销。

## 11.4 Executor API：新一代执行接口

`Executor` 是 TensorRT-LLM 推荐的最新推理接口，定义在 `cpp/include/tensorrt_llm/executor/executor.h` 中。与 `GptSession` 的同步 API 不同，`Executor` 采用异步设计，天然支持 In-flight Batching：

```cpp
namespace tensorrt_llm::executor {

class Executor {
public:
    Executor(std::filesystem::path const& modelPath,
             ModelType modelType,
             ExecutorConfig const& executorConfig);

    // 异步提交请求
    IdType enqueueRequest(Request const& request);

    // 非阻塞获取已完成的结果
    std::vector<Response> awaitResponses(
        std::optional<IdType> requestId = std::nullopt,
        std::optional<std::chrono::milliseconds> timeout = std::nullopt);

    // 取消请求
    void cancelRequest(IdType requestId);

    // 获取运行时统计信息
    std::string getLatestIterationStats();
};

} // namespace tensorrt_llm::executor
```

`Executor` 内部启动了一个独立的执行线程，持续从请求队列中取出请求、组装 batch、执行推理、分发结果。调用方通过 `enqueueRequest()` 提交请求后可以立即返回，然后通过 `awaitResponses()` 获取结果。这种生产者-消费者模型非常适合高并发的在线服务场景。

`ExecutorConfig` 允许配置关键的运行时参数：

```cpp
ExecutorConfig config;
config.setMaxBeamWidth(1);
config.setSchedulerConfig(SchedulerConfig(SchedulerPolicy::kMAX_UTILIZATION));
config.setKvCacheConfig(KvCacheConfig(/* ... */));
config.setBatchingType(BatchingType::kINFLIGHT);
```

## 11.5 BufferManager：GPU 内存管理

`BufferManager` 定义在 `cpp/include/tensorrt_llm/runtime/bufferManager.h` 中，是 TRT-LLM 统一的内存管理接口：

```cpp
class BufferManager {
public:
    using IBufferPtr = std::unique_ptr<IBuffer>;
    using ITensorPtr = std::unique_ptr<ITensor>;

    explicit BufferManager(CudaStreamPtr stream);

    // 分配 GPU 内存
    IBufferPtr gpu(std::size_t size, nvinfer1::DataType type) const;

    // 分配 CPU pinned 内存
    IBufferPtr pinned(std::size_t size, nvinfer1::DataType type) const;

    // 内存拷贝（自动判断方向）
    void copy(void const* src, IBuffer& dst, MemoryType srcType) const;

    // 获取关联的 CUDA Stream
    CudaStreamPtr getStream() const;
};
```

`BufferManager` 的一个关键设计决策是使用**内存池**来避免频繁的 `cudaMalloc` 和 `cudaFree` 调用。CUDA 的内存分配是昂贵的操作——每次 `cudaMalloc` 都可能触发设备同步，延迟通常在微秒到毫秒级别。对于需要在每个 iteration 动态分配临时 buffer 的推理场景，这种开销是不可接受的。

TRT-LLM 的内存池策略是：在初始化阶段预分配一大块 GPU 内存，运行时通过 bump allocator 或 free list 从这块预分配的内存中快速分配和回收。这种方式将内存分配的延迟从毫秒级降低到纳秒级。

## 11.6 CUDA Stream 管理

CUDA Stream 是 GPU 上任务调度的核心机制。TRT-LLM 封装了 `CudaStream` 类来管理 stream 的生命周期：

```cpp
class CudaStream {
public:
    CudaStream(unsigned int flags = cudaStreamNonBlocking,
               int priority = 0);
    ~CudaStream();

    cudaStream_t get() const { return mStream; }
    void synchronize() const;

private:
    cudaStream_t mStream;
};
```

在典型的推理流程中，TRT-LLM 使用主 stream 执行 TensorRT engine 的推理，同时可能使用辅助 stream 进行 H2D（Host-to-Device）数据传输，以实现计算与传输的重叠。stream 之间的同步通过 CUDA event 来实现，而非昂贵的全局 `cudaDeviceSynchronize()`。

## 11.7 C++/Python 绑定

TRT-LLM 使用 pybind11 将 C++ 运行时暴露给 Python。绑定代码主要在 `cpp/pybind/` 目录下：

```cpp
// 简化示意
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    // 绑定 Executor
    py::class_<executor::Executor>(m, "Executor")
        .def(py::init<std::filesystem::path const&,
                       executor::ModelType,
                       executor::ExecutorConfig const&>())
        .def("enqueue_request", &executor::Executor::enqueueRequest)
        .def("await_responses", &executor::Executor::awaitResponses,
             py::call_guard<py::gil_scoped_release>())
        .def("cancel_request", &executor::Executor::cancelRequest);

    // 绑定 ExecutorConfig
    py::class_<executor::ExecutorConfig>(m, "ExecutorConfig")
        .def(py::init<>())
        .def("set_max_beam_width", &executor::ExecutorConfig::setMaxBeamWidth);
}
```

值得注意的是 `py::call_guard<py::gil_scoped_release>()` 的使用。`awaitResponses()` 可能会阻塞等待结果，如果不释放 GIL，整个 Python 进程都会被阻塞。通过显式释放 GIL，其他 Python 线程可以继续执行——这对于在 Python 服务框架中集成 TRT-LLM 非常重要。

## 11.8 Runtime 生命周期

理解 C++ 运行时的初始化与销毁顺序对于避免资源泄漏和崩溃至关重要：

**初始化阶段**：创建 CUDA Stream → 创建 BufferManager → 反序列化 TensorRT Engine（创建 TllmRuntime）→ 预分配 KV Cache 和工作 buffer → 初始化 Executor 执行线程。

**运行阶段**：接收请求 → 调度组 batch → Context Phase → Generation Phase（循环）→ 返回结果 → 回收临时 buffer。

**销毁阶段**：停止 Executor 执行线程 → 等待所有 in-flight 请求完成 → 释放 KV Cache → 释放 GPU buffer → 销毁 TensorRT Engine → 销毁 CUDA Stream。

销毁顺序必须与初始化顺序严格相反。特别是 CUDA Stream 必须在所有使用它的 buffer 和 engine 之后销毁，否则会触发 CUDA 错误。TRT-LLM 通过 RAII（Resource Acquisition Is Initialization）模式和 `std::unique_ptr` 的析构顺序来保证这一点——成员变量的销毁顺序与声明顺序相反，因此只要声明顺序正确，资源就能安全释放。

## 本章小结

本章深入分析了 TensorRT-LLM 的 C++ 运行时架构。`TllmRuntime` 是最底层的 TensorRT engine 执行封装，支持多个 execution context 以适应不同推理阶段的需求。`GptSession` 在此之上编排完整的推理流程，而 `Executor` API 则提供了面向生产环境的异步接口，天然支持 In-flight Batching。`BufferManager` 通过内存池技术将 GPU 内存分配的开销降到最低，CUDA Stream 的精细管理实现了计算与数据传输的重叠。pybind11 绑定层通过合理的 GIL 管理，使得 Python 调用方能够高效地使用 C++ 运行时。理解这些 C++ 组件的设计，不仅有助于排查性能问题，也为定制和扩展 TRT-LLM 奠定了基础。
