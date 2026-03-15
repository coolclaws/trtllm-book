# 第 19 章：性能基准测试

> "没有度量就没有优化——你无法改进一个你没有测量的系统。" —— Peter Drucker（管理学意义上）

性能是 TensorRT-LLM 存在的核心理由。但 "性能好" 这三个字背后隐藏着无数细节：什么场景下好？好多少？用什么指标衡量？本章将系统性地介绍 TensorRT-LLM 的基准测试方法论、工具链和调优策略，帮助读者建立科学的性能评估体系。

## 19.1 benchmarks/ 目录结构

TensorRT-LLM 仓库中的基准测试工具位于 `benchmarks/` 目录下：

```
benchmarks/
├── python/
│   ├── benchmark.py            # 主入口脚本
│   ├── gpt_benchmark.py        # GPT 系列模型测试
│   ├── bert_benchmark.py       # BERT 系列模型测试
│   ├── enc_dec_benchmark.py    # 编码器-解码器模型测试
│   └── utils/
│       ├── benchmark_profiler.py
│       └── benchmark_utils.py
├── cpp/
│   ├── gptManagerBenchmark.cpp  # C++ 级别的高精度测试
│   └── CMakeLists.txt
└── suite/
    └── tensorrt_llm_bench.py   # 综合测试套件
```

核心的 Python benchmark 脚本 `benchmarks/python/benchmark.py` 提供了一个统一的命令行接口，支持测试不同模型在各种配置下的性能表现。

## 19.2 关键性能指标

在评估 LLM 推理性能时，有几个核心指标必须理解清楚：

**吞吐量指标：**
- **Throughput（tokens/s）**：每秒生成的 token 数，是最直观的性能指标
- **Requests/s**：每秒完成的请求数，适合固定输出长度的场景

**延迟指标：**
- **Time To First Token（TTFT）**：从请求发出到收到第一个 token 的时间，直接影响用户感知
- **Inter-Token Latency（ITL）**：相邻两个 token 之间的生成间隔
- **End-to-End Latency**：完成整个请求的总时间
- **P50/P90/P99 延迟**：不同百分位的延迟值，P99 特别重要因为它代表最差情况

这些指标之间存在根本性的 trade-off：增大 batch size 可以提升吞吐量，但会增加单请求延迟。

## 19.3 benchmark 脚本使用方法

使用 Python benchmark 脚本的典型命令如下：

```bash
# 基本用法：测试已构建的引擎
python benchmarks/python/benchmark.py \
    --engine_dir /engines/llama-7b/fp16/1-gpu \
    --tokenizer_dir meta-llama/Llama-2-7b-hf \
    --dataset_path benchmarks/data/ShareGPT.json \
    --num_requests 1000 \
    --concurrency 64 \
    --output_csv results.csv
```

对于更高级的测试，C++ benchmark 工具 `gptManagerBenchmark` 提供了更精确的性能数据，因为它绕过了 Python 层的开销：

```bash
# C++ benchmark（需要先编译）
./benchmarks/cpp/gptManagerBenchmark \
    --engine_dir /engines/llama-7b/fp16/1-gpu \
    --type IFB \
    --dataset benchmarks/data/ShareGPT.json \
    --num_requests 2000 \
    --max_batch_size 256 \
    --kv_cache_free_gpu_mem_fraction 0.85
```

`tensorrt_llm_bench` 是更新的综合测试工具，集成了端到端的测试流程：

```bash
# 综合测试套件
python benchmarks/suite/tensorrt_llm_bench.py \
    --model llama-7b \
    --tp_size 1 \
    --dtype float16 \
    --task throughput \
    --dataset ShareGPT \
    --num_requests 500
```

## 19.4 不同 Batch Size 下的性能曲线

Batch size 是影响 LLM 推理性能最重要的参数之一。以 LLaMA-2-7B FP16 在 A100-80GB 上的典型数据为例：

```
Batch Size | Throughput (tokens/s) | TTFT (ms) | ITL (ms)
-----------|-----------------------|-----------|----------
    1      |        45             |    12     |   22
    4      |       170             |    14     |   23
   16      |       620             |    18     |   26
   64      |      2100             |    35     |   30
  128      |      3500             |    65     |   37
  256      |      4800             |   120     |   53
```

可以观察到几个规律：吞吐量随 batch size 近似线性增长（在未饱和时），但 TTFT 和 ITL 也会逐渐增加。在 batch size 超过某个临界点后，GPU 显存中的 KV cache 空间不足，性能会急剧下降甚至触发 OOM。

通过 Python 脚本自动化跑出完整曲线：

```python
import subprocess
import json

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
results = []

for bs in batch_sizes:
    cmd = [
        "python", "benchmarks/python/benchmark.py",
        "--engine_dir", "/engines/llama-7b/fp16/1-gpu",
        "--max_batch_size", str(bs),
        "--num_requests", "500",
        "--output_json", f"result_bs{bs}.json",
    ]
    subprocess.run(cmd, check=True)

    with open(f"result_bs{bs}.json") as f:
        data = json.load(f)
        results.append({
            "batch_size": bs,
            "throughput": data["throughput_tokens_per_sec"],
            "ttft_p50": data["ttft_p50_ms"],
            "itl_p50": data["itl_p50_ms"],
        })

# 可视化结果
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot([r["batch_size"] for r in results],
         [r["throughput"] for r in results], "o-")
ax1.set_xlabel("Batch Size")
ax1.set_ylabel("Throughput (tokens/s)")
ax1.set_title("Throughput vs Batch Size")

ax2.plot([r["batch_size"] for r in results],
         [r["ttft_p50"] for r in results], "o-", label="TTFT P50")
ax2.plot([r["batch_size"] for r in results],
         [r["itl_p50"] for r in results], "s-", label="ITL P50")
ax2.set_xlabel("Batch Size")
ax2.set_ylabel("Latency (ms)")
ax2.legend()
ax2.set_title("Latency vs Batch Size")

plt.tight_layout()
plt.savefig("batch_size_analysis.png", dpi=150)
```

## 19.5 不同量化精度的性能对比

量化是 TensorRT-LLM 最强大的性能优化手段之一。以下是 LLaMA-2-7B 在 A100-80GB 上不同精度的典型对比：

```
精度       | 模型大小 | Throughput | TTFT P50 | 质量损失
-----------|---------|------------|----------|--------
FP16       | 13.5 GB |  3500 t/s  |  65 ms   | 基准
FP8 (E4M3) |  6.8 GB |  5800 t/s  |  42 ms   | 极小
INT8 (W8A8)|  6.8 GB |  5200 t/s  |  48 ms   | 小
INT4 (AWQ) |  3.5 GB |  6500 t/s  |  35 ms   | 可接受
INT4 (GPTQ)|  3.5 GB |  6200 t/s  |  38 ms   | 可接受
```

FP8 是 Hopper 架构（H100）的杀手级特性，它在几乎不损失模型质量的情况下将吞吐量提升约 65%。INT4 量化则进一步降低了显存占用，使得更大的 batch size 成为可能。

构建不同精度引擎的命令对比：

```bash
# FP16 引擎
trtllm-build --model_dir llama-7b-hf \
    --dtype float16 \
    --output_dir engines/fp16

# FP8 引擎（需要先校准）
trtllm-build --model_dir llama-7b-hf \
    --dtype float16 \
    --quantization fp8 \
    --output_dir engines/fp8

# INT4 AWQ 引擎
trtllm-build --model_dir llama-7b-hf-awq \
    --dtype float16 \
    --quantization int4_awq \
    --output_dir engines/int4_awq
```

## 19.6 与 vLLM 的性能对比解读

TensorRT-LLM 与 vLLM 的性能对比是社区中最受关注的话题之一。需要注意的是，公平的对比需要控制以下变量：

- **相同硬件**：GPU 型号、数量、互联拓扑
- **相同模型**：模型权重、精度、参数量
- **相同负载**：请求分布、输入/输出长度分布
- **相同调度策略**：连续批处理、相似的 KV cache 配置

在控制了这些变量后，一般结论是：

1. **低并发（batch size < 8）**：两者差异不大，TRT-LLM 有 10-20% 优势
2. **中并发（batch size 16-64）**：TRT-LLM 优势明显，吞吐量高 30-50%
3. **高并发（batch size > 128）**：TRT-LLM 优势最大，可达 50-80%
4. **FP8 精度**：TRT-LLM 明显领先（vLLM 的 FP8 支持较晚）

但这些数据会随版本迭代快速变化。vLLM 社区的迭代速度极快，差距在持续缩小。

## 19.7 Profiling 工具

当 benchmark 数据不符合预期时，需要深入 profiling 找到瓶颈。NVIDIA 提供了几个关键工具：

**Nsight Systems（nsys）**——系统级 timeline 分析：

```bash
# 收集 profiling 数据
nsys profile -o profile_output \
    --trace=cuda,nvtx,osrt \
    python benchmarks/python/benchmark.py \
    --engine_dir /engines/llama-7b/fp16/1-gpu \
    --num_requests 100

# 使用 Nsight Systems GUI 打开分析
# nsys-ui profile_output.nsys-rep
```

Nsight Systems 的 timeline 视图可以清晰地看到 CPU 与 GPU 之间的交互、kernel 执行时间、内存拷贝耗时以及调度间隙。

**Nsight Compute（ncu）**——kernel 级深度分析：

```bash
# 分析单个 kernel 的性能细节
ncu --set full \
    --target-processes all \
    -o kernel_analysis \
    python run_single_iteration.py
```

Nsight Compute 可以给出每个 CUDA kernel 的 occupancy、memory bandwidth utilization、compute throughput 等详细指标，帮助判断是 compute bound 还是 memory bound。

## 19.8 性能调优建议

基于大量实践经验，以下是 TensorRT-LLM 性能调优的核心建议：

**KV Cache 配置：**
```python
# 建议将 85-90% 的空闲显存分配给 KV cache
executor_config = trtllm.ExecutorConfig(
    kv_cache_config=trtllm.KvCacheConfig(
        free_gpu_memory_fraction=0.85,
        enable_block_reuse=True,  # 启用 KV cache 复用
    )
)
```

**Plugin 选择：**
- 使用 `gpt_attention_plugin`（默认启用）可获得 20-30% 的 attention 加速
- `gemm_plugin` 在特定矩阵尺寸下比 cuBLAS 更快
- Paged KV cache 对高并发场景至关重要

**硬件配置建议：**

| 模型规模 | 推荐 GPU | 张量并行 | 预期吞吐 |
|---------|---------|---------|---------|
| 7B      | 1x A100-80G | TP=1 | 3000-5000 t/s |
| 13B     | 1x A100-80G | TP=1 | 1800-3000 t/s |
| 34B     | 2x A100-80G | TP=2 | 1200-2000 t/s |
| 70B     | 4x A100-80G | TP=4 | 800-1500 t/s |
| 7B      | 1x H100-80G | TP=1 | 5000-8000 t/s |
| 70B     | 4x H100-80G | TP=4 | 1500-2800 t/s |
| 7B      | 1x L40S-48G | TP=1 | 2000-3500 t/s |

H100 相比 A100 的优势主要来自三个方面：更高的 HBM 带宽（3.35 TB/s vs 2.0 TB/s）、FP8 Tensor Core 支持以及更大的 L2 cache。对于注重性价比的场景，L40S 是一个值得考虑的选项——它没有 NVLink，不适合多卡张量并行，但单卡性能不错且价格较低。

## 19.9 自动化测试框架

建议在 CI/CD 中集成性能回归测试，避免代码变更引入性能退化：

```python
# performance_regression_test.py
import json
import sys

BASELINE_FILE = "baseline_results.json"
THRESHOLD = 0.05  # 允许 5% 的波动

def check_regression(current_results, baseline):
    regressions = []
    for metric, value in current_results.items():
        base = baseline[metric]
        if metric.startswith("throughput"):
            # 吞吐量不应显著下降
            if value < base * (1 - THRESHOLD):
                regressions.append(
                    f"{metric}: {value:.1f} < baseline {base:.1f}"
                )
        elif metric.startswith("latency"):
            # 延迟不应显著增加
            if value > base * (1 + THRESHOLD):
                regressions.append(
                    f"{metric}: {value:.1f} > baseline {base:.1f}"
                )
    return regressions

with open("current_results.json") as f:
    current = json.load(f)
with open(BASELINE_FILE) as f:
    baseline = json.load(f)

regressions = check_regression(current, baseline)
if regressions:
    print("Performance regressions detected:")
    for r in regressions:
        print(f"  - {r}")
    sys.exit(1)
else:
    print("No performance regressions detected.")
```

## 本章小结

本章系统地介绍了 TensorRT-LLM 的性能基准测试方法论。从核心指标（throughput、TTFT、ITL、P99 latency）的定义出发，我们深入讨论了 benchmark 工具的使用方法、不同 batch size 和量化精度下的性能特征，以及与 vLLM 的横向对比。Nsight Systems 和 Nsight Compute 是定位性能瓶颈的利器。性能优化没有银弹，关键在于理解工作负载特征，选择合适的量化精度、batch size 和 KV cache 配置，并通过持续的测量和迭代来逼近最优状态。
