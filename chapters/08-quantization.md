# 第 8 章：量化编译

> "量化是 LLM 推理优化中投入产出比最高的技术手段。用 2 倍的模型压缩换取不到 1% 的精度损失，这笔账怎么算都划算。"

大语言模型的参数量动辄数十亿甚至数千亿，以 FP16 精度存储和计算需要海量的显存与带宽。量化技术通过降低权重和激活值的数值精度，在几乎不影响模型质量的前提下，显著减少显存占用、提升推理吞吐量。本章将深入分析 TensorRT-LLM 所支持的各种量化方案及其编译实现。

## 8.1 为什么量化对 LLM 推理至关重要

LLM 推理的 decode 阶段（自回归生成）是典型的 **memory-bound** 工作负载。每生成一个 token，模型需要从显存中读取全部参数，但每个参数只参与一次乘加运算。以 LLaMA-70B 为例：

- FP16 权重：70B × 2 bytes = 140 GB，需要至少 2 张 A100-80G
- INT4 权重：70B × 0.5 bytes = 35 GB，单张 A100-80G 即可容纳

量化不仅减少了显存占用，更关键的是减少了每次推理需要从 HBM 读取的数据量。在 decode 阶段，推理速度几乎正比于显存带宽利用率，因此 4 倍的数据压缩可以带来接近 4 倍的速度提升。

## 8.2 FP8 量化：Hopper 架构的原生支持

FP8（8-bit 浮点数）是 NVIDIA Hopper 架构（H100/H200）引入的原生数据类型，包含两种格式：

- **E4M3**（4 位指数 + 3 位尾数）：动态范围较大，适合权重和激活值
- **E5M2**（5 位指数 + 2 位尾数）：动态范围更大但精度稍低，通常用于梯度

TensorRT-LLM 的 FP8 量化实现位于 `tensorrt_llm/quantization/` 目录下。FP8 的优势在于：

1. **硬件原生支持**：H100 的 Tensor Core 直接支持 FP8 矩阵乘法，无需额外的量化/反量化开销
2. **精度损失极小**：FP8 E4M3 保留了浮点数的指数结构，对异常值的容忍度远高于 INT8
3. **workflow 简单**：通常只需少量校准数据即可确定 scaling factor

```python
# FP8 量化的典型工作流
from tensorrt_llm.quantization import quantize_model

# 使用 NVIDIA modelopt 进行 FP8 校准
import modelopt.torch.quantization as mtq

model = load_pretrained_model("llama-70b")
mtq.quantize(model, quant_cfg=mtq.FP8_DEFAULT_CFG, forward_loop=calibration_loop)

# 导出量化后的 checkpoint
model.save_checkpoint("llama-70b-fp8-checkpoint")
```

FP8 量化后的模型在 `trtllm-build` 编译时会自动选择 FP8 GEMM kernel，无需额外的 plugin 配置。

## 8.3 INT8 SmoothQuant

SmoothQuant 是一种 INT8 量化方法，专门针对 LLM 中激活值分布不均匀的问题。传统的 per-tensor INT8 量化在遇到异常值（outlier）时会严重损失精度，而 SmoothQuant 的核心思想是在量化前对激活值进行"平滑"处理。

其数学原理是将激活值的量化难度"转移"到权重上：

```python
# SmoothQuant 的核心数学变换
# 原始计算: Y = X @ W
# 引入对角矩阵 s: Y = (X * diag(s)^{-1}) @ (diag(s) * W)
# X_smooth = X / s  (激活值变得更平滑)
# W_smooth = s * W  (权重吸收了缩放因子)
```

缩放因子 `s` 的选择是 SmoothQuant 的关键。TensorRT-LLM 使用以下公式：

```python
# alpha 控制平滑程度，通常取 0.5
s = max(|X|)^alpha / max(|W|)^(1-alpha)
```

在 TensorRT-LLM 中，SmoothQuant 的编译流程如下：

1. 使用校准数据集（calibration dataset）运行模型，收集激活值统计信息
2. 计算平滑因子 `s` 并应用到权重上
3. 导出包含量化参数的 checkpoint
4. `trtllm-build` 编译时启用 `smooth_quant_gemm_plugin`

```python
trtllm-build \
    --checkpoint_dir ./llama-7b-sq-checkpoint \
    --output_dir ./llama-7b-sq-engine \
    --smoothquant 0.5
```

## 8.4 INT4 AWQ 与 GPTQ 权重量化

INT4 量化将每个权重压缩到 4 bit，实现 4 倍的压缩比。TensorRT-LLM 支持两种主流的 INT4 量化方法：

### AWQ（Activation-aware Weight Quantization）

AWQ 的核心洞察是：并非所有权重通道对模型输出的影响相同。那些对应于大激活值的权重通道更为重要，应该在量化时给予特殊保护。

```python
# AWQ 量化流程概述
# 1. 使用校准数据收集激活值统计
# 2. 识别"显著"权重通道（对应大激活值的通道）
# 3. 对显著通道使用更高精度的量化参数
# 4. 导出 INT4 量化 checkpoint
```

AWQ 使用 group-wise 量化，即每 `group_size`（通常 128）个权重共享一组量化参数（scale 和 zero-point）。这比 per-tensor 量化精度高很多，额外的参数存储开销也可以接受。

### GPTQ

GPTQ 是基于逆 Hessian 信息的逐列量化方法。它依次量化权重矩阵的每一列，并利用 Hessian 信息补偿量化引入的误差：

```python
# GPTQ 的核心思想（简化）
for col in range(weight.shape[1]):
    # 量化当前列
    w_q = quantize(weight[:, col])
    # 计算量化误差
    error = weight[:, col] - dequantize(w_q)
    # 利用 Hessian 逆将误差分散到未量化的列
    weight[:, col+1:] -= error @ H_inv[col, col+1:] / H_inv[col, col]
```

GPTQ 的量化精度通常优于简单的 RTN（Round-to-Nearest），但校准过程更耗时。

### 编译 INT4 量化模型

```python
# 使用 AWQ 量化的 checkpoint 进行编译
trtllm-build \
    --checkpoint_dir ./llama-7b-awq-checkpoint \
    --output_dir ./llama-7b-awq-engine \
    --max_batch_size 8
```

INT4 量化模型在编译时会自动使用 `weight_only_quant_matmul_plugin`。该 plugin 在 GEMM kernel 内部将 INT4 权重反量化为 FP16，然后与 FP16 激活值进行矩阵乘法。

## 8.5 量化工作流：NVIDIA Modelopt

TensorRT-LLM 推荐使用 NVIDIA Modelopt（原 AMMO）工具进行量化校准。Modelopt 提供了统一的 API 来执行各种量化方法：

```python
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 选择量化配置
quant_cfg = mtq.INT4_AWQ_CFG  # 或 FP8_DEFAULT_CFG, W8A8_SQ_CFG 等

# 校准
def calibration_loop(model):
    for batch in calibration_dataloader:
        model(batch)

mtq.quantize(model, quant_cfg, forward_loop=calibration_loop)

# 导出为 TensorRT-LLM checkpoint 格式
export_tensorrt_llm_checkpoint(
    model,
    decoder_type="llama",
    dtype=torch.float16,
    export_dir="./llama-7b-int4-awq-checkpoint",
)
```

Modelopt 在校准过程中会统计激活值的分布（min/max 或 percentile），计算最优的量化参数，并将这些参数嵌入导出的 checkpoint 中。

## 8.6 QuantMode 类

`QuantMode` 定义在 `tensorrt_llm/quantization/mode.py` 中，是一个使用位掩码（bitmask）实现的量化模式标志类：

```python
# tensorrt_llm/quantization/mode.py
class QuantMode:
    # 各个量化选项对应不同的 bit 位
    NONE = 0
    INT4_WEIGHTS = 1 << 0
    INT8_WEIGHTS = 1 << 1
    ACTIVATIONS = 1 << 2       # 激活值也量化
    PER_CHANNEL = 1 << 3       # per-channel 权重量化
    PER_TOKEN = 1 << 4         # per-token 激活量化
    PER_GROUP = 1 << 5         # group-wise 量化
    FP8_QDQ = 1 << 6           # FP8 quantize-dequantize
    FP8_KV_CACHE = 1 << 7     # KV Cache FP8 量化
```

`QuantMode` 的位掩码设计使得多种量化选项可以灵活组合。例如，SmoothQuant W8A8 对应 `INT8_WEIGHTS | ACTIVATIONS | PER_CHANNEL | PER_TOKEN`，而 AWQ INT4 对应 `INT4_WEIGHTS | PER_GROUP`。

模型的各层会在编译时查询 `QuantMode` 来决定使用哪种量化 kernel：

```python
# 在模型层中根据 QuantMode 选择实现
def forward(self, x):
    if self.quant_mode.has_fp8_qdq():
        return self.fp8_gemm(x, self.weight, self.scale)
    elif self.quant_mode.is_int4_weight_only():
        return self.int4_weight_only_gemm(x, self.weight, self.scale, self.zeros)
    else:
        return self.default_gemm(x, self.weight)
```

## 8.7 不同量化方法的精度与性能权衡

各量化方法的特性对比如下：

| 方法 | 精度影响 | 压缩比 | 硬件要求 | 校准复杂度 |
|------|---------|--------|---------|-----------|
| FP8 E4M3 | 极小 | 2x | Hopper+ | 低 |
| INT8 SmoothQuant | 小 | 2x | Ampere+ | 中 |
| INT4 AWQ | 中等 | 4x | Ampere+ | 中 |
| INT4 GPTQ | 中等 | 4x | Ampere+ | 高 |

**选择建议**：

- 如果使用 H100/H200，**FP8** 是首选，精度损失最小，且有硬件原生加速
- 如果使用 A100 且显存充足，**INT8 SmoothQuant** 是平衡精度和性能的好选择
- 如果需要单卡部署大模型，**INT4 AWQ** 提供最高的压缩比
- **GPTQ** 在学术界更流行，但 AWQ 在工程实践中通常更方便

## 8.8 量化模型的 Checkpoint 格式

量化 checkpoint 与标准 checkpoint 的主要区别在于 `config.json` 中包含了量化配置信息：

```python
{
    "architecture": "LlamaForCausalLM",
    "dtype": "float16",
    "quantization": {
        "quant_algo": "W4A16_AWQ",    # 量化算法
        "group_size": 128,             # group-wise 量化的组大小
        "has_zero_point": false,       # 是否有零点
        "pre_quant_scale": true,       # 是否有预量化缩放
        "exclude_modules": ["lm_head"] # 不量化的模块
    },
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    ...
}
```

权重文件（`.safetensors`）中，量化权重以压缩格式存储。以 INT4 AWQ 为例，每 8 个 INT4 权重被 pack 到一个 INT32 中，同时附带 `scale` 和可选的 `zero_point` 参数。

`trtllm-build` 在读取 checkpoint 时会解析量化配置，自动设置 `QuantMode` 并选择对应的 plugin。整个流程对用户透明——只需提供正确的量化 checkpoint，编译器会自动完成剩余工作。

## 本章小结

本章全面分析了 TensorRT-LLM 的量化编译体系。从 FP8 到 INT8 SmoothQuant，再到 INT4 AWQ/GPTQ，每种量化方法都有其适用场景和权衡取舍。量化工作流以 NVIDIA Modelopt 为校准工具，以 `QuantMode` 位掩码为内部表示，以量化 plugin 为执行引擎，形成了完整的端到端链路。理解量化不仅是性能优化的需要，更是大模型工程化落地的必备技能——在实际部署中，量化方案的选择往往直接决定了服务的成本效益比。
