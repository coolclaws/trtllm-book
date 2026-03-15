# 附录 B：build_config 参数速查

本附录整理了 `trtllm-build` 命令（对应源码 `tensorrt_llm/commands/build.py`）及 `BuildConfig`（定义于 `tensorrt_llm/builder.py`）中常用参数的完整说明。这些参数直接影响 Engine 的构建行为、推理性能和资源占用。

> **源码参考：**
> - Python 侧：`tensorrt_llm/builder.py` 中的 `BuildConfig` 类
> - CLI 入口：`tensorrt_llm/commands/build.py` 中的 `parse_arguments()`
> - Plugin 配置：`tensorrt_llm/plugin/plugin.py` 中的 `PluginConfig` 类

---

## 1. 基础构建参数

这些参数定义了 Engine 的基本能力边界。在构建后无法动态修改，需要在构建前根据业务需求合理规划。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_batch_size` | `int` | `1` | 最大并发 batch 大小。增大可提升吞吐量，但会增加显存占用。需与 Runtime 的 Batch Manager 配合调优 |
| `max_input_len` | `int` | `1024` | 单条请求的最大输入 token 数。影响 Context Phase 的计算量和内存分配 |
| `max_seq_len` | `int` | `2048` | 最大序列长度（输入 + 输出之和）。决定 KV Cache 的最大容量需求 |
| `max_beam_width` | `int` | `1` | Beam Search 的最大宽度。设为 1 表示 greedy/sampling 解码，大于 1 启用 Beam Search |
| `max_num_tokens` | `int` | `None` | In-flight Batching 模式下一次迭代处理的最大 token 总数。未设置时由 `max_batch_size * max_input_len` 推算 |
| `max_prompt_embedding_table_size` | `int` | `0` | Prompt Tuning 嵌入表的最大尺寸。不使用 Prompt Tuning 时保持 0 |
| `builder_opt` | `int` | `None` | TensorRT builder 优化级别（0~5）。级别越高构建越慢但推理越快，`None` 使用 TensorRT 默认值 |

---

## 2. Plugin 配置参数

Plugin 是 TensorRT-LLM 实现高性能推理的核心机制。大部分关键算子通过 Plugin 而非原生 TensorRT Layer 实现，以获得更好的性能。相关源码位于 `cpp/tensorrt_llm/plugins/` 目录。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `gpt_attention_plugin` | `str` | `"auto"` | GPT Attention Plugin 的数据类型（`"float16"` / `"bfloat16"` / `"float32"` / `"auto"`）。`"auto"` 自动匹配模型精度。源码实现：`cpp/tensorrt_llm/plugins/gptAttentionPlugin/` |
| `gemm_plugin` | `str` | `"auto"` | GEMM Plugin 数据类型。启用后使用 CUTLASS 优化的矩阵乘法替代 cuBLAS 默认实现，对小 batch 场景效果显著 |
| `context_fmha` | `bool` | `True` | 是否在 Context Phase 启用 Fused Multi-Head Attention。对应 Flash Attention 实现，大幅减少显存占用并提升速度 |
| `context_fmha_fp32_acc` | `bool` | `False` | Context FMHA 是否使用 FP32 累加器。启用可提升数值精度，但会略微降低性能 |
| `multi_block_mode` | `bool` | `False` | Generation Phase 中 Attention 是否使用 Multi-Block 模式。对长序列场景可提升 GPU 利用率，将单个 Attention 计算分散到多个 CUDA Block |
| `enable_xqa` | `bool` | `True` | 是否启用 XQA（Cross Query Attention）优化内核。对 GQA/MQA 架构模型有显著加速效果 |
| `paged_kv_cache` | `bool` | `True` | 是否启用 Paged KV Cache。启用后 KV Cache 按页分配，减少显存碎片化，支持更大的并发量。实现位于 `cpp/tensorrt_llm/runtime/kvCacheManager.cpp` |
| `tokens_per_block` | `int` | `64` | Paged KV Cache 中每个 block 包含的 token 数。较小的值减少浪费但增加管理开销 |
| `use_paged_context_fmha` | `bool` | `False` | Context Phase 是否也使用 Paged KV Cache 布局。启用可统一 Context/Generation 的内存布局 |
| `use_custom_all_reduce` | `bool` | `True` | 是否使用自定义 AllReduce 内核替代 NCCL。在 NVLink 互联的 GPU 间可获得更低延迟。源码：`cpp/tensorrt_llm/kernels/customAllReduceKernels/` |
| `remove_input_padding` | `bool` | `True` | 是否移除输入 padding。启用后不同长度的请求紧密排列，避免无效计算 |
| `lookup_plugin` | `str` | `None` | Embedding Lookup Plugin 数据类型。特定场景下可提升 Embedding 层性能 |
| `lora_plugin` | `str` | `None` | LoRA Plugin 数据类型。启用以支持运行时加载 LoRA adapter |

---

## 3. 量化相关参数

量化是提升 LLM 推理吞吐量的最有效手段之一。TensorRT-LLM 支持多种量化方案，各方案在精度和性能之间有不同的权衡。量化配置主要通过 `tensorrt_llm/quantization/mode.py` 中的 `QuantMode` 控制。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_fp8` | `bool` | `False` | 启用 FP8 量化（Hopper 架构及以上）。对权重和激活同时量化为 FP8 E4M3 格式，吞吐量接近翻倍 |
| `fp8_kv_cache` | `bool` | `False` | KV Cache 是否使用 FP8 存储。可大幅减少 KV Cache 显存占用，支持更大并发 |
| `use_smooth_quant` | `bool` | `False` | 启用 SmoothQuant 量化。通过平滑激活分布来降低量化误差，适用于 INT8 量化 |
| `per_channel` | `bool` | `False` | SmoothQuant 是否使用 per-channel 权重量化。配合 `use_smooth_quant` 使用，提升量化精度 |
| `per_token` | `bool` | `False` | SmoothQuant 是否使用 per-token 激活量化。配合 `use_smooth_quant` 使用 |
| `use_weight_only` | `bool` | `False` | 启用 Weight-Only 量化。仅对权重量化（INT8 或 INT4），激活保持原始精度 |
| `weight_only_precision` | `str` | `"int8"` | Weight-Only 量化精度（`"int8"` / `"int4"`）。INT4 压缩比更高但精度损失更大 |
| `int8_kv_cache` | `bool` | `False` | KV Cache 是否使用 INT8 存储。与 `fp8_kv_cache` 互斥 |
| `quant_algo` | `str` | `None` | 量化算法名称（`"W8A8_SQ_PER_CHANNEL"` / `"W4A16_AWQ"` / `"W4A16_GPTQ"` / `"FP8"` 等）。新版 API 中推荐使用此参数替代单独的布尔标志 |
| `kv_cache_quant_algo` | `str` | `None` | KV Cache 量化算法（`"INT8"` / `"FP8"`）。新版 API 中推荐使用此参数 |
| `group_size` | `int` | `128` | AWQ / GPTQ 分组量化的 group size。较小的值精度更高但性能开销更大 |
| `has_zero_point` | `bool` | `False` | 量化是否包含 zero point（非对称量化）。GPTQ 通常需要启用 |
| `pre_quant_scale` | `bool` | `False` | 是否应用 SmoothQuant 的 pre-quantization scale |
| `exclude_modules` | `list` | `None` | 排除在量化之外的模块名列表（如 `["lm_head"]`）。某些层量化后精度损失过大时可排除 |

---

## 4. 并行相关参数

多卡并行是部署大模型的必要手段。TensorRT-LLM 支持 Tensor Parallelism（TP）和 Pipeline Parallelism（PP）两种并行策略，可组合使用。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `tp_size` | `int` | `1` | Tensor Parallelism 度数。模型的 Attention 和 FFN 层按列/行切分到多张 GPU。通常设为同一节点内的 GPU 数 |
| `pp_size` | `int` | `1` | Pipeline Parallelism 度数。模型按层切分为多个 stage，每个 stage 运行在不同 GPU 上 |
| `world_size` | `int` | `1` | 总 GPU 数量，需满足 `world_size = tp_size * pp_size` |
| `gpus_per_node` | `int` | `8` | 每个节点的 GPU 数量。影响通信拓扑选择（节点内 NVLink vs 节点间 InfiniBand） |
| `auto_parallel` | `bool` | `False` | 是否启用自动并行策略搜索。由框架自动决定 TP/PP 切分方式 |
| `auto_parallel_world_size` | `int` | `1` | 自动并行搜索的目标 GPU 数量 |

---

## 5. 内存管理参数

合理的内存配置对生产环境至关重要。过小的配置会限制并发能力，过大则浪费资源或导致 OOM。

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_tokens_in_paged_kv_cache` | `int` | `None` | Paged KV Cache 可容纳的最大 token 总数。未设置时 Runtime 根据可用显存自动计算 |
| `kv_cache_free_gpu_mem_fraction` | `float` | `0.9` | KV Cache 可使用的空闲显存比例。Engine 加载后剩余显存的这个比例用于分配 KV Cache |
| `enable_chunked_context` | `bool` | `False` | 是否启用分块 Context 处理。将长输入拆分为多个 chunk 处理，避免单次 Context Phase 显存峰值过高 |
| `max_draft_len` | `int` | `0` | Speculative Decoding 中 draft model 的最大预测长度。不使用 Speculative Decoding 时保持 0 |

---

## 6. 其他参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `strongly_typed` | `bool` | `True` | 是否启用 TensorRT 强类型模式。启用后 Engine 中各层的数据类型严格匹配，有助于性能优化 |
| `logits_dtype` | `str` | `"float32"` | 输出 logits 的数据类型。通常保持 FP32 以确保 sampling 精度 |
| `gather_context_logits` | `bool` | `False` | 是否收集 Context Phase 中所有位置的 logits。通常只需最后一个位置，启用会增加显存和计算开销 |
| `gather_generation_logits` | `bool` | `False` | 是否收集 Generation Phase 每步的 logits。用于需要 log-probabilities 的场景 |
| `output_dir` | `str` | `"engine_output"` | Engine 文件输出目录路径 |
| `model_dir` | `str` | `None` | 模型 checkpoint 目录路径（`convert_checkpoint.py` 输出） |
| `max_multimodal_len` | `int` | `0` | 多模态输入的最大 token 长度。用于 Vision-Language 模型 |
| `use_fused_mlp` | `bool` | `True` | 是否融合 MLP 中的 Gate 和 Up 投影。对 SwiGLU/GeGLU 等 gated MLP 结构有效 |
| `multiple_profiles` | `bool` | `False` | 是否生成多个 TensorRT Optimization Profile。启用后可为不同 batch size / sequence length 范围生成专属优化策略 |
| `speculative_decoding_mode` | `str` | `None` | Speculative Decoding 模式（`"draft_tokens_external"` / `"medusa"` / `"eagle"` 等） |

---

## 参数组合推荐

以下是几种常见部署场景的参数组合建议：

### 场景一：单卡低延迟（适合交互式对话）

```bash
trtllm-build \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --gpt_attention_plugin auto \
    --gemm_plugin auto \
    --context_fmha enable \
    --paged_kv_cache enable \
    --remove_input_padding enable
```

### 场景二：多卡高吞吐（适合离线批处理）

```bash
trtllm-build \
    --max_batch_size 256 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --tp_size 4 \
    --gpt_attention_plugin auto \
    --gemm_plugin auto \
    --context_fmha enable \
    --paged_kv_cache enable \
    --remove_input_padding enable \
    --use_custom_all_reduce enable \
    --multi_block_mode enable
```

### 场景三：显存受限（使用量化压缩）

```bash
trtllm-build \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --gpt_attention_plugin auto \
    --gemm_plugin auto \
    --context_fmha enable \
    --paged_kv_cache enable \
    --use_weight_only \
    --weight_only_precision int4 \
    --int8_kv_cache enable
```

> **注意：** 以上参数名和默认值基于 TensorRT-LLM v0.9.x ~ v0.12.x 版本。不同版本的参数名可能有所变化，请以实际版本的 `trtllm-build --help` 输出为准。
