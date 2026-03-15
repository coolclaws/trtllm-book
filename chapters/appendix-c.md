# 附录 C：名词解释

本附录汇总了全书涉及的核心术语，按英文字母顺序排列，供读者查阅。

---

| 术语 | 全称 / 英文 | 解释 |
|------|-------------|------|
| **ALiBi** | Attention with Linear Biases | 一种位置编码方案，通过在 Attention Score 上加线性偏置来编码位置信息，无需可学习的位置嵌入。由 BLOOM 等模型采用。参见 `tensorrt_llm/functional.py` 中的相关实现 |
| **AllReduce** | All-Reduce | 分布式通信原语，将所有参与者的数据归约（如求和）后广播到每个参与者。在 Tensor Parallelism 中用于汇总各 GPU 的部分计算结果。TensorRT-LLM 提供自定义实现：`cpp/tensorrt_llm/kernels/customAllReduceKernels/` |
| **AWQ** | Activation-aware Weight Quantization | 一种 INT4 权重量化方法，根据激活值的分布来选择量化参数，在极低比特下保持较好精度。对应 `tensorrt_llm/quantization/` 中的 AWQ 相关代码 |
| **Beam Search** | Beam Search | 一种序列解码策略，每步保留得分最高的 k 个候选序列（k 即 beam width），最终选择总得分最高的序列作为输出。通过 `max_beam_width` 参数配置 |
| **BPE** | Byte Pair Encoding | 一种子词分词算法，通过反复合并出现频率最高的字节对来构建词表。GPT、LLaMA 等主流模型均使用 BPE 或其变体 |
| **Continuous Batching** | Continuous Batching | 一种动态 batching 策略，在每个解码步骤中动态地加入新请求或移除已完成的请求，最大化 GPU 利用率。与 In-flight Batching 含义相同 |
| **Context Phase** | Context Phase / Prefill Phase | LLM 推理的第一阶段，处理用户输入的全部 prompt token。该阶段是计算密集型（compute-bound），可高度并行化。也称为 Prefill 阶段 |
| **CUDA Kernel** | CUDA Kernel | 在 NVIDIA GPU 上执行的并行计算函数。TensorRT-LLM 中的高性能 Kernel 实现位于 `cpp/tensorrt_llm/kernels/` 目录下 |
| **Decode** | Decode / Generation | 见 Generation Phase |
| **Engine** | TensorRT Engine | TensorRT 编译优化后的推理引擎文件（`.engine`）。包含优化后的计算图、内核选择方案和内存规划，可直接加载到 GPU 执行推理 |
| **Flash Attention** | Flash Attention | 一种高效的 Attention 计算算法，通过 tiling 和 recomputation 技术避免存储完整的 Attention 矩阵，将显存复杂度从 O(n^2) 降至 O(n)。TensorRT-LLM 在 Context Phase 中通过 `context_fmha` 参数启用 |
| **FP8** | 8-bit Floating Point | 8 位浮点格式（E4M3 / E5M2），在 NVIDIA Hopper 架构（H100）及以上硬件原生支持。相比 FP16/BF16 可将吞吐量提升约一倍，精度损失较小 |
| **Generation Phase** | Generation Phase / Decode Phase | LLM 推理的第二阶段，逐个生成输出 token。每步仅处理一个新 token，是访存密集型（memory-bound）。也称为 Decode 阶段 |
| **GQA** | Grouped Query Attention | 一种 Attention 变体，将 Query Head 分组，每组共享一组 Key/Value Head。介于 MHA 和 MQA 之间，在精度和效率间取得平衡。LLaMA 2（70B）等模型采用此架构 |
| **GPTQ** | GPTQ | 一种基于逐层二阶信息的训练后权重量化方法，支持 INT4/INT3/INT2 精度。量化后的模型可通过 TensorRT-LLM 的 Weight-Only 量化路径加载 |
| **In-flight Batching** | In-flight Batching | 与 Continuous Batching 同义。TensorRT-LLM 文档和源码中多使用此术语。核心实现位于 `cpp/tensorrt_llm/batch_manager/` |
| **INT4** | 4-bit Integer | 4 位整数量化格式，将模型权重压缩为 4 位表示。通常配合分组量化（group-wise）使用，常见于 AWQ 和 GPTQ 方案 |
| **INT8** | 8-bit Integer | 8 位整数量化格式。可用于 Weight-Only 量化或 Weight + Activation 同时量化（如 SmoothQuant） |
| **KV Cache** | Key-Value Cache | 在自回归生成过程中缓存已计算的 Key 和 Value 张量，避免重复计算。KV Cache 的显存占用与序列长度和 batch 大小成正比，是 LLM 推理的主要显存消耗来源 |
| **MHA** | Multi-Head Attention | 标准的多头注意力机制，每个 Query Head 对应独立的 Key Head 和 Value Head。GPT-2、BERT 等早期 Transformer 模型使用此架构 |
| **MQA** | Multi-Query Attention | 一种 Attention 变体，所有 Query Head 共享同一组 Key/Value Head。大幅减少 KV Cache 显存占用，但可能影响模型质量。Falcon 等模型采用此架构 |
| **NCCL** | NVIDIA Collective Communications Library | NVIDIA 提供的多 GPU / 多节点集合通信库，实现 AllReduce、AllGather 等通信原语。TensorRT-LLM 在某些场景下使用自定义 AllReduce 替代 NCCL 以获得更低延迟 |
| **NVLink** | NVLink | NVIDIA GPU 之间的高带宽互联技术（最高可达 900 GB/s per GPU，H100 SXM）。Tensor Parallelism 的通信性能高度依赖 NVLink 带宽 |
| **ONNX** | Open Neural Network Exchange | 开放的神经网络模型交换格式。TensorRT-LLM 的部分模型转换流程会经过 ONNX 中间表示 |
| **Paged KV Cache** | Paged KV Cache | 受操作系统虚拟内存分页机制启发的 KV Cache 管理方式。将 KV Cache 划分为固定大小的 block（page），按需分配和回收，大幅减少显存碎片化。源码：`cpp/tensorrt_llm/runtime/kvCacheManager.cpp` |
| **Pipeline Parallelism** | Pipeline Parallelism (PP) | 一种模型并行策略，将模型按层划分为多个 stage，每个 stage 运行在不同的 GPU 上。适合跨节点部署超大模型，但引入 pipeline bubble 开销 |
| **Plugin** | TensorRT Plugin | TensorRT 的扩展机制，允许用户注册自定义算子实现。TensorRT-LLM 大量使用 Plugin 实现高性能 LLM 算子（Attention、GEMM 等）。Plugin 注册入口：`cpp/tensorrt_llm/plugins/api/tllmPlugin.cpp` |
| **Prefill** | Prefill | 见 Context Phase |
| **RoPE** | Rotary Position Embedding | 旋转位置编码，通过对 Query 和 Key 向量施加旋转变换来编码位置信息。具有良好的外推性，被 LLaMA、Qwen、Mistral 等主流模型广泛采用 |
| **SmoothQuant** | SmoothQuant | 一种 INT8 量化方法，核心思路是将激活中的量化难度通过数学等价变换「平滑」地转移到权重上，使得激活和权重都更容易量化。对应 `tensorrt_llm/quantization/` 中的相关实现 |
| **Speculative Decoding** | Speculative Decoding | 一种加速自回归生成的技术。使用小型 draft model 快速生成多个候选 token，再由大模型并行验证。在不改变输出分布的前提下提升解码速度 |
| **Tensor Parallelism** | Tensor Parallelism (TP) | 一种模型并行策略，将模型中的矩阵运算按行或列切分到多张 GPU 上并行计算，通过 AllReduce 通信同步结果。适合同一节点内的多 GPU 部署 |
| **TensorRT** | TensorRT | NVIDIA 推出的高性能深度学习推理优化器和运行时。通过算子融合、内核自动调优、精度校准等技术最大化 GPU 推理性能 |
| **TensorRT-LLM** | TensorRT-LLM | NVIDIA 基于 TensorRT 开发的 LLM 推理加速框架。提供模型定义 DSL、高性能 CUDA Kernel、分布式推理运行时等完整工具链。开源仓库：`github.com/NVIDIA/TensorRT-LLM` |
| **Tokenizer** | Tokenizer（分词器） | 将原始文本转换为模型可处理的 token ID 序列的组件。TensorRT-LLM 本身不包含 Tokenizer，通常使用 HuggingFace Transformers 的 Tokenizer |
| **Top-K** | Top-K Sampling | 一种随机采样策略，在每步仅从概率最高的前 K 个 token 中采样。K 越小输出越确定性，K 越大输出越多样 |
| **Top-P** | Top-P (Nucleus) Sampling | 一种随机采样策略，从累积概率达到阈值 P 的最小 token 集合中采样。也称为 Nucleus Sampling，比 Top-K 更自适应地调整候选集大小 |
| **Triton Inference Server** | Triton Inference Server | NVIDIA 开源的模型推理服务框架，支持多种推理后端（TensorRT、PyTorch、ONNX Runtime 等）。TensorRT-LLM 通过 `tensorrtllm_backend` 与 Triton 集成，提供 HTTP/gRPC 推理服务 |
| **Weight-Only Quantization** | Weight-Only Quantization | 仅对模型权重进行量化（INT8 或 INT4），激活保持 FP16/BF16 精度。推理时权重反量化后与激活做矩阵乘法，主要优势是减少权重加载的显存带宽需求 |
| **XQA** | XQA Kernel | TensorRT-LLM 针对 GQA/MQA 架构优化的 Attention Kernel，在 Generation Phase 中利用 Key/Value Head 共享特性减少计算和内存访问 |

---

> **提示：** 本表收录了书中出现频率最高的术语。如遇到未收录的术语，可参考 [NVIDIA TensorRT-LLM 官方文档](https://nvidia.github.io/TensorRT-LLM/) 或 [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)。
