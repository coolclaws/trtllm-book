# 第 5 章：内置模型支持

> "不要重复发明轮子，但要理解轮子是怎么造的。" —— 工程谚语

TensorRT-LLM 内置了对数十种主流大语言模型的支持，涵盖 LLaMA、Mistral、Qwen、GPT、Falcon、ChatGLM、Baichuan 等。这些预定义的模型实现让用户无需从零编写模型代码，只需提供权重文件即可快速构建高性能推理引擎。本章将深入 `tensorrt_llm/models/` 目录，以 LLaMA 为主线，详解模型定义、权重转换和注册机制。

## 5.1 模型目录结构

`tensorrt_llm/models/` 的目录结构清晰地反映了支持的模型列表：

```
tensorrt_llm/models/
├── __init__.py           # 模型注册表
├── modeling_utils.py     # 公共基类和工具函数
├── llama/
│   ├── __init__.py
│   ├── model.py          # 模型定义
│   └── convert_checkpoint.py  # 权重转换
├── mistral/
│   ├── model.py
│   └── convert_checkpoint.py
├── qwen/
│   ├── model.py
│   └── convert_checkpoint.py
├── gpt/
│   ├── model.py
│   └── convert_checkpoint.py
├── falcon/
│   ├── model.py
│   └── convert_checkpoint.py
├── chatglm/
│   ├── model.py
│   └── convert_checkpoint.py
└── ...
```

每个模型通常包含两个核心文件：`model.py` 负责定义网络结构，`convert_checkpoint.py` 负责将 HuggingFace 等来源的权重转换为 TensorRT-LLM 格式。

## 5.2 PretrainedConfig：统一的配置体系

在深入具体模型之前，先了解配置类。`tensorrt_llm/models/modeling_utils.py` 中定义了 `PretrainedConfig`，它是所有模型配置的基类：

```python
# tensorrt_llm/models/modeling_utils.py（简化）
class PretrainedConfig:
    def __init__(self,
                 architecture: str,
                 dtype: str = 'float16',
                 hidden_size: int = 0,
                 num_hidden_layers: int = 0,
                 num_attention_heads: int = 0,
                 num_key_value_heads: Optional[int] = None,
                 vocab_size: int = 0,
                 max_position_embeddings: int = 0,
                 hidden_act: str = 'gelu',
                 intermediate_size: Optional[int] = None,
                 mapping: Optional[Mapping] = None,
                 quantization: Optional[QuantConfig] = None,
                 **kwargs):
        self.architecture = architecture
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.vocab_size = vocab_size
        # ... 更多字段

    @classmethod
    def from_json_file(cls, config_path: str):
        with open(config_path) as f:
            config = json.load(f)
        return cls(**config)

    def to_dict(self):
        return asdict(self)
```

`PretrainedConfig` 的设计哲学是**统一所有模型的共性参数**（如 `hidden_size`、`num_hidden_layers`），同时通过 `**kwargs` 允许各模型携带特有参数。它还包含了 `Mapping` 对象（描述张量并行和流水线并行的拓扑）以及 `QuantConfig`（描述量化配置）。

配置会被序列化为 `config.json` 文件，与引擎文件一起保存。这确保了加载引擎时能正确恢复所有配置信息。

## 5.3 以 LLaMA 为例：模型定义详解

LLaMA 是当前开源 LLM 生态的基石模型，我们以它为例详细分析模型定义。核心代码位于 `tensorrt_llm/models/llama/model.py`。

### 5.3.1 LlamaDecoderLayer

每个 Transformer 层由一个 `LlamaDecoderLayer` 表示：

```python
# tensorrt_llm/models/llama/model.py（简化）
class LlamaDecoderLayer(Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__()
        # 前置 RMSNorm
        self.input_layernorm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            dtype=config.dtype,
        )
        # Self-Attention（使用第 4 章分析的 Attention 类）
        self.self_attn = Attention(
            local_layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_embedding_base=config.rotary_base,
            max_position_embeddings=config.max_position_embeddings,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            dtype=config.dtype,
        )
        # Post-Attention RMSNorm
        self.post_layernorm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.norm_epsilon,
            dtype=config.dtype,
        )
        # MLP（SwiGLU 结构）
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            hidden_act=config.hidden_act,  # 'silu' for LLaMA
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            dtype=config.dtype,
        )

    def forward(self, hidden_states, attention_mask=None,
                use_cache=False, kv_cache_params=None, attention_params=None):
        # Pre-Norm 架构
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Attention
        hidden_states, present_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )
        hidden_states = residual + hidden_states
        # MLP
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_kv
```

这段代码与 HuggingFace Transformers 中的 `LlamaDecoderLayer` 在结构上几乎一一对应。Pre-Norm 架构（先 norm 后 attention/MLP）是 LLaMA 的特征，区别于 GPT-2 的 Post-Norm 架构。

### 5.3.2 LlamaForCausalLM

顶层模型类 `LlamaForCausalLM` 组装了完整的网络：

```python
# tensorrt_llm/models/llama/model.py（简化）
class LlamaModel(Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = ModuleList([
            LlamaDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype)

class LlamaForCausalLM(DecoderModelForCausalLM):
    def __init__(self, config: PretrainedConfig):
        transformer = LlamaModel(config)
        lm_head = ColumnLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            dtype=config.dtype,
        )
        super().__init__(config, transformer, lm_head)
```

`DecoderModelForCausalLM` 是一个公共基类（定义在 `modeling_utils.py` 中），它实现了通用的 `forward()` 逻辑和 `prepare_inputs()` 方法。后者负责创建 TensorRT network 的输入占位符（input_ids、position_ids、cache 相关张量等），是图构建流程的起点。

## 5.4 权重转换流程

模型定义完成后，还需要将预训练权重加载进去。由于 TensorRT-LLM 的权重格式与 HuggingFace 不同（需要处理张量并行切分、数据类型转换、量化等），每个模型都配备了 `convert_checkpoint.py` 脚本。

以 LLaMA 为例，转换流程大致如下：

```python
# tensorrt_llm/models/llama/convert_checkpoint.py（流程简化）
def convert_hf_llama(hf_model_dir: str, output_dir: str,
                     dtype: str = 'float16', tp_size: int = 1):
    # 1. 加载 HuggingFace 配置
    hf_config = AutoConfig.from_pretrained(hf_model_dir)

    # 2. 创建 TRT-LLM 配置
    config = PretrainedConfig(
        architecture='LlamaForCausalLM',
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        # ...
    )

    # 3. 逐层转换权重
    for layer_idx in range(hf_config.num_hidden_layers):
        # 加载 HF 权重
        q_weight = hf_weights[f'model.layers.{layer_idx}.self_attn.q_proj.weight']
        k_weight = hf_weights[f'model.layers.{layer_idx}.self_attn.k_proj.weight']
        v_weight = hf_weights[f'model.layers.{layer_idx}.self_attn.v_proj.weight']

        # 合并 QKV（TRT-LLM 使用 fused QKV）
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        # 按张量并行维度切分
        if tp_size > 1:
            qkv_weight = split_qkv_tp(qkv_weight, tp_size, tp_rank, ...)

        # 类型转换
        qkv_weight = qkv_weight.to(str_dtype_to_torch(dtype))

        # 保存
        weights[f'transformer.layers.{layer_idx}.self_attn.qkv.weight'] = qkv_weight

    # 4. 保存为 safetensors 格式
    save_checkpoint(output_dir, config, weights)
```

权重转换中几个需要注意的要点：

1. **QKV 合并**：HuggingFace 模型通常将 Q、K、V 作为独立的 Linear 层，而 TensorRT-LLM 将它们合并为一个 `ColumnLinear`（见第 4 章），因此需要在转换时进行 concatenation。

2. **张量并行切分**：如果使用多 GPU 推理，需要将权重按 tensor parallelism 的规则切分。ColumnLinear 的权重按列切分，RowLinear 的权重按行切分。

3. **权重命名映射**：HuggingFace 和 TensorRT-LLM 的权重命名规范不同，需要逐一映射。例如 `model.layers.0.self_attn.q_proj.weight` 对应 `transformer.layers.0.self_attn.qkv.weight` 的一部分。

## 5.5 模型注册机制

TensorRT-LLM 通过一个中心化的注册表来管理所有支持的模型。`tensorrt_llm/models/__init__.py` 中维护了从架构名到模型类的映射：

```python
# tensorrt_llm/models/__init__.py（简化）
MODEL_MAP = {
    'LlamaForCausalLM': LlamaForCausalLM,
    'MistralForCausalLM': MistralForCausalLM,
    'QWenForCausalLM': QWenForCausalLM,
    'GPTForCausalLM': GPTForCausalLM,
    'FalconForCausalLM': FalconForCausalLM,
    'ChatGLMForCausalLM': ChatGLMForCausalLM,
    'BaichuanForCausalLM': BaichuanForCausalLM,
    # ... 更多模型
}

def from_config(config: PretrainedConfig):
    """根据配置中的 architecture 字段实例化对应的模型类"""
    arch = config.architecture
    if arch not in MODEL_MAP:
        raise ValueError(f"Unsupported architecture: {arch}")
    model_cls = MODEL_MAP[arch]
    return model_cls(config)
```

这种注册机制使得构建引擎的通用流程只需要读取 `config.json` 中的 `architecture` 字段，即可自动实例化正确的模型类。用户无需关心具体是哪个模型，整个流程高度自动化。

## 5.6 其他模型的差异化实现

虽然各模型共享 `Attention`、`Linear`、`RMSNorm` 等基础组件，但它们之间仍有显著差异需要在模型定义中体现：

**Mistral**：与 LLaMA 架构几乎相同，主要区别在于使用了 Sliding Window Attention（滑动窗口注意力）。在 `Attention` 类中通过 `max_attention_window_size` 参数控制。实际上 Mistral 的模型定义直接复用了 LLaMA 的大部分代码。

**Qwen**：使用了不同的 RoPE 实现和 MLP 结构。Qwen-1 使用了独特的 NTK-aware RoPE 缩放策略。Qwen-2 则更接近 LLaMA 架构。

**GPT（GPT-2/GPT-J/GPT-NeoX）**：使用 Post-Norm 架构（先 attention/MLP 后 norm），且使用 LayerNorm 而非 RMSNorm。GPT-J 和 GPT-NeoX 还使用了 parallel attention（attention 和 MLP 并行计算而非串行）。

**Falcon**：早期版本使用了 Multi-Query Attention，新版本使用 GQA。它还使用了独特的 alibi 位置编码变体（部分 Falcon 模型）。

**ChatGLM**：基于 GLM 架构，使用了独特的双向 attention mask（prefix 部分双向、generation 部分单向），以及自定义的 RoPE 实现（2D 位置编码）。这些差异在 `chatglm/model.py` 中有专门处理。

```python
# 不同模型的关键差异（伪代码对比）

# LLaMA 系列：Pre-Norm + RMSNorm + SwiGLU + RoPE
h = rmsnorm(h) -> attention(h) -> residual -> rmsnorm(h) -> swiglu_mlp(h) -> residual

# GPT-2：Post-Norm + LayerNorm + GELU
h = attention(h) -> residual -> layernorm(h) -> gelu_mlp(h) -> residual -> layernorm(h)

# GPT-J：Parallel Attention + LayerNorm + GELU
h = layernorm(h) -> [attention(h) + gelu_mlp(h)] -> residual
```

## 5.7 添加新模型的路径

如果需要为一个尚未支持的模型添加 TensorRT-LLM 实现，典型的步骤是：

1. 在 `tensorrt_llm/models/` 下创建新目录，添加 `model.py`
2. 继承 `DecoderModelForCausalLM`，定义 `DecoderLayer` 和顶层模型类
3. 编写 `convert_checkpoint.py` 实现权重转换逻辑
4. 在 `__init__.py` 的 `MODEL_MAP` 中注册新模型
5. 如果模型有特殊的 attention 或 MLP 结构，可能需要在 Layer 层甚至 plugin 层做相应扩展

由于 TensorRT-LLM 的模型定义方式与 HuggingFace Transformers 高度对齐，移植过程通常是相对直观的——最大的挑战往往在于权重转换的正确性验证和性能调优。

## 本章小结

本章我们系统分析了 TensorRT-LLM 的内置模型支持体系。`tensorrt_llm/models/` 目录下的每个子目录对应一个模型实现，由 `model.py`（网络结构）和 `convert_checkpoint.py`（权重转换）组成。`PretrainedConfig` 提供了统一的配置基类，将模型超参数、并行策略和量化配置集中管理。以 LLaMA 为例，`LlamaDecoderLayer` 组装了 RMSNorm、Attention 和 GatedMLP，`LlamaForCausalLM` 继承 `DecoderModelForCausalLM` 完成完整的网络定义。权重转换流程负责处理 QKV 合并、张量并行切分和格式映射等细节。模型注册机制通过 `MODEL_MAP` 字典实现架构名到模型类的自动路由，使得整个构建流程高度自动化。理解了这套体系，读者不仅能更好地使用现有模型，也具备了为新模型编写 TensorRT-LLM 支持的能力。
