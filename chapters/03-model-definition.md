# 第 3 章：模型定义方式

> "All models are wrong, but some are useful." —— George Box

在前两章中，我们了解了 TensorRT-LLM 的整体架构和编译流程。从本章开始，我们将深入源码，逐层拆解 TensorRT-LLM 是如何让用户以"写 PyTorch"的方式来构建高性能推理引擎的。这套模型定义体系是整个框架的基石——它决定了开发者如何描述一个大语言模型，以及这些描述如何最终转化为 TensorRT 的计算图。

## 3.1 Functional API：算子层的封装

打开 `tensorrt_llm/functional.py`，你会看到一组熟悉的函数名：`matmul`、`softmax`、`gelu`、`relu`、`layer_norm`、`concat`、`select` 等等。这些函数的命名和签名与 PyTorch 的 `torch.nn.functional` 几乎一致，但它们的内部实现完全不同。

```python
# tensorrt_llm/functional.py 中的 matmul 实现（简化）
def matmul(input: Tensor, mat2: Tensor, transa: bool = False, transb: bool = False) -> Tensor:
    # 获取当前正在构建的 TensorRT network
    input, mat2 = broadcast_helper(input, mat2)
    layer = default_trtnet().add_matrix_multiply(
        input.trt_tensor,
        trt.MatrixOperation.TRANSPOSE if transa else trt.MatrixOperation.NONE,
        mat2.trt_tensor,
        trt.MatrixOperation.TRANSPOSE if transb else trt.MatrixOperation.NONE,
    )
    return _create_tensor(layer.get_output(0), layer)
```

关键点在于：这里的 `matmul` **并不执行矩阵乘法**，而是在 TensorRT 的 `INetworkDefinition` 上添加一个 `IMatrixMultiplyLayer`。返回的 `Tensor` 对象包裹的是 `trt.ITensor`，即计算图中的一条边，而非实际数据。

`default_trtnet()` 是一个全局函数，返回当前线程正在构建的 TensorRT network 对象。整个 Functional API 的设计模式可以概括为：

1. 接收 TensorRT-LLM 的 `Tensor` 对象作为输入
2. 调用 TensorRT 的 C++ API 向 network 添加一个 layer
3. 将输出的 `trt.ITensor` 包装成 TensorRT-LLM 的 `Tensor` 返回

这种设计使得后续的 Module 层可以像写 PyTorch 一样组合这些算子，而底层实际上是在构建 TensorRT 的计算图。

除了标准算子外，`functional.py` 中还包含一些 LLM 推理特有的函数，例如 `rotary_embedding`、`smooth_quant_layer_norm` 和 `rms_norm` 等。这些函数通常通过 TensorRT plugin 来实现，因为标准的 TensorRT layer 无法高效表达这些操作。

```python
# 通过 plugin 实现的 RMSNorm（简化）
def rms_norm(input: Tensor, normalized_shape: int, weight: Tensor, eps: float = 1e-6) -> Tensor:
    plugin_creator = trt.get_plugin_registry().get_plugin_creator('Rmsnorm', '1', TRT_LLM_PLUGIN_NAMESPACE)
    # 构造 plugin field collection
    p_dtype = trt.PluginField("type_id", ...)
    p_eps = trt.PluginField("eps", ...)
    plugin = plugin_creator.create_plugin("rmsnorm", trt.PluginFieldCollection([p_dtype, p_eps]))
    layer = default_trtnet().add_plugin_v2([input.trt_tensor, weight.trt_tensor], plugin)
    return _create_tensor(layer.get_output(0), layer)
```

## 3.2 Module 抽象：熟悉的 nn.Module 范式

`tensorrt_llm/module.py` 中定义了 `Module` 基类，它的 API 设计与 `torch.nn.Module` 高度相似：

```python
# tensorrt_llm/module.py（核心结构简化）
class Module:
    def __init__(self):
        self._modules = {}    # 子模块字典
        self._parameters = {} # 参数字典

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_modules(self):
        # 递归遍历所有子模块，与 PyTorch 行为一致
        ...

    def named_parameters(self):
        # 递归遍历所有参数
        ...
```

开发者可以通过继承 `Module` 来定义自己的网络结构，在 `__init__` 中声明子模块和参数，在 `forward` 中描述前向计算逻辑。这种模式对于 PyTorch 用户来说毫无学习成本。

但这里有一个**本质差异**必须理解：PyTorch 的 `Module.forward()` 在调用时会真正执行张量运算，输入和输出都是包含实际数据的 `torch.Tensor`。而 TensorRT-LLM 的 `Module.forward()` 在调用时只是在 TensorRT network 上添加节点——它是一个**图构建过程**，而非计算过程。

```python
# PyTorch：forward() 时执行计算
output = pytorch_model(real_input_data)  # output 包含实际数值

# TensorRT-LLM：forward() 时构建图
output = trtllm_model(symbolic_tensor)   # output 是图中的一条边
```

这意味着 TensorRT-LLM 的 `forward()` 中不能包含依赖于具体数值的控制流（例如 `if tensor.sum() > 0`），因为在构建阶段张量还没有实际数值。这一限制与 TorchScript 的 trace 模式类似。

## 3.3 Parameter 类

`Module` 中使用的参数并非 `torch.nn.Parameter`，而是 TensorRT-LLM 自己的 `Parameter` 类（定义在 `tensorrt_llm/parameter.py`）。`Parameter` 本质上是一个带有元信息的 `trt.ITensor` 占位符，在图构建阶段它代表网络的权重输入。

```python
# tensorrt_llm/parameter.py（简化）
class Parameter:
    def __init__(self, value=None, shape=None, dtype=None):
        self._value = value    # 可以为 None，在 build 时填充
        self.shape = shape
        self.dtype = dtype
```

权重的实际数值是在引擎编译（build）阶段通过 `named_parameters()` 遍历并绑定的。这种延迟绑定的设计允许同一个模型定义适配不同的权重来源。

## 3.4 Layer 类：开箱即用的网络层

`tensorrt_llm/layers/` 目录下提供了丰富的预定义网络层，这些是实际构建 LLM 时最常用的组件。

**Linear 层**（`tensorrt_llm/layers/linear.py`）是最基础的全连接层：

```python
# tensorrt_llm/layers/linear.py（简化）
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype=None):
        super().__init__()
        self.weight = Parameter(shape=(out_features, in_features), dtype=dtype)
        if bias:
            self.bias = Parameter(shape=(out_features,), dtype=dtype)

    def forward(self, x):
        # 调用 functional.py 中的 matmul
        y = matmul(x, self.weight.value, transb=True)
        if self.bias is not None:
            y = y + self.bias.value
        return y
```

在此基础上，TensorRT-LLM 提供了**张量并行**版本的线性层：

- `ColumnLinear`：沿输出维度切分权重，每个 GPU 持有部分列。适用于 MLP 的第一个线性层和 Attention 的 QKV 投影。
- `RowLinear`：沿输入维度切分权重，每个 GPU 持有部分行。适用于 MLP 的第二个线性层和 Attention 的输出投影。计算完成后需要一次 AllReduce。

```python
# ColumnLinear：按列切分（简化）
class ColumnLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, dtype=None,
                 tp_group=None, tp_size=1, gather_output=True):
        # out_features 除以 tp_size
        super().__init__(in_features, out_features // tp_size, bias, dtype)
        self.tp_group = tp_group
        self.gather_output = gather_output

    def forward(self, x):
        y = matmul(x, self.weight.value, transb=True)
        if self.gather_output and self.tp_group is not None:
            y = allgather(y, self.tp_group)  # 收集所有 GPU 的结果
        return y
```

**Embedding 层**（`tensorrt_llm/layers/embedding.py`）负责将 token id 映射为向量表示。LLM 中的 embedding 表通常很大（词表大小 x 隐藏维度），因此也支持张量并行切分。

**LayerNorm 和 RMSNorm** 定义在 `tensorrt_llm/layers/normalization.py` 中。现代 LLM 大多使用 RMSNorm（如 LLaMA、Mistral），它省去了均值计算，效率更高。这些归一化层通常通过 TensorRT plugin 实现，以获得 kernel fusion 带来的性能提升。

## 3.5 网络构建全流程

理解了各层组件后，让我们串联起整个网络构建流程。从模型定义到推理引擎，共经历三个阶段：

**阶段一：定义模型结构**

```python
# 用户定义模型（伪代码）
class MyModel(Module):
    def __init__(self, config):
        self.embed = Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.ln_f = RMSNorm(config.hidden_size)
        self.lm_head = ColumnLinear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, position_ids):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits
```

**阶段二：构建 TensorRT network**

```python
# tensorrt_llm/builder.py 中的关键逻辑（简化）
builder = trt.Builder(logger)
network = builder.create_network()

with net_guard(network):
    # 创建输入占位符
    input_ids = Tensor(name='input_ids', dtype=trt.int32, shape=[-1, -1])
    position_ids = Tensor(name='position_ids', dtype=trt.int32, shape=[-1, -1])

    # 调用 forward() —— 此时在构建图，不是在计算
    logits = model(input_ids, position_ids)
    logits.mark_output('logits')
```

**阶段三：编译为引擎**

```python
# 绑定权重、设置优化参数、编译
build_config = builder.create_builder_config()
engine = builder.build_serialized_network(network, build_config)
```

这三个阶段清晰地分离了**模型描述**、**图构建**和**引擎编译**的职责。正是这种分离使得 TensorRT-LLM 既能提供友好的 Python API，又能生成高度优化的推理引擎。

## 本章小结

本章我们深入分析了 TensorRT-LLM 的模型定义体系。Functional API（`tensorrt_llm/functional.py`）提供了与 PyTorch 对齐的算子接口，但底层是在构建 TensorRT 计算图。Module 抽象（`tensorrt_llm/module.py`）复刻了 `nn.Module` 的设计范式，使模型定义代码几乎可以从 PyTorch 直接迁移。Layer 层（`tensorrt_llm/layers/`）提供了 `Linear`、`ColumnLinear`、`RowLinear`、`Embedding`、`LayerNorm` 等开箱即用的组件，并内置了张量并行支持。最关键的认知是：TensorRT-LLM 的 `forward()` 是图构建而非计算执行，这一本质差异决定了它与 PyTorch 在使用上的所有不同。下一章我们将深入 Attention 层的实现，看看 TensorRT-LLM 是如何将 Flash Attention、KV Cache 等优化技术融合进这套体系的。
