import{_ as n,o as a,c as p,ag as i}from"./chunks/framework.BZohXCq9.js";const k=JSON.parse('{"title":"第 2 章：Repo 结构全景","description":"","frontmatter":{},"headers":[],"relativePath":"chapters/02-repo-structure.md","filePath":"chapters/02-repo-structure.md","lastUpdated":null}'),e={name:"chapters/02-repo-structure.md"};function l(t,s,h,o,r,c){return a(),p("div",null,[...s[0]||(s[0]=[i(`<h1 id="第-2-章-repo-结构全景" tabindex="-1">第 2 章：Repo 结构全景 <a class="header-anchor" href="#第-2-章-repo-结构全景" aria-label="Permalink to &quot;第 2 章：Repo 结构全景&quot;">​</a></h1><blockquote><p>&quot;Show me your data structures, and I won&#39;t need your code. Show me your directory layout, and I won&#39;t need your architecture doc.&quot; —— 改编自 Fred Brooks</p></blockquote><p>在深入源码之前，我们需要先建立对整个项目的全局认知。TensorRT-LLM 是一个典型的 Python + C++/CUDA 混合项目，代码规模庞大、模块众多。本章将带你从顶层目录开始，逐步建立对整个代码仓库的空间感。</p><h2 id="_2-1-顶层目录结构" tabindex="-1">2.1 顶层目录结构 <a class="header-anchor" href="#_2-1-顶层目录结构" aria-label="Permalink to &quot;2.1 顶层目录结构&quot;">​</a></h2><p>克隆 <code>NVIDIA/TensorRT-LLM</code> 仓库后，顶层目录结构如下：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>TensorRT-LLM/</span></span>
<span class="line"><span>├── tensorrt_llm/          # Python API 层（核心）</span></span>
<span class="line"><span>├── cpp/                   # C++ 运行时与 CUDA kernel（核心）</span></span>
<span class="line"><span>├── examples/              # 各模型的使用示例</span></span>
<span class="line"><span>├── scripts/               # 构建脚本与工具</span></span>
<span class="line"><span>├── 3rdparty/              # 第三方依赖（git submodule）</span></span>
<span class="line"><span>├── tests/                 # 测试套件</span></span>
<span class="line"><span>├── benchmarks/            # 性能基准测试</span></span>
<span class="line"><span>├── docker/                # Docker 构建文件</span></span>
<span class="line"><span>├── docs/                  # 文档</span></span>
<span class="line"><span>├── setup.py               # Python 包安装入口</span></span>
<span class="line"><span>├── CMakeLists.txt         # C++ 构建入口</span></span>
<span class="line"><span>└── README.md</span></span></code></pre></div><p>从代码量的角度粗略估计，整个项目的构成大致为：</p><table tabindex="0"><thead><tr><th>语言</th><th>占比（估算）</th><th>主要分布</th></tr></thead><tbody><tr><td>Python</td><td>~40%</td><td><code>tensorrt_llm/</code>、<code>examples/</code>、<code>tests/</code></td></tr><tr><td>C++</td><td>~35%</td><td><code>cpp/</code> 下的运行时与插件</td></tr><tr><td>CUDA</td><td>~20%</td><td><code>cpp/</code> 下的 kernel 实现</td></tr><tr><td>其他（CMake、Shell 等）</td><td>~5%</td><td><code>scripts/</code>、构建文件</td></tr></tbody></table><p>Python 和 C++/CUDA 的代码量几乎对半分，这反映了 TensorRT-LLM 的双层架构设计：<strong>Python 负责易用性，C++/CUDA 负责极致性能</strong>。</p><h2 id="_2-2-tensorrt-llm-——-python-api-层" tabindex="-1">2.2 tensorrt_llm/ —— Python API 层 <a class="header-anchor" href="#_2-2-tensorrt-llm-——-python-api-层" aria-label="Permalink to &quot;2.2 tensorrt_llm/ —— Python API 层&quot;">​</a></h2><p>这是用户直接接触最多的部分，提供了模型定义、编译构建、运行时调用的完整 Python 接口。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>tensorrt_llm/</span></span>
<span class="line"><span>├── models/                # 各模型架构的 Python 定义</span></span>
<span class="line"><span>│   ├── llama/</span></span>
<span class="line"><span>│   │   ├── model.py       # LLaMA 模型结构定义</span></span>
<span class="line"><span>│   │   └── convert.py     # 权重转换逻辑</span></span>
<span class="line"><span>│   ├── gpt/</span></span>
<span class="line"><span>│   ├── chatglm/</span></span>
<span class="line"><span>│   ├── qwen/</span></span>
<span class="line"><span>│   ├── falcon/</span></span>
<span class="line"><span>│   └── ...                # 50+ 模型支持</span></span>
<span class="line"><span>├── runtime/               # Python 运行时</span></span>
<span class="line"><span>│   ├── generation.py      # 文本生成逻辑</span></span>
<span class="line"><span>│   └── model_runner.py    # 模型加载与执行</span></span>
<span class="line"><span>├── builder.py             # TRT engine 构建器</span></span>
<span class="line"><span>├── functional.py          # 函数式 API（类比 torch.nn.functional）</span></span>
<span class="line"><span>├── module.py              # 基础 Module 类（类比 torch.nn.Module）</span></span>
<span class="line"><span>├── layers/                # 通用层定义</span></span>
<span class="line"><span>│   ├── linear.py          # 线性层</span></span>
<span class="line"><span>│   ├── attention.py       # 注意力层</span></span>
<span class="line"><span>│   ├── embedding.py       # 嵌入层</span></span>
<span class="line"><span>│   └── moe.py             # Mixture of Experts 层</span></span>
<span class="line"><span>├── quantization/          # 量化相关</span></span>
<span class="line"><span>│   ├── quantize.py</span></span>
<span class="line"><span>│   └── mode.py</span></span>
<span class="line"><span>└── plugin/                # TRT plugin 的 Python 封装</span></span>
<span class="line"><span>    └── plugin.py</span></span></code></pre></div><p>几个关键文件值得特别关注：</p><p><strong><code>builder.py</code></strong> 是整个编译流程的入口。它负责将用户定义的模型转换为 TensorRT network，然后调用 TensorRT 编译器生成优化后的 engine。</p><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># tensorrt_llm/builder.py 中的核心流程（简化）</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">class</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> Builder</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    def</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> build_engine</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(self, model, build_config):</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # 1. 创建 TensorRT network</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        network </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tensorrt_llm.Network()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # 2. 在 network 上下文中执行模型前向传播</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        #    这一步不是真正的计算，而是构建计算图</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        with</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> network:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            inputs </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> model.prepare_inputs(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            model(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">**</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">inputs)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # 3. 调用 TensorRT 编译器优化并生成 engine</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        engine </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> trt_builder.build_serialized_network(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            network.trt_network, config</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        )</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> engine</span></span></code></pre></div><p><strong><code>functional.py</code></strong> 提供了所有基础算子的函数式接口，如 <code>matmul</code>、<code>softmax</code>、<code>gelu</code>、<code>rope</code> 等。这些函数并不执行实际计算，而是向 TensorRT network 中添加相应的计算节点。这种设计与 PyTorch 的 <code>torch.nn.functional</code> 类似，但底层机制完全不同。</p><p><strong><code>models/</code> 目录</strong>是模型支持的核心。每个模型子目录通常包含：</p><ul><li><code>model.py</code>：用 TensorRT-LLM 的 <code>Module</code> API 定义模型结构</li><li><code>convert.py</code>：从 HuggingFace checkpoint 转换权重格式</li></ul><h2 id="_2-3-cpp-——-c-运行时与-cuda-kernel" tabindex="-1">2.3 cpp/ —— C++ 运行时与 CUDA Kernel <a class="header-anchor" href="#_2-3-cpp-——-c-运行时与-cuda-kernel" aria-label="Permalink to &quot;2.3 cpp/ —— C++ 运行时与 CUDA Kernel&quot;">​</a></h2><p>这是 TensorRT-LLM 性能的根基所在。所有对延迟敏感的操作都在 C++ 层实现。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>cpp/</span></span>
<span class="line"><span>├── tensorrt_llm/</span></span>
<span class="line"><span>│   ├── runtime/               # C++ 运行时核心</span></span>
<span class="line"><span>│   │   ├── gptSession.cpp     # 推理会话管理</span></span>
<span class="line"><span>│   │   ├── gptDecoder.cpp     # 解码器逻辑</span></span>
<span class="line"><span>│   │   ├── bufferManager.cpp  # 显存管理</span></span>
<span class="line"><span>│   │   └── tllmRuntime.cpp    # TRT engine 加载与执行</span></span>
<span class="line"><span>│   ├── kernels/               # 自定义 CUDA kernel</span></span>
<span class="line"><span>│   │   ├── decoderMaskedMultiheadAttention/  # MHA kernel</span></span>
<span class="line"><span>│   │   ├── samplingTopKKernels.cu            # Top-K 采样</span></span>
<span class="line"><span>│   │   ├── samplingTopPKernels.cu            # Top-P 采样</span></span>
<span class="line"><span>│   │   ├── beamSearchKernels.cu              # Beam Search</span></span>
<span class="line"><span>│   │   └── quantization/                      # 量化 kernel</span></span>
<span class="line"><span>│   ├── plugins/               # TensorRT 插件</span></span>
<span class="line"><span>│   │   ├── gptAttentionPlugin/    # GPT Attention 插件</span></span>
<span class="line"><span>│   │   ├── ncclPlugin/            # NCCL 通信插件</span></span>
<span class="line"><span>│   │   ├── lookupPlugin/          # Embedding lookup 插件</span></span>
<span class="line"><span>│   │   └── smoothQuantGemmPlugin/ # SmoothQuant GEMM 插件</span></span>
<span class="line"><span>│   ├── batch_manager/         # 批处理调度器</span></span>
<span class="line"><span>│   │   ├── inferenceRequest.cpp</span></span>
<span class="line"><span>│   │   ├── schedulerPolicy.cpp</span></span>
<span class="line"><span>│   │   └── kvCacheManager.cpp     # KV Cache 分页管理</span></span>
<span class="line"><span>│   └── common/                # 公共工具</span></span>
<span class="line"><span>│       ├── memoryUtils.cu</span></span>
<span class="line"><span>│       └── cudaUtils.cpp</span></span>
<span class="line"><span>├── include/                   # 头文件</span></span>
<span class="line"><span>│   └── tensorrt_llm/</span></span>
<span class="line"><span>│       ├── runtime/</span></span>
<span class="line"><span>│       ├── kernels/</span></span>
<span class="line"><span>│       └── batch_manager/</span></span>
<span class="line"><span>├── tests/                     # C++ 单元测试</span></span>
<span class="line"><span>└── CMakeLists.txt</span></span></code></pre></div><p><strong><code>kernels/</code> 目录</strong>是 CUDA 工程的核心。以 <code>decoderMaskedMultiheadAttention/</code> 为例，这个目录下实现了 generation phase 的 multi-head attention kernel，针对不同的 head size（64、128、256）、不同的数据类型（FP16、BF16、FP8）、不同的 KV Cache 布局做了高度特化的实现。这些 kernel 直接操作 GPU 共享内存和寄存器，手工优化了数据加载、计算和存储的每一个步骤。</p><div class="language-cpp vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">cpp</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// cpp/tensorrt_llm/kernels/ 下的 kernel 设计哲学（概念示例）</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 不是逐行读代码，而是理解其架构决策</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 1. 针对 head_size 做编译期特化，避免运行时分支</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">template</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> &lt;</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">int</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> HEAD_SIZE</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">typename</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> T</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">typename</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> KVCacheT</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&gt;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">__global__ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">void</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> masked_multihead_attention_kernel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> AttentionParams</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">T</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> params) {</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    // 2. 每个 thread block 处理一个 head</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    // 3. 利用 shared memory 做 Q*K^T 的分块计算</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    // 4. 在线 softmax（streaming softmax）避免两遍扫描</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    // 5. 最终的 attention_output 直接写回 global memory</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">// 6. 通过 dispatcher 在运行时选择正确的模板实例</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">void</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> dispatch_mha_kernel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> AttentionParams</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;</span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;"> params</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    switch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (params.head_size) {</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        case</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:  </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">launch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&lt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">half</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ...&gt;(); </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">break</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        case</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">: </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">launch</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">&lt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">half</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ...&gt;(); </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">break</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">;</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        // ...</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    }</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p><strong><code>plugins/</code> 目录</strong>实现了 TensorRT 插件。TensorRT 的标准算子库无法覆盖 LLM 的所有需求（如带 KV Cache 的 attention、RoPE 位置编码等），因此 TensorRT-LLM 通过插件机制将自定义 CUDA kernel 注册到 TensorRT 编译器中。<code>gptAttentionPlugin/</code> 是最核心的插件之一，它将 attention 计算、KV Cache 更新、RoPE 编码等操作封装为一个整体插件，避免了拆分为多个小算子带来的性能损失。</p><p><strong><code>batch_manager/</code> 目录</strong>实现了 In-flight Batching 的调度逻辑。<code>kvCacheManager.cpp</code> 管理分页 KV Cache 的分配与释放，<code>schedulerPolicy.cpp</code> 决定每一步哪些请求参与计算。这是 TensorRT-LLM 在服务化场景下获得高吞吐量的关键组件。</p><h2 id="_2-4-examples-——-模型使用示例" tabindex="-1">2.4 examples/ —— 模型使用示例 <a class="header-anchor" href="#_2-4-examples-——-模型使用示例" aria-label="Permalink to &quot;2.4 examples/ —— 模型使用示例&quot;">​</a></h2><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>examples/</span></span>
<span class="line"><span>├── llama/</span></span>
<span class="line"><span>│   ├── convert_checkpoint.py   # 权重转换</span></span>
<span class="line"><span>│   └── README.md</span></span>
<span class="line"><span>├── gpt/</span></span>
<span class="line"><span>├── chatglm/</span></span>
<span class="line"><span>├── qwen/</span></span>
<span class="line"><span>├── whisper/                    # 语音模型也支持</span></span>
<span class="line"><span>└── ...</span></span></code></pre></div><p>每个模型目录通常提供从权重转换、engine 构建到推理执行的完整流程示例。这些示例是学习 TensorRT-LLM 使用方法的最佳起点。</p><h2 id="_2-5-其他重要目录" tabindex="-1">2.5 其他重要目录 <a class="header-anchor" href="#_2-5-其他重要目录" aria-label="Permalink to &quot;2.5 其他重要目录&quot;">​</a></h2><p><strong><code>3rdparty/</code></strong> 以 git submodule 形式引入第三方依赖：</p><ul><li><strong>CUTLASS</strong>：NVIDIA 的 CUDA 模板线性代数库，TensorRT-LLM 中大量 GEMM kernel 基于它实现</li><li><strong>NCCL</strong>：NVIDIA 集合通信库，用于多卡并行推理</li><li><strong>FasterTransformer</strong>：部分遗留的高性能 kernel 仍来自 FT</li><li><strong>json</strong>：nlohmann/json，C++ JSON 解析</li><li><strong>googletest</strong>：C++ 单元测试框架</li></ul><p><strong><code>tests/</code></strong> 包含 Python 端的测试套件，覆盖模型正确性验证、精度对比、功能测试等。<code>cpp/tests/</code> 下是 C++ 单元测试。</p><p><strong><code>benchmarks/</code></strong> 提供了标准化的性能基准测试脚本，可以用来对比不同配置下的吞吐量和延迟。</p><p><strong><code>scripts/</code></strong> 包含构建脚本、Docker 镜像构建脚本和各种工具脚本。<code>build_wheel.py</code> 是构建 Python wheel 包的入口。</p><h2 id="_2-6-核心模块关系图" tabindex="-1">2.6 核心模块关系图 <a class="header-anchor" href="#_2-6-核心模块关系图" aria-label="Permalink to &quot;2.6 核心模块关系图&quot;">​</a></h2><p>用文字描述 TensorRT-LLM 各模块之间的依赖关系：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>用户代码</span></span>
<span class="line"><span>  │</span></span>
<span class="line"><span>  ▼</span></span>
<span class="line"><span>tensorrt_llm.models (模型定义)</span></span>
<span class="line"><span>  │  调用</span></span>
<span class="line"><span>  ▼</span></span>
<span class="line"><span>tensorrt_llm.layers (通用层: attention, linear, ...)</span></span>
<span class="line"><span>  │  调用</span></span>
<span class="line"><span>  ▼</span></span>
<span class="line"><span>tensorrt_llm.functional (基础算子: matmul, softmax, ...)</span></span>
<span class="line"><span>  │  构建</span></span>
<span class="line"><span>  ▼</span></span>
<span class="line"><span>TensorRT Network (计算图)</span></span>
<span class="line"><span>  │  编译 (Builder)</span></span>
<span class="line"><span>  ▼</span></span>
<span class="line"><span>TensorRT Engine (.engine 二进制文件)</span></span>
<span class="line"><span>  │  加载</span></span>
<span class="line"><span>  ▼</span></span>
<span class="line"><span>cpp/runtime (C++ 运行时)</span></span>
<span class="line"><span>  │  调用                    │  调度</span></span>
<span class="line"><span>  ▼                         ▼</span></span>
<span class="line"><span>cpp/plugins (TRT 插件)   cpp/batch_manager (批调度)</span></span>
<span class="line"><span>  │  调用                    │  管理</span></span>
<span class="line"><span>  ▼                         ▼</span></span>
<span class="line"><span>cpp/kernels (CUDA kernel)  KV Cache / 显存管理</span></span>
<span class="line"><span>  │</span></span>
<span class="line"><span>  ▼</span></span>
<span class="line"><span>cuBLAS / CUTLASS / NCCL (底层库)</span></span></code></pre></div><p>从这张关系图可以看出，TensorRT-LLM 的架构是一个清晰的<strong>分层设计</strong>：Python 层负责模型定义和编译流程的编排，C++ 层负责运行时的高性能执行。两层之间通过 TensorRT engine 文件作为桥梁——这也意味着编译阶段和运行阶段可以完全分离。</p><h2 id="_2-7-从入口到引擎-一次推理的调用链" tabindex="-1">2.7 从入口到引擎：一次推理的调用链 <a class="header-anchor" href="#_2-7-从入口到引擎-一次推理的调用链" aria-label="Permalink to &quot;2.7 从入口到引擎：一次推理的调用链&quot;">​</a></h2><p>为了建立端到端的直觉，我们简述一次完整推理请求经历的调用链：</p><div class="language-python vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">python</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;"># 1. 用户发起请求</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">output </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> model_runner.generate(input_ids, </span><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70;">max_new_tokens</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>generate()                          # tensorrt_llm/runtime/model_runner.py</span></span>
<span class="line"><span>  └─► Session.run()                 # Python → C++ 绑定</span></span>
<span class="line"><span>       └─► GptSession::generateV2() # cpp/tensorrt_llm/runtime/gptSession.cpp</span></span>
<span class="line"><span>            ├─► Context Phase        # 处理完整 prompt</span></span>
<span class="line"><span>            │   └─► TRT Engine 执行  # attention plugin → custom kernel</span></span>
<span class="line"><span>            │       └─► KV Cache 填充</span></span>
<span class="line"><span>            └─► Generation Phase     # 逐 token 生成</span></span>
<span class="line"><span>                ├─► TRT Engine 执行  # masked MHA kernel</span></span>
<span class="line"><span>                ├─► Sampling         # top-k / top-p 采样 kernel</span></span>
<span class="line"><span>                └─► KV Cache 追加</span></span></code></pre></div><p><strong>Context Phase</strong>（也叫 prefill phase）处理整个输入 prompt，是一个计算密集的矩阵乘操作，主要瓶颈在 GEMM 的计算带宽上。<strong>Generation Phase</strong> 逐 token 生成，每一步只有一个新 token 的 query 向量，主要瓶颈在显存带宽（读取 KV Cache）上。这两个阶段的性能特征截然不同，TensorRT-LLM 为它们分别优化了不同的 kernel 实现。</p><p>理解了这条调用链，后续章节对每个模块的深入分析就有了清晰的锚点。</p><h2 id="本章小结" tabindex="-1">本章小结 <a class="header-anchor" href="#本章小结" aria-label="Permalink to &quot;本章小结&quot;">​</a></h2><p>本章对 TensorRT-LLM 代码仓库进行了全景式的梳理。项目采用 Python + C++/CUDA 双层架构：<code>tensorrt_llm/</code> 目录提供用户友好的 Python API，涵盖模型定义（<code>models/</code>）、编译构建（<code>builder.py</code>）、函数式算子（<code>functional.py</code>）等核心模块；<code>cpp/</code> 目录实现高性能运行时，包括 CUDA kernel（<code>kernels/</code>）、TensorRT 插件（<code>plugins/</code>）、批处理调度（<code>batch_manager/</code>）等关键组件。两层之间通过 TensorRT engine 文件连接，编译阶段与运行阶段清晰分离。<code>examples/</code>、<code>tests/</code>、<code>benchmarks/</code> 等目录提供了完善的示例、测试与基准支持。在后续章节中，我们将沿着本章建立的全景地图，逐一深入每个核心模块的实现细节。</p>`,46)])])}const g=n(e,[["render",l]]);export{k as __pageData,g as default};
