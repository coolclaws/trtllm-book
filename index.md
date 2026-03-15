---
layout: home

hero:
  name: "TensorRT-LLM"
  text: "源码解析"
  tagline: "NVIDIA 官方 LLM 推理加速引擎——从量化编译到 TensorRT 图优化深度剖析"
  image:
    src: /logo.png
    alt: TensorRT-LLM
  actions:
    - theme: brand
      text: 开始阅读 →
      link: /chapters/01-overview
    - theme: alt
      text: GitHub 仓库
      link: https://github.com/NVIDIA/TensorRT-LLM

features:
  - icon:
      src: /icons/architecture.svg
    title: 系统架构全景
    details: 解析 TensorRT-LLM 在 NVIDIA 生态中的定位，深入 Python 模型层、C++ Runtime、TensorRT Plugin 三层架构的协作关系。
    link: /chapters/01-overview
  - icon:
      src: /icons/model.svg
    title: 模型定义与权重转换
    details: 剖析 Functional API、Module 抽象体系、内置模型实现与权重转换工具链，理解从 HuggingFace 到 TRT-LLM 格式的完整路径。
    link: /chapters/03-model-definition
  - icon:
      src: /icons/compile.svg
    title: 编译与量化优化
    details: 深入 trtllm-build 编译流程、TensorRT 插件体系、AWQ/GPTQ/FP8 量化编译与 Graph 优化，掌握推理图的构建与加速机制。
    link: /chapters/06-build-flow
  - icon:
      src: /icons/performance.svg
    title: 高性能运行时
    details: 覆盖 GenerationSession、Paged KV Cache、In-flight Batching、Tensor/Pipeline 并行与 Triton 服务化，理解 NVIDIA 推理引擎的生产级实现。
    link: /chapters/10-generation-session
---
