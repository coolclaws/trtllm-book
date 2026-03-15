import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'TensorRT-LLM 源码解析',
  description: 'NVIDIA 官方 LLM 推理加速引擎——从量化编译到 TensorRT 图优化深度剖析',
  lang: 'zh-CN',
  base: '/',
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#76b900' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:title', content: 'TensorRT-LLM 源码解析' }],
    ['meta', { name: 'og:description', content: 'NVIDIA 官方 LLM 推理加速引擎——从量化编译到 TensorRT 图优化深度剖析' }],
  ],

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'TensorRT-LLM 源码解析',

    nav: [
      { text: '首页', link: '/' },
      { text: '开始阅读', link: '/chapters/01-overview' },
      {
        text: '章节导航',
        items: [
          { text: '第一部分：宏观认知', link: '/chapters/01-overview' },
          { text: '第二部分：模型定义层', link: '/chapters/03-model-definition' },
          { text: '第三部分：编译与构建', link: '/chapters/06-build-flow' },
          { text: '第四部分：运行时', link: '/chapters/10-generation-session' },
          { text: '第五部分：分布式推理', link: '/chapters/14-tensor-parallelism' },
          { text: '第六部分：服务层', link: '/chapters/17-triton-backend' },
          { text: '第七部分：工具链', link: '/chapters/19-benchmark' },
          { text: '附录', link: '/chapters/appendix-a' },
        ]
      }
    ],

    sidebar: [
      {
        text: '第一部分：宏观认知',
        collapsed: false,
        items: [
          { text: '第 1 章：项目概览与设计哲学', link: '/chapters/01-overview' },
          { text: '第 2 章：Repo 结构全景', link: '/chapters/02-repo-structure' },
        ]
      },
      {
        text: '第二部分：模型定义层',
        collapsed: false,
        items: [
          { text: '第 3 章：模型定义方式', link: '/chapters/03-model-definition' },
          { text: '第 4 章：Attention 实现', link: '/chapters/04-attention' },
          { text: '第 5 章：内置模型支持', link: '/chapters/05-builtin-models' },
        ]
      },
      {
        text: '第三部分：编译与构建',
        collapsed: false,
        items: [
          { text: '第 6 章：trtllm-build 编译流程', link: '/chapters/06-build-flow' },
          { text: '第 7 章：TensorRT 插件体系', link: '/chapters/07-plugins' },
          { text: '第 8 章：量化编译', link: '/chapters/08-quantization' },
          { text: '第 9 章：Graph 优化', link: '/chapters/09-graph-optimization' },
        ]
      },
      {
        text: '第四部分：运行时',
        collapsed: false,
        items: [
          { text: '第 10 章：GenerationSession 与 Runner', link: '/chapters/10-generation-session' },
          { text: '第 11 章：C++ 运行时', link: '/chapters/11-cpp-runtime' },
          { text: '第 12 章：Paged KV Cache', link: '/chapters/12-paged-kv-cache' },
          { text: '第 13 章：In-flight Batching', link: '/chapters/13-inflight-batching' },
        ]
      },
      {
        text: '第五部分：分布式推理',
        collapsed: false,
        items: [
          { text: '第 14 章：Tensor Parallelism', link: '/chapters/14-tensor-parallelism' },
          { text: '第 15 章：Pipeline Parallelism', link: '/chapters/15-pipeline-parallelism' },
          { text: '第 16 章：多节点部署', link: '/chapters/16-multi-node' },
        ]
      },
      {
        text: '第六部分：服务层',
        collapsed: false,
        items: [
          { text: '第 17 章：Triton Backend 集成', link: '/chapters/17-triton-backend' },
          { text: '第 18 章：OpenAI API 兼容', link: '/chapters/18-openai-compat' },
        ]
      },
      {
        text: '第七部分：工具链',
        collapsed: false,
        items: [
          { text: '第 19 章：性能基准测试', link: '/chapters/19-benchmark' },
          { text: '第 20 章：与 vLLM/SGLang 选型对比', link: '/chapters/20-comparison' },
        ]
      },
      {
        text: '附录',
        collapsed: false,
        items: [
          { text: '附录 A：推荐阅读路径', link: '/chapters/appendix-a' },
          { text: '附录 B：build_config 参数速查', link: '/chapters/appendix-b' },
          { text: '附录 C：名词解释', link: '/chapters/appendix-c' },
        ]
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/NVIDIA/TensorRT-LLM' }
    ],

    footer: {
      message: '基于 NVIDIA TensorRT-LLM 开源项目的源码解析',
      copyright: 'Copyright © 2024-2026'
    },

    outline: {
      level: [2, 3],
      label: '本页目录'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },

    docFooter: {
      prev: '上一章',
      next: '下一章'
    },

    lastUpdated: {
      text: '最后更新于'
    }
  }
})
