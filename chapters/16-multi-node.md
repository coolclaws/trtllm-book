# 第 16 章：多节点部署

> "Scaling LLM inference beyond a single node is not just a matter of adding more GPUs — it requires careful orchestration of communication, synchronization, and fault tolerance." —— NVIDIA TensorRT-LLM Documentation

当模型规模突破千亿参数，即便是配备 8 张 H100 80GB 的单台服务器也无法完整容纳。例如，LLaMA-2 70B 在 FP16 下需要约 140GB 显存存放权重，加上 KV cache 和激活值，8 卡 H100 勉强够用；而对于 Mixtral 8x22B、GPT-4 级别的模型，多节点部署已成为必然选择。本章将系统讲解 TensorRT-LLM 的多节点推理方案，涵盖 MPI 启动方式、NCCL 跨节点通信、Triton Inference Server 集成部署，以及生产环境中的最佳实践。

## 16.1 多节点推理场景分析

多节点推理的核心驱动因素有三个：

1. **显存容量**：模型权重 + KV cache + 激活值的总和超过单机 GPU 总显存
2. **吞吐需求**：需要更多的 GPU 来支撑高并发请求
3. **冗余与可用性**：分布式部署提供更好的容错能力

以一个具体场景为例：假设我们需要部署一个 175B 参数的模型（FP16 约 350GB），使用 H100 80GB GPU。单机 8 卡提供 640GB 显存，扣除 KV cache 和运行时开销后空间不足。这时需要至少 2 个节点共 16 张 GPU，采用 `tp_size=8, pp_size=2` 的配置——每个节点内部做 TP，两个节点之间做 PP。

## 16.2 MPI 启动方式

TensorRT-LLM 使用 MPI（Message Passing Interface）作为多进程协调的基础设施。MPI 负责进程的启动、rank 分配和初始化阶段的信息交换，而实际的 GPU 间通信由 NCCL 完成。

单节点启动：

```bash
mpirun -n 8 --allow-run-as-root \
    python run.py \
    --engine_dir ./engines \
    --tp_size 8 \
    --pp_size 1
```

多节点启动需要指定 hostfile：

```bash
# hostfile.txt
node0 slots=8
node1 slots=8
```

```bash
mpirun -n 16 \
    --hostfile hostfile.txt \
    --bind-to none \
    -map-by slot \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_NET_GDR_LEVEL=5 \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x LD_LIBRARY_PATH \
    python run.py \
    --engine_dir ./engines \
    --tp_size 8 \
    --pp_size 2
```

这里的关键参数说明：

- `-n 16`：总共启动 16 个进程
- `--hostfile`：指定每个节点的 hostname 和可用 GPU 数（slots）
- `--bind-to none`：不绑定进程到特定 CPU 核
- `-x`：将环境变量传递到所有进程

在 TensorRT-LLM 的代码中（`tensorrt_llm/runtime/session.py`），MPI 的初始化过程如下：

```python
# tensorrt_llm/runtime/session.py (简化)

from mpi4py import MPI

def initialize_mpi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # 设置当前进程使用的 GPU
    local_rank = rank % get_local_size()
    torch.cuda.set_device(local_rank)

    return rank, world_size
```

每个进程通过 `MPI.COMM_WORLD` 获取自己的全局 rank 和 world_size，然后根据 local_rank（节点内编号）选择对应的 GPU 设备。

## 16.3 环境变量配置

多节点部署需要正确配置一系列环境变量，这些变量控制进程间的通信行为：

```bash
# 基础并行配置
export WORLD_SIZE=16          # 全局进程总数
export LOCAL_SIZE=8            # 每个节点的进程数（GPU 数）
export RANK=0                  # 当前进程的全局 rank（由 mpirun 自动设置）
export LOCAL_RANK=0            # 当前进程的节点内 rank（由 mpirun 自动设置）

# NCCL 通信配置
export NCCL_IB_DISABLE=0      # 启用 InfiniBand（多节点必须）
export NCCL_NET_GDR_LEVEL=5   # GPU Direct RDMA 级别
export NCCL_SOCKET_IFNAME=eth0 # 用于 NCCL bootstrap 的网络接口
export NCCL_IB_HCA=mlx5       # 指定 InfiniBand HCA 设备

# 调试配置
export NCCL_DEBUG=INFO         # NCCL 调试信息级别
export NCCL_DEBUG_SUBSYS=ALL   # 调试子系统
```

其中 `NCCL_NET_GDR_LEVEL` 控制 GPU Direct RDMA 的行为级别：

| 级别 | 含义 |
|-----|------|
| 0 | 禁用 GDR |
| 1 | GDR 仅用于 NVLink 直连的 GPU |
| 2 | GDR 用于同一 PCIe switch 下的设备 |
| 5 | GDR 用于所有可能的路径（推荐） |

## 16.4 多节点通信：NCCL over RDMA/InfiniBand

多节点场景下，GPU 间通信的性能瓶颈从 NVLink 转移到了节点间网络。NCCL 支持多种网络后端：

**InfiniBand + GPU Direct RDMA**：这是生产环境的首选方案。GPU Direct RDMA 允许网卡直接访问 GPU 显存，避免了数据经过 CPU 内存的中转，大幅降低延迟和 CPU 开销。

```
GPU 0 (Node 0) ←NVLink→ GPU 1 (Node 0)
    ↕ GPU Direct RDMA
IB HCA (Node 0)
    ↕ InfiniBand Fabric (200-400 Gb/s)
IB HCA (Node 1)
    ↕ GPU Direct RDMA
GPU 0 (Node 1) ←NVLink→ GPU 1 (Node 1)
```

典型的 InfiniBand 网络带宽：

| 代际 | 单端口带宽 | 多端口聚合 |
|-----|---------|---------|
| HDR (200G) | 25 GB/s | 4x = 100 GB/s |
| NDR (400G) | 50 GB/s | 4x = 200 GB/s |

在 TRT-LLM 中，NCCL communicator 的创建会自动检测网络拓扑并选择最优的通信路径。相关代码位于 `cpp/tensorrt_llm/runtime/worldConfig.cpp`：

```cpp
// cpp/tensorrt_llm/runtime/worldConfig.cpp (简化)

WorldConfig WorldConfig::mpi(
    int tensorParallelism, int pipelineParallelism)
{
    auto& comm = MpiComm::world();
    int rank = comm.getRank();
    int worldSize = comm.getSize();

    // 初始化 NCCL communicator
    ncclUniqueId ncclId;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
    // 通过 MPI 广播 NCCL ID 到所有进程
    comm.bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0);

    ncclComm_t ncclComm;
    NCCLCHECK(ncclCommInitRank(&ncclComm, worldSize, ncclId, rank));

    return WorldConfig(rank, worldSize, tensorParallelism,
                       pipelineParallelism, ncclComm);
}
```

这段代码展示了 NCCL 在多节点环境下的初始化流程：rank 0 生成一个唯一的 `ncclUniqueId`，通过 MPI 广播给所有进程，然后每个进程调用 `ncclCommInitRank` 加入同一个 NCCL communicator。之后所有的 AllReduce、Send/Recv 操作都通过这个 communicator 执行。

## 16.5 Triton Inference Server 集成部署

在生产环境中，TensorRT-LLM 通常通过 NVIDIA Triton Inference Server 进行服务化部署。Triton 提供了 HTTP/gRPC 接口、请求调度、模型版本管理等企业级功能。多节点部署架构如下：

```
Client Requests
      ↓
Load Balancer
      ↓
┌─────────────────────────────────┐
│  Triton Server (Node 0)         │
│  ┌────────────────────────────┐ │
│  │  TRT-LLM Backend           │ │
│  │  GPU 0-7 (TP Group, PP=0)  │ │
│  └────────────────────────────┘ │
└──────────────┬──────────────────┘
               │ NCCL over IB
┌──────────────┴──────────────────┐
│  Triton Server (Node 1)         │
│  ┌────────────────────────────┐ │
│  │  TRT-LLM Backend           │ │
│  │  GPU 0-7 (TP Group, PP=1)  │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
```

TRT-LLM 的 Triton backend 配置文件（`model_repository/tensorrt_llm/config.pbtxt`）需要指定并行参数：

```
# config.pbtxt

name: "tensorrt_llm"
backend: "tensorrtllm"
max_batch_size: 128

parameters {
  key: "engine_dir"
  value: { string_value: "/models/engines" }
}
parameters {
  key: "executor_worker_path"
  value: { string_value: "/opt/tritonserver/backends/tensorrtllm/trtllm_executor_worker" }
}
parameters {
  key: "gpu_device_ids"
  value: { string_value: "0,1,2,3,4,5,6,7" }
}
```

多节点 Triton 部署采用 Leader-Worker 架构：

- **Leader 节点**（Node 0）：运行完整的 Triton Server，接收客户端请求，负责请求调度
- **Worker 节点**（Node 1+）：运行 TRT-LLM executor worker 进程，只负责计算

启动方式：

```bash
# Node 0 (Leader)
tritonserver --model-repository=/models \
    --backend-config=tensorrtllm,worker_path=/opt/tritonserver/backends/tensorrtllm/trtllm_executor_worker

# Node 1 (Worker) - 由 Leader 通过 MPI 自动拉起
# 或手动启动：
mpirun -n 8 --host node1 \
    /opt/tritonserver/backends/tensorrtllm/trtllm_executor_worker
```

## 16.6 Docker 容器部署最佳实践

多节点部署通常在容器化环境中进行。以下是关键的 Docker 配置要点：

```dockerfile
# Dockerfile
FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

# 安装额外依赖
RUN pip install tensorrt_llm -U

# 复制模型和 engine
COPY engines/ /models/engines/
COPY model_repository/ /models/
```

运行容器时需要特别注意网络和设备的配置：

```bash
docker run --gpus all \
    --network=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add=IPC_LOCK \
    -v /dev/infiniband:/dev/infiniband \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_NET_GDR_LEVEL=5 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    triton-trtllm:latest
```

关键参数说明：

- `--network=host`：使用主机网络栈，确保 MPI 和 NCCL 能正常通信
- `--ipc=host`：共享主机 IPC 命名空间，NCCL 使用共享内存进行节点内通信
- `--ulimit memlock=-1`：允许锁定无限内存，GPU Direct RDMA 需要锁定页面
- `--cap-add=IPC_LOCK`：允许容器锁定内存页面
- `-v /dev/infiniband`：挂载 InfiniBand 设备到容器内

对于 Kubernetes 部署，还需要安装 NVIDIA GPU Operator 和 NCCL RDMA Sharp Plugin，并配置 `hostNetwork: true` 和适当的资源限制。

## 16.7 网络带宽需求分析

多节点部署的性能很大程度上取决于节点间网络带宽。让我们量化分析 PP 场景下的带宽需求。

PP 的 Send/Recv 通信量为每个 micro-batch 传输一次 hidden states：

$$\text{data\_per\_step} = B \times S \times H \times \text{sizeof(dtype)}$$

以 LLaMA-2 70B（hidden_size=8192）、batch_size=64、seq_len=1（decode 阶段）为例：

$$64 \times 1 \times 8192 \times 2 = 1\text{MB}$$

在 decode 阶段，每个 step 的计算时间约 10~20ms，因此网络需要在这个时间内完成 1MB 的传输。即使是 25 GB/s 的 HDR InfiniBand，传输 1MB 只需约 0.04ms，远小于计算时间，不会成为瓶颈。

但在 prefill 阶段，seq_len 可能达到 4096 或更长：

$$64 \times 4096 \times 8192 \times 2 = 4\text{GB}$$

这时候网络带宽就至关重要了。在 200Gb/s InfiniBand 上传输 4GB 需要约 160ms，可能成为显著的瓶颈。解决方案包括：减小 micro-batch 大小、使用更高带宽的网络（NDR 400G）、或采用序列分块策略减少单次传输量。

如果在多节点之间使用 TP（不推荐但有时不得已），通信量会更大且对延迟更敏感。AllReduce 的通信量为：

$$2 \times \frac{N-1}{N} \times B \times S \times H \times \text{sizeof(dtype)}$$

此时 InfiniBand 的带宽很可能不够，会严重影响推理延迟。因此**强烈建议**：TP 仅在节点内使用，节点间只用 PP。

## 16.8 常见部署问题排查

多节点部署中常见的问题及排查方法：

**1. NCCL 通信超时**

```
NCCL WARN Timeout waiting for NCCL call
```

排查步骤：检查防火墙规则、验证 InfiniBand 链路状态（`ibstat`）、确认 `NCCL_SOCKET_IFNAME` 指向正确的网络接口。

**2. GPU Direct RDMA 失败**

```bash
# 检查 GDR 支持
nvidia-smi topo -m   # 查看 GPU 拓扑
ibv_devinfo           # 查看 IB 设备信息
cat /proc/driver/nvidia-peermem/version  # 检查 peermem 驱动
```

如果 `nvidia-peermem` 模块未加载，GPU Direct RDMA 将回退到经过 CPU 内存的传输路径，性能显著下降。

**3. Rank 不匹配**

确保所有节点的环境一致：相同的 Docker 镜像版本、相同的 NCCL 版本、相同的 engine 文件。每个 rank 必须加载与其 rank 编号对应的 engine 文件。

**4. 性能不及预期**

```bash
# 使用 NCCL 测试工具验证通信带宽
# https://github.com/NVIDIA/nccl-tests
mpirun -n 16 --hostfile hostfile.txt \
    ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

通过 nccl-tests 可以独立于 TRT-LLM 验证节点间的实际通信带宽，帮助定位是网络问题还是模型配置问题。

**5. OOM（显存不足）**

多节点不意味着单卡显存充裕。需要合理配置 `max_batch_size`、`max_input_len`、`max_beam_width` 等参数，确保 KV cache 分配不超过剩余显存。TRT-LLM 的 `BuildConfig` 提供了 `max_num_tokens` 参数来控制显存使用上限。

## 本章小结

本章全面介绍了 TensorRT-LLM 的多节点部署方案。从 MPI 进程启动到 NCCL 跨节点通信初始化（`cpp/tensorrt_llm/runtime/worldConfig.cpp`），从环境变量配置到 InfiniBand GPU Direct RDMA 的性能优化，我们覆盖了多节点推理的完整技术栈。在生产部署层面，Triton Inference Server 的 Leader-Worker 架构提供了成熟的服务化方案，Docker 容器化部署需要特别注意网络和设备的挂载配置。通过带宽需求分析，我们理解了为什么节点间应优先使用 PP 而非 TP，以及不同推理阶段（prefill vs decode）对网络带宽的差异化需求。多节点部署是大模型推理的"最后一公里"，掌握这些知识将帮助读者在真实的数据中心环境中顺利完成超大规模模型的落地部署。
