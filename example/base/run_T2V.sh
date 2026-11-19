#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

export MASTER_ADDR="${MASTER_ADDR:-localhost}" # 单机多卡时候，所有进程都是运行到一个机器上的，localhost，如果是多机训练或者推理的时候就需要些主节点的IP地址。
export MASTER_PORT="${MASTER_PORT:-6013}"  # 主节点通信端口
export NNODES="${NNODES:-1}"  # 只有一台机器
export NODE_RANK="${NODE_RANK:-0}"  # 当前机器编号
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"  # 每台机器使用1张GPU，如果是2-8就是使用对应数字的GPU
export WORLD_SIZE="$((GPUS_PER_NODE * NNODES))"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # pytorch的cuda内存分配模式，可以减少显存碎片
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"  # 把项目根目录加入python模块的搜索路径
# rdzv 是 rendezvous 的缩写，表示分布式进程集合和初始化过程，c10d 是 PyTorch 使用的分布式通信后端之一，用来让不同进程互相发现并建立通信。后面就是表示进程初始化时连接的地址和端口。
DISTRIBUTED_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${GPUS_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT}"

# ==============================================================================================
# RUNNING ON CONSUMER GPUs (e.g., RTX 5090)
# ==============================================================================================
# If you want to run this script on a consumer GPU, please follow these steps to avoid OOM errors:
#
# 1. Define MAGI_COMPILER_OFFLOAD_ARGS and append it to the `torchrun` command below.
# 2. Update `engine_config.cp_size` in `config.json` to exactly match the number of GPUs on your machine.
# 3. Depending on your NUMA node configuration, use `numactl` as a prefix to optimize memory bandwidth:
#    - If spanning multiple NUMA nodes: `numactl --interleave=all`
#    - If on a single NUMA node:        `numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE`
#
# --- Example Usage ---
# MAGI_COMPILER_OFFLOAD_ARGS="--offload_config.model_cpu_offload --offload_config.gpu_resident_weight_ratio 0.35 --offload_config.offload_policy HEURISTIC"
# numactl --interleave=all torchrun ${DISTRIBUTED_ARGS} inference/pipeline/entry.py ... $MAGI_COMPILER_OFFLOAD_ARGS
# ==============================================================================================

torchrun ${DISTRIBUTED_ARGS} inference/pipeline/entry.py \
  --config-load-path example/base/config.json \
  --prompt "$(<example/assets/prompt.txt)" \
  --seconds 4 \
  --br_width 448 \
  --br_height 256 \
  --output_path "output_example_base_t2v_$(date '+%Y%m%d_%H%M%S')" \
  2>&1 | tee "log_example_base_t2v_$(date '+%Y%m%d_%H%M%S').log"
