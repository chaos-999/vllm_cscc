#!/bin/bash
# DCU-optimized vLLM serve script for CSCC Competition
# Model: Qwen3.5-27B | Hardware: DCU K100
#
# Usage:
#   ./scripts/dcu_serve.sh [additional_args...]

set -euo pipefail

echo "===== DCU Optimized vLLM Server ====="
echo "Model: Qwen3.5-27B"
echo "Hardware: DCU (1000 GB/s HBM)"
echo ""

# ---- DCU Environment Setup ----
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MHA=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_FP8BMM=1
export VLLM_ROCM_CUSTOM_PAGED_ATTN=1
export VLLM_DCU_BLOCK_SIZE=32
export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA

# Performance tuning
export VLLM_ROCM_FP8_PADDING=1
export VLLM_ROCM_MOE_PADDING=1
export VLLM_ROCM_SKINNY_GEMM=1

echo "[Env] AITER enabled (linear, MHA, RMSNorm, FP8BMM)"
echo "[Env] Attention backend: ROCM_AITER_FA"
echo "[Env] Block size: 32"
echo ""

# ---- Launch vLLM Server ----
vllm serve Qwen/Qwen3.5-27B \
    --max-model-len 32768 \
    --block-size 32 \
    --kv-cache-dtype fp8 \
    --quantization fp8 \
    --activation-scheme dynamic \
    --enable-prefix-caching false \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 1 \
    --attention-backend rocm_aiter_fa \
    -tp 1 \
    --temperature 0 \
    --max-tokens 32768 \
    --served-model-name Qwen3.5-27B \
    "$@"
