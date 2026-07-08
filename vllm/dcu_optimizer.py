"""
DCU Optimizer for CSCC Competition
====================================
Central optimization module for DCU (K100) accelerator.
Aggregates all DCU-specific optimizations for Qwen3.5-27B inference.

Competition: 先导杯 — 基于国产加速卡的千问大模型推理服务优化
Framework: vLLM 0.18.1
Model: Qwen3.5-27B (bf16, 32K context)
Hardware: DCU (K100, 1000 GB/s HBM, 480 BF16 TFLOPS)
Concurrency: 1

Modification boundaries (per competition rules):
  ✅ KV cache management & block allocation
  ✅ Decode stage scheduler customization
  ✅ Operator fusion & kernel optimization for DCU
  ✅ KV Cache quantization (fp8)
  ✅ Activation dynamic quantization
  ✅ Low-precision matrix multiplication (kernel-level)
  ❌ No persistent weight quantization (in-memory only)
  ❌ No batch scheduler code changes
  ❌ No model structure changes
  ❌ No speculative decoding
"""

import os
import torch
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def get_dcu_optimized_config() -> dict[str, Any]:
    """
    Returns a dict of DCU-optimized vLLM configuration overrides.
    These are applied at engine initialization time.

    Priority: maximize throughput under SLA constraints (TTFT P99 ≤ 1.5x baseline,
    TPOT P99 ≤ 1.5x baseline).

    Three input length tiers: 4K-8K (20%), 8K-16K (50%), 16K-32K (30%)
    Single request (concurrency=1).
    """
    return {
        # ---- KV Cache ----
        # Block size: 32 (vs default 16)
        # - Reduces block management overhead by 50%
        # - Better HBM alignment for DCU's memory hierarchy
        # - 32K context → 1024 blocks (vs 2048 with block_size=16)
        "block_size": 32,

        # KV Cache dtype: fp8 (vs bf16)
        # - Halves KV cache HBM reads → directly improves TPOT
        # - KV cache is ~4.3GB at 32K → ~2.1GB with fp8
        # - Allows more blocks for long context
        "kv_cache_dtype": "fp8",

        # Calculate KV scales dynamically (not from checkpoint)
        "calculate_kv_scales": True,

        # GPU memory utilization: max possible for single-request
        # Dedicated card, no other processes competing for HBM
        "gpu_memory_utilization": 0.95,

        # ---- Prefix Caching ----
        # Disable prefix caching for single-request scenario
        # No sharing between concurrent requests → pure overhead
        # Saves block hash computation and cache lookup time
        "enable_prefix_caching": False,

        # ---- Scheduling ----
        # Max model length: 32K as required by competition
        # (must match --max-model-len 32768)
        "max_model_len": 32768,

        # ---- Attention ----
        # Force ROCm attention backend for optimal DCU performance
        # Available backends: ROCM_ATTN, ROCM_AITER_FA, FLASH_ATTN
        # ROCM_AITER_FA is preferred for unified prefill+decode on DCU
        "attention_backend": "ROCM_AITER_FA",
    }


def apply_dcu_patch():
    """
    Apply DCU-specific runtime patches.
    Called once at engine startup before model loading.

    Current patches:
    1. Disable unnecessary logging
    2. Optimize torch settings for DCU
    3. Set environment variables for DCU kernel tuning
    """
    # ---- PyTorch optimizations for DCU ----
    torch.set_num_threads(1)  # Avoid thread contention on CPU side

    # Enable TF32 for linear layers (TF32 is 8x faster than FP32 and more accurate)
    # DCU has 240 TF32 TFLOPS vs 60 FP32 TFLOPS
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Environment variables for DCU kernel tuning ----
    # Force ROCm attention backend
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "ROCM_AITER_FA")

    # Disable flash_attn for prefilling when using ROCM_AITER_FA
    os.environ.setdefault("VLLM_ROCM_USE_AITER_FA", "1")

    # Enable FP8 linear ops on DCU
    os.environ.setdefault("VLLM_ROCM_FP8_LINEAR", "1")

    # Disable CUDA graph logging noise
    os.environ.setdefault("VLLM_LOG_CUDA_GRAPH", "0")

    logger.info("[DCU Optimizer] Applied DCU runtime patches")
    logger.info("[DCU Optimizer] Precision: FP8 W8A8 dynamic (weights: bf16→fp8 in-kernel)")
    logger.info("[DCU Optimizer] KV Cache: fp8, Block Size: 32")
    logger.info("[DCU Optimizer] Attention Backend: ROCM_AITER_FA")
    logger.info("[DCU Optimizer] Prefix Caching: disabled (single-request)")


def get_env_setup_cmd() -> str:
    """
    Returns shell commands to set up DCU environment variables.

    Usage:
        eval "$(python -c 'from vllm.dcu_optimizer import get_env_setup_cmd; print(get_env_setup_cmd())')"
    """
    return "\n".join([
        'export VLLM_ROCM_USE_AITER=1',
        'export VLLM_ROCM_USE_AITER_LINEAR=1',
        'export VLLM_ROCM_USE_AITER_MHA=1',
        'export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1',
        'export VLLM_ROCM_USE_AITER_RMSNORM=1',
        'export VLLM_ROCM_USE_AITER_FP8BMM=1',
        'export VLLM_ROCM_CUSTOM_PAGED_ATTN=1',
        'export VLLM_DCU_BLOCK_SIZE=32',
        'export VLLM_ATTENTION_BACKEND=ROCM_AITER_FA',
    ])


def get_serve_command() -> str:
    """
    Returns the recommended vllm serve command for DCU-optimized inference.

    Usage:
        eval "$(python -c 'from vllm.dcu_optimizer import get_serve_command; print(get_serve_command())')"
    """
    return (
        "vllm serve Qwen/Qwen3.5-27B "
        "--max-model-len 32768 "
        "--block-size 32 "
        "--kv-cache-dtype fp8 "
        "--quantization fp8 "
        "--activation-scheme dynamic "
        "--enable-prefix-caching=false "
        "--gpu-memory-utilization 0.95 "
        "--max-num-seqs 1 "
        "--attention-backend rocm_aiter_fa "
        "-tp 1 "
        "--temperature 0 "
        "--max-tokens 32768 "
        "--served-model-name Qwen3.5-27B"
    )
