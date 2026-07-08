# 环境变量说明文档

## 概述

本方案在 vLLM 推理服务优化中使用了以下自定义环境变量。这些变量在 `vllm/env_override.py` 中于模块加载时自动设置，无需用户手动 export。此处列出以供评审方参考。

---

## 环境变量清单

### 1. `VLLM_ROCM_USE_AITER`
- **取值**: `1`
- **作用**: 启用 AITER（AI Tensor Engine for ROCm）自定义算子库的总开关。AITER 提供了针对 DCU 硬件优化的高性能算子实现，包括 Attention、Linear、RMSNorm 等。
- **配置原因**: DCU 加速卡基于 ROCm 平台，AITER 算子库比默认的 Triton 或 HIP 实现具有更好的硬件适配性，能够充分利用 DCU 的 HBM 带宽和计算单元。

### 2. `VLLM_ROCM_USE_AITER_MHA`
- **取值**: `1`
- **作用**: 启用 AITER 的 Multi-Head Attention（MHA）Flash Attention 实现，用于统一处理 Prefill 和 Decode 阶段的注意力计算。
- **配置原因**: 默认的 PagedAttention 在长上下文场景下存在块查找开销。AITER 的 Flash Attention 实现使用更高效的访存模式，降低 HBM 访问次数。

### 3. `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION`
- **取值**: `1`
- **作用**: 启用 AITER 的统一 Attention 后端（ROCM_AITER_FA），在所有注意力层中使用统一的 kernel，避免 Prefill 和 Decode 之间的 kernel 切换开销。
- **配置原因**: 统一 Attention 调度可以减少 kernel launch 延迟，缩短 TTFT，同时对 Decode 阶段的 TPOT 也有优化效果。

### 4. `VLLM_ROCM_USE_AITER_LINEAR`
- **取值**: `1`
- **作用**: 启用 AITER 的 FP8 Linear 算子，用于模型中所有 Linear 层（QKV projection、O projection、MLP 等）的矩阵乘法。
- **配置原因**: FP8 Linear 算子在 DCU 上支持低精度矩阵乘法，配合动态量化可在几乎无损的前提下将计算带宽需求减半。

### 5. `VLLM_ROCM_USE_AITER_FP8BMM`
- **取值**: `1`
- **作用**: 启用 AITER 的 FP8 Batch Matrix Multiply（BMM），用于 Attention 打分矩阵计算。
- **配置原因**: FP8 BMM 在保持精度的同时减少 HBM 读写带宽。

### 6. `VLLM_ROCM_FP8_PADDING`
- **取值**: `1`
- **作用**: 启用 FP8 的 padding 对齐优化，确保 FP8 数据传输满足 DCU 内存对齐要求。
- **配置原因**: DCU 的 FP8 计算单元需要特定的数据对齐方式，开启此选项可以避免不必要的内存拷贝。

### 7. `VLLM_DCU_BLOCK_SIZE`
- **取值**: `32`
- **作用**: 指定 DCU 平台的 KV Cache block size（每个 block 包含的 token 数）。默认值为 16。
- **配置原因**: DCU 的 HBM 带宽特性对大块连续访存更友好。将 block size 从默认的 16 增大到 32，可以减少 block table 管理开销约 50%，同时保持较好的空间利用率。

---

## 自动配置机制

以上环境变量在 `vllm/env_override.py` 的 `_auto_configure_dcu()` 函数中自动检测并设置。检测逻辑为：

```python
is_rocm = os.path.exists("/opt/rocm") or os.environ.get("ROCM_HOME", "") != ""
```

当检测到 ROCm/DCU 环境时，上述环境变量自动生效，无需人工干预。在非 DCU 平台上（如 NVIDIA CUDA），这些变量不会设置，不影响原有功能。
