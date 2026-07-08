# 优化方案说明文档

## CSCC 先导杯 — 基于国产加速卡的千问大模型推理服务优化

---

## 1. 赛题概述

- **赛题**: 基于国产加速卡（DCU）的 Qwen3.5-27B 推理服务优化
- **框架**: vLLM 0.18.1（固定版本）
- **模型**: Qwen3.5-27B (bf16, 32K 上下文)
- **硬件**: DCU 单卡（HBM 带宽 1000 GB/s, BF16 算力 480 TFLOPS）
- **并发**: 固定为 1（单请求在线推理）
- **评测**: LongBench（问答/摘要）+ RULER（检索/聚合）混合数据集
- **评分**: 三档输入长度 4K-8K(20%) / 8K-16K(50%) / 16K-32K(30%)，按输出吞吐量评分

---

## 2. 优化技术路线

### 核心瓶颈分析

| 项目 | 计算 |
|------|------|
| 模型权重大小 (bf16) | ~54 GB |
| KV Cache 大小 (@32K) | ~4.3 GB |
| 每次 Decode 读取 HBM | ~58 GB |
| 理论最小 TPOT | ~58 ms |
| 瓶颈定位 | **Memory-bound**，95%+ 消耗在权重读取 |

Decode 阶段算术强度远低于 DCU 的算力/带宽比（0.48 FLOP/byte），因此优化核心是**减少 HBM 读取量**。

### 优化方法总览

本方案采用以下四大类优化措施：

| 类别 | 优化点 | 预期收益 |
|------|--------|----------|
| A. 量化优化 | FP8 动态量化 (W8A8) | 权重带宽减半，TPOT ~40%+ |
| B. KV Cache 优化 | FP8 存储 + Block Size 调整 | KV 带宽减半，碎片减少 |
| C. 算子优化 | AITER 统一 Attention + FP8 Linear | kernel 延迟降低 |
| D. 平台适配 | DCU 自动检测 + 环境变量配置 | 零人工干预 |

---

## 3. 各项优化措施详述

### 3.1 FP8 动态量化（W8A8）

**技术方案**：
- 利用 vLLM 内置的 `Fp8OnlineLinearMethod`，在模型加载时将 bf16 权重动态量化为 fp8（仅内存中，不写回磁盘）
- 激活值采用 per-token 动态量化，每个 token 独立计算 scale
- 所有 Linear 层（QKV projection、O projection、gate/up/down projection）使用 fp8 矩阵乘法

**合规性**：
- ✅ "推理过程中的非持久化、算子级低精度计算优化"（第7条(2)）
- ✅ "kernel 内部临时类型转换"（第7条(2)）
- ✅ "低精度矩阵乘法"（第7条(2)）
- ❌ 未持久化量化权重到磁盘（遵守第7条(2)禁止）
- ❌ 未修改模型结构（遵守第7条(1)禁止）

**性能贡献**：
- 权重读取从 ~54 GB → ~27 GB（减半）
- Decode 阶段 HBM 带宽需求降低约 50%

**实现文件**：
- `vllm/config/model.py` — `_verify_quantization()` 中自动启用 fp8
- `vllm/model_executor/layers/quantization/fp8.py` — `Fp8OnlineLinearMethod`
- `vllm/model_executor/layers/quantization/input_quant_fp8.py` — `QuantFP8`

### 3.2 KV Cache FP8 量化

**技术方案**：
- 将 KV Cache 存储格式从 bf16（16-bit）改为 fp8（8-bit）
- 使用 per-tensor dynamic scaling 保证精度
- 在 `CacheConfig` 中默认启用

**合规性**：
- ✅ "KV Cache 量化"（第7条(2)明确允许）
- ✅ 非持久化，仅在运行时生效

**性能贡献**：
- KV Cache 存储减半：@32K 从 ~4.3 GB → ~2.1 GB
- Decode 阶段 KV Cache 读取 HBM 带宽减半

**实现文件**：
- `vllm/config/cache.py` — `_validate_cache_dtype()` 中自动选择 fp8

### 3.3 Block Size 优化

**技术方案**：
- 将 PagedAttention 的 block size 从默认 16 增大到 32
- 32K 上下文：block 数量从 2048 减至 1024（减少 50%）

**性能贡献**：
- block table 管理开销减半
- DCU HBM 对齐效率提升
- Page fault 和 eviction 频率降低

**实现文件**：
- `vllm/config/cache.py` — `_apply_block_size_default()` 中设置

### 3.4 AITER 统一 Attention

**技术方案**：
- 使用 DCU 专用的 AITER 算子库，启用 `ROCM_AITER_FA` attention 后端
- 统一处理 Prefill 和 Decode 阶段的注意力计算
- 启用 AITER 的 FP8 Attention kernel

**性能贡献**：
- 减少 kernel launch 延迟
- 优化长上下文场景下的 attention 访存模式
- 减少 Python 层调度开销

**实现文件**：
- `vllm/env_override.py` — `_auto_configure_dcu()` 中设置环境变量

### 3.5 本地路径兼容性修复

**技术方案**：
- 修复 `maybe_override_with_speculators()` 对本地模型路径的兼容性
- 修复 `HFConfigParser.parse()` 和 `get_config()` 对本地路径的处理
- 修复 tokenizer 加载对本地路径的兼容性

**背景**：
竞赛容器中 transformers 5.5.0 对新版 huggingface_hub 的 repo_id 格式校验导致绝对路径（如 `/public/home/xxx/Qwen3.5-27B`）被错误拒绝。本方案在 vLLM 层面绕过该校验，直接加载本地配置文件。

**实现文件**：
- `vllm/transformers_utils/config.py`
- `vllm/tokenizers/hf.py`
- `vllm/engine/arg_utils.py`

---

## 4. 优化点汇总表

| 编号 | 优化点 | 分类 | 涉及文件 | 收益评估 |
|------|--------|------|----------|----------|
| O1 | FP8 动态量化 (W8A8) | 量化 | `vllm/config/model.py`, `vllm/model_executor/layers/quantization/fp8.py` | 高 |
| O2 | KV Cache FP8 存储 | KV 管理 | `vllm/config/cache.py` | 中 |
| O3 | Block Size 16→32 | KV 管理 | `vllm/config/cache.py` | 低-中 |
| O4 | AITER 统一 Attention | 算子 | `vllm/env_override.py` | 中-高 |
| O5 | AITER FP8 Linear 算子 | 算子 | `vllm/env_override.py` | 中 |
| O6 | 本地路径兼容修复 | 基础 | `vllm/transformers_utils/config.py`, `vllm/tokenizers/hf.py` | 必要 |
| O7 | DCU 自动检测 | 平台 | `vllm/env_override.py`, `vllm/config/cache.py`, `vllm/config/model.py` | 必要 |

---

## 5. 关键代码路径说明

### 5.1 自动配置入口
```
vllm/env_override.py → _auto_configure_dcu()  # 模块加载时自动检测 DCU
```

### 5.2 量化路径
```
模型加载 → ModelConfig._verify_quantization()
         → Fp8Config.get_quant_method() → Fp8OnlineLinearMethod
         → process_weights_after_loading() → ops.scaled_fp8_quant()
```

### 5.3 KV Cache 路径
```
CacheConfig.__init__ → _validate_cache_dtype() → cache_dtype = "fp8"
                     → _apply_block_size_default() → block_size = 32
```

### 5.4 Attention 路径
```
env_override → VLLM_ROCM_USE_AITER=1
             → rocm.py → _get_backend_priorities() → ROCM_AITER_FA
             → attention/selector.py → get_attn_backend()
```

---

## 6. 性能对比数据

*（待补充：在 DCU 上跑 Baseline 与优化后服务的对比结果）*

| 指标 | Baseline | 优化后 | 提升 |
|------|----------|--------|------|
| 4-8K Output 吞吐 | - | - | - |
| 8-16K Output 吞吐 | - | - | - |
| 16-32K Output 吞吐 | - | - | - |
| TPOT P99 | - | - | - |
| TTFT P99 (4-8K) | - | - | - |
| TTFT P99 (8-16K) | - | - | - |
| TTFT P99 (16-32K) | - | - | - |
