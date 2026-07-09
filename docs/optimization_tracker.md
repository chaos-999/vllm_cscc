# DCU 推理服务优化文档
# 优化点汇总

## 已实现优化

### O1: 默认 Block Size 从 16 增大到 32
- **文件**: `vllm/config/cache.py`
- **原理**: 对于 32K 上下文，block 数量从 2048 减至 1024，block table 管理开销减半
- **收益**: 中
- **状态**: ✅ 已提交

### O2: 默认禁用 Prefix Caching（单请求场景）
- **文件**: `vllm/config/cache.py`
- **原理**: 单请求并发=1 时，Prefix Caching 无收益，反而带来 block hash 计算和缓存查找开销
- **收益**: 低-中
- **状态**: ✅ 已提交

### O3: FP8 动态量化 (W8A8)
- **文件**: `vllm/config/model.py`
- **原理**: 权重运行时动态量化为 fp8（不持久化到磁盘），矩阵乘法带宽需求减半
- **收益**: 高
- **状态**: ✅ 已提交（需 ROCm 平台检测通过）

### O4: KV Cache FP8 量化
- **文件**: `vllm/config/cache.py`
- **原理**: KV cache 从 bf16 转为 fp8，存储和带宽减半
- **收益**: 中
- **状态**: ✅ 已提交（需 ROCm 平台检测通过）

### O5: AITER 算子库启用
- **文件**: `vllm/env_override.py`
- **原理**: 启用 DCU 专用的优化算子（Attention、Linear、RMSNorm）
- **收益**: 中-高
- **状态**: ✅ 已提交（需 ROCm 平台检测通过）

### O6: ROCm 平台检测优化
- **文件**: `vllm/config/cache.py`, `vllm/config/model.py`, `vllm/env_override.py`
- **原理**: 使用 torch.version.hip + /opt/dtk 路径 + ROCM_HOME 三路检测
- **状态**: ✅ 已提交

### O7: 本地模型路径兼容修复
- **文件**: `vllm/transformers_utils/config.py`, `vllm/tokenizers/hf.py`
- **原理**: 绕过 transformers 5.5.0 的 HuggingFace Hub repo_id 校验
- **状态**: ✅ 已提交

---

## 首次测评结果 (60.09/100)

| 指标 | Baseline | 优化后 | 提升 |
|------|----------|--------|------|
| 4-8K 输出吞吐 | ~12.18 tok/s | 12.94 tok/s | +6.2% |
| 8-16K 输出吞吐 | ~8.79 tok/s | 10.07 tok/s | +14.6% |
| 16-32K 输出吞吐 | - | 5.78 tok/s | - |
| SLA 扣分 | 0 | 0 | ✅ |
| 精度扣分 | 0 | 0 | ✅ |

**分析**: 优化提升有限，原因可能是 FP8 量化未正确生效（检测未通过）。
已修复检测逻辑，需重新测评验证。

---

## 待实现优化

### T1: KV Cache 预分配（单请求场景）
- **描述**: 并发=1 时，在请求到达时一次性分配 max_model_len/block_size 个 block
- **文件**: `vllm/v1/core/single_type_kv_cache_manager.py`
- **预期收益**: 中

### T2: 单请求调度路径精简
- **描述**: 跳过不需要的多请求调度决策（优先级排队等）
- **文件**: `vllm/v1/core/sched/scheduler.py`
- **预期收益**: 低-中
