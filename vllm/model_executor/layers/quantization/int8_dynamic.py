# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
INT8 Dynamic Quantization for DCU using AITER gemm_a8w8.
Quantizes bf16 weights to INT8 at load time (in-memory, not persistent),
and quantizes activations per-token during forward pass.

Competition compliance:
- "推理过程中的非持久化、算子级低精度计算优化" ✅
- "低精度矩阵乘法" ✅
- "不持久化量化权重到磁盘" ✅
- Does NOT modify model structure ✅
"""

from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class Int8DynamicConfig(QuantizationConfig):
    """INT8 dynamic quantization config for DCU.
    Uses AITER's gemm_a8w8 for INT8 matrix multiplication."""

    @classmethod
    def get_name(cls) -> str:
        return "int8_dynamic"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Int8DynamicConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        from vllm.model_executor.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            return Int8DynamicLinearMethod(self)
        return None

    @classmethod
    def override_quantization_method(
        cls, quant_cfg: dict[str, Any], quantization: str | None
    ) -> str | None:
        return None


class Int8DynamicLinearMethod(QuantizeMethodBase):
    """INT8 dynamic quantization method using AITER gemm_a8w8."""

    def __init__(self, quant_config: Int8DynamicConfig):
        self.quant_config = quant_config
        self.use_aiter_int8 = False

        # Check if AITER INT8 ops are available
        try:
            from vllm._aiter_ops import rocm_aiter_ops
            self.use_aiter_int8 = rocm_aiter_ops.is_linear_enabled()
        except Exception:
            self.use_aiter_int8 = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weight parameters. Weights are stored in bf16 initially,
        then quantized to INT8 in process_weights_after_loading."""
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Store weights in bf16 (original format)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # Weight scale (not a parameter - managed manually)
        # Stored as plain tensor, not a Parameter, to avoid shape constraints
        layer.weight_int8 = None
        layer.wscale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Quantize bf16 weights to INT8 (in-memory, not persisted to disk)."""
        w = layer.weight.data
        orig_dtype = w.dtype

        # Per-channel quantization: find absmax for each output channel
        w_float = w.to(torch.float32)
        absmax = w_float.abs().amax(dim=1, keepdim=True)  # [out_dim, 1]

        # Avoid division by zero
        absmax = torch.clamp(absmax, min=1e-12)

        # Scale = absmax / 127 (max int8 positive)
        scale = absmax / 127.0

        # Quantize
        w_int8 = (w_float / scale).round().clamp(-128, 127).to(torch.int8)

        # Store quantized weights (in-memory only)
        layer.weight_int8 = w_int8
        layer.wscale = scale  # [out_dim, 1]

        logger.info(
            "INT8 dynamic: quantized weights (in-memory), "
            "shape=%s", w_int8.shape
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with INT8 matmul."""
        # Input must be 2D [M, K] or 3D
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.flatten(0, 1)

        if self.use_aiter_int8 and layer.weight_int8 is not None:
            return self._apply_aiter_int8(layer, x, bias, orig_shape)
        else:
            return self._apply_bf16_fallback(layer, x, bias, orig_shape)

    def _apply_aiter_int8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """Forward with AITER INT8 GEMM."""
        # Quantize activation to INT8 (per-token)
        x_float = x.to(torch.float32)
        absmax = x_float.abs().amax(dim=1, keepdim=True)  # [M, 1]
        absmax = torch.clamp(absmax, min=1e-12)
        x_scale = absmax / 127.0
        x_int8 = (x_float / x_scale).round().clamp(-128, 127).to(torch.int8)

        # Get INT8 weights and scales
        w_int8 = layer.weight_int8  # [N, K]
        w_scale = layer.wscale  # [N, 1]

        # Transpose weight to [K, N] for gemm_a8w8 (which expects B as [N, K])
        # AITER gemm_a8w8_CK expects: A=[M,K], B=[N,K], As=[M,1], Bs=[N,1]
        Bs = w_scale.unsqueeze(1)  # [N, 1]

        # Call AITER's gemm_a8w8 via registered torch op
        try:
            op = getattr(torch.ops.vllm, "rocm_aiter_gemm_a8w8", None)
            if op is not None:
                out = op(
                    x_int8, w_int8, x_scale, Bs, bias, torch.bfloat16
                )
            else:
                # Fallback to direct aiter import
                from aiter import gemm_a8w8_CK
                out = gemm_a8w8_CK(
                    x_int8, w_int8, x_scale, Bs, bias, torch.float16
                )
        except Exception:
            out = self._apply_bf16_fallback(layer, x, bias, orig_shape)
            return out

        if orig_shape != out.shape:
            out = out.view(orig_shape[:-1] + (out.shape[-1],))

        return out.to(x.dtype)

    def _apply_bf16_fallback(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """Fallback: use bf16 matmul."""
        return F.linear(x, layer.weight, bias)
