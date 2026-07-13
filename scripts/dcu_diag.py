#!/usr/bin/env python3
"""
DCU K100 诊断工具
在目标机器上运行，输出完整的环境信息。
用法：python dcu_diag.py
"""

import importlib
import json
import os
import subprocess
import sys

RESULTS = []


def section(title):
    RESULTS.append("")
    RESULTS.append("=" * 60)
    RESULTS.append(f"  {title}")
    RESULTS.append("=" * 60)


def info(key, value):
    RESULTS.append(f"  {key}: {value}")


def try_import(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        info(f"import {mod_name}", "✅ OK")
        return mod
    except Exception as e:
        info(f"import {mod_name}", f"❌ {e}")
        return None


def try_import_attr(mod_name, attr):
    mod = try_import(mod_name)
    if mod:
        if hasattr(mod, attr):
            info(f"  has {attr}", "✅")
            return True
        else:
            info(f"  has {attr}", "❌ missing")
            return False
    return False


# ========================
# 1. 系统环境
# ========================
section("1. 系统环境")
info("Python", sys.version)
info("架构", os.uname().machine)

# 环境变量
for var in [
    "ROCM_HOME", "ROCM_PATH", "HIP_PLATFORM", "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES", "VLLM_TARGET_DEVICE", "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MHA", "VLLM_ROCM_USE_AITER_LINEAR",
    "VLLM_ATTENTION_BACKEND",
]:
    info(f"${var}", os.environ.get(var, "(unset)"))

# ========================
# 2. DTK/ROCm 路径
# ========================
section("2. DTK/ROCm 路径")
for path in ["/opt/rocm", "/opt/dtk"] + [
    p for p in (os.environ.get("ROCM_HOME", "").split(":") if os.environ.get("ROCM_HOME") else [])
]:
    if os.path.exists(path):
        info(f"{path}", "✅ exists")
        for f in sorted(os.listdir(path))[:10]:
            info(f"  {f}", "")

# Wildcard DTK paths
import glob
for p in sorted(glob.glob("/opt/dtk-*")):
    info(f"{p}", "✅ exists")
    for f in sorted(os.listdir(p))[:5]:
        info(f"  {f}", "")

for p in sorted(glob.glob("/opt/rocm-*")):
    info(f"{p}", "✅ exists")

# ========================
# 3. HIP 和 GPU
# ========================
section("3. HIP/GPU")
# hipconfig
for cmd in ["hipconfig --platform", "hipconfig --arch", "hipconfig --rocminput"]:
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
        info(f"$ {cmd}", out)
    except Exception as e:
        info(f"$ {cmd}", f"❌ {e}")

# torch GPU info
torch = try_import("torch")
if torch:
    info("torch.version", torch.__version__)
    info("torch.version.cuda", torch.version.cuda)
    info("torch.version.hip", torch.version.hip)
    cuda_avail = torch.cuda.is_available()
    info("cuda.is_available", cuda_avail)
    if cuda_avail:
        info("device_count", torch.cuda.device_count())
        info("device_name", torch.cuda.get_device_name(0))
        cap = torch.cuda.get_device_capability(0)
        info("compute_capability", cap)
        try:
            props = torch.cuda.get_device_properties(0)
            info("total_memory_GB", f"{props.total_mem / 1e9:.1f}")
            info("gcnArchName", getattr(props, "gcnArchName", "N/A"))
        except Exception as e:
            info("get_device_properties", f"❌ {e}")

# ========================
# 4. AITER 可用性详细检测
# ========================
section("4. AITER 可用性")
aiter_mod = try_import("aiter")
if aiter_mod:
    info("aiter.__file__", aiter_mod.__file__)
    # List all aiter attributes
    aiter_attrs = [a for a in dir(aiter_mod) if not a.startswith("_")]
    info("aiter exports", str(aiter_attrs))
    
    # Test specific function imports
    for func in [
        "flash_attn_varlen_func", "gemm_a8w8_CK", "gemm_a8w8_blockscale",
        "rms_norm", "topk_softmax", "rmsnorm2d_fwd_with_add",
        "rmsnorm2d_fwd_with_dynamicquant",
    ]:
        try:
            getattr(aiter_mod, func)
            info(f"  aiter.{func}", "✅")
        except AttributeError:
            info(f"  aiter.{func}", "❌")

    # Check aiter submodules
    for sub in ["ops", "ops.triton", "ops.quant"]:
        try:
            sub_mod = importlib.import_module(f"aiter.{sub}")
            keep = [a for a in dir(sub_mod) if not a.startswith("_") and "flash" in a.lower() or "gemm" in a.lower() or "quant" in a.lower()]
            if keep:
                info(f"  aiter.{sub}", f"✅ has: {keep[:10]}")
            else:
                info(f"  aiter.{sub}", f"✅ ({len(dir(sub_mod))} items)")
        except Exception:
            info(f"  aiter.{sub}", "❌")

# vLLM's AITER ops
rocm_aiter = try_import("vllm._aiter_ops")
if rocm_aiter:
    for check_name, check_func in [
        ("is_aiter_found", lambda: rocm_aiter.is_aiter_found()),
        ("is_aiter_found_and_supported", lambda: rocm_aiter.is_aiter_found_and_supported()),
    ]:
        try:
            result = check_func()
            info(f"  {check_name}()", str(result))
        except Exception as e:
            info(f"  {check_name}()", f"❌ {e}")

    if hasattr(rocm_aiter, "rocm_aiter_ops"):
        ops = rocm_aiter.rocm_aiter_ops
        for chk in ["is_enabled", "is_linear_enabled", "is_linear_fp8_enabled",
                     "is_mha_enabled", "is_fused_moe_enabled", "is_rmsnorm_enabled"]:
            try:
                r = getattr(ops, chk)()
                info(f"  ops.{chk}()", str(r))
            except Exception as e:
                info(f"  ops.{chk}()", f"❌ {e}")

        # Check flash_attn_varlen_func
        if hasattr(ops, "flash_attn_varlen_func"):
            info("  ops.flash_attn_varlen_func", "✅ has method")
        else:
            info("  ops.flash_attn_varlen_func", "❌ missing")

# ========================
# 5. _rocm_C 扩展检测
# ========================
section("5. _rocm_C 扩展")
rocm_c = try_import("vllm._rocm_C")
if rocm_c:
    info("_rocm_C.__file__", rocm_c.__file__)
    rocm_c_ops = [a for a in dir(rocm_c) if not a.startswith("_")]
    info("_rocm_C exports", str(rocm_c_ops))
    
    # Check critical ops
    for op in ["wvSplitK", "wvSplitKrc", "wvSplitKQ", 
               "attention", "skinny_gemm"]:
        found = any(op.lower() in str(a).lower() for a in rocm_c_ops)
        info(f"  {op}", "✅" if found else "❌")

# ========================
# 6. ROCm 平台检测
# ========================
section("6. ROCm Platform")
plat = try_import("vllm.platforms")
if plat:
    info("current_platform", str(type(plat.current_platform)))
    info("device_name", plat.current_platform.device_name)
    info("device_type", plat.current_platform.device_type)
    if hasattr(plat.current_platform, "is_rocm"):
        info("is_rocm()", plat.current_platform.is_rocm())
    # Check rocm-specific functions
    rocm_mod = try_import("vllm.platforms.rocm")
    if rocm_mod:
        for fn_name in ["on_gfx9", "on_mi3xx", "on_gfx942", "on_gfx950"]:
            fn = getattr(rocm_mod, fn_name, None)
            if fn:
                info(f"  rocm.{fn_name}()", str(fn()))
        info("  _GCN_ARCH", getattr(rocm_mod, "_GCN_ARCH", "N/A"))
        info("  _ON_GFX9", getattr(rocm_mod, "_ON_GFX9", "N/A"))

# ========================
# 7. Torch 编译/算力测试
# ========================
section("7. Torch 编译/算力")
if torch and torch.cuda.is_available():
    # Test INT8 matmul
    try:
        a = torch.randn(1, 4096, device="cuda").to(torch.int8)
        b = torch.randn(4608, 4096, device="cuda").to(torch.int8)
        c = (a @ b.T)
        info("INT8 matmul (4096x4608)", f"✅ {c.shape}")
    except Exception as e:
        info("INT8 matmul", f"❌ {e}")

    # Test torch._scaled_mm
    try:
        a = torch.randn(1, 4096, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(4608, 4096, device="cuda").to(torch.float8_e4m3fn)
        c = torch._scaled_mm(a, b.T)
        info("_scaled_mm (FP8)", "✅")
    except Exception as e:
        info("_scaled_mm (FP8)", f"❌ {e}")

    # Test _rocm_C ops if available
    if rocm_c and "wvSplitK" in [str(a) for a in dir(rocm_c)]:
        try:
            x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(4608, 4096, device="cuda", dtype=torch.bfloat16)
            out = rocm_c.wvSplitK(x, w, 1)
            info("wvSplitK", f"✅ {out.shape}")
        except Exception as e:
            info("wvSplitK", f"❌ {e}")

# ========================
# 8. 编译环境检测
# ========================
section("8. 编译环境")
for cmd in ["cmake --version", "hipcc --version", "which cmake", "which hipcc"]:
    try:
        out = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT, timeout=5).decode().strip()[:100]
        info(f"$ {cmd}", out.split("\\n")[0])
    except Exception as e:
        info(f"$ {cmd}", f"❌ {e}")


# ========================
# 输出
# ========================
print("\\n".join(RESULTS))
print("\\n" + "=" * 60)
print("  诊断完成")
print("=" * 60)
