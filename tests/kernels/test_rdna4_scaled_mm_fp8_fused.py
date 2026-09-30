#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness tests for fused act-quant ⊕ FP8 scaled_mm (+ LoRA) on gfx1201."""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna4_scaled_mm_fp8_fused import (  # noqa: E402
    reference_scaled_mm_fp8_fused,
    scaled_mm_fp8_fused,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(
        f"RDNA4 fused FP8 scaled-MM requires gfx120x, got {ARCH}",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(128, 128, 128, id="128x128x128"),
        pytest.param(130, 144, 128, id="ragged-130x144x128"),
    ],
)
def test_fused_act_quant_scaled_mm(m, n, k):
    """Fused path matches separate quant + scaled_mm reference."""
    torch.manual_seed(17)
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_f32 = torch.randn((n, k), device="cuda", dtype=torch.float32).clamp(-1, 1)
    b_nk = b_f32.to(torch.float8_e4m3fn).contiguous()
    scale_a = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.25], device="cuda", dtype=torch.float32)

    out = scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = reference_scaled_mm_fp8_fused(
        a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.15)


@pytest.mark.parametrize("rank", [8, 16])
def test_fused_scaled_mm_lora_epilogue(rank):
    """Single-adapter LoRA epilogue matches separate quant+mm + (x@A.T)@B.T."""
    torch.manual_seed(23)
    m = n = k = 64
    a_f = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).clamp(-1, 1)
    b_nk = (
        torch.randn((n, k), device="cuda", dtype=torch.float32)
        .clamp(-1, 1)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    scale_a = torch.tensor([0.8], device="cuda", dtype=torch.float32)
    scale_b = torch.tensor([1.1], device="cuda", dtype=torch.float32)
    lora_down = torch.randn((rank, k), device="cuda", dtype=torch.bfloat16) * 0.02
    lora_up = torch.randn((n, rank), device="cuda", dtype=torch.bfloat16) * 0.02
    lora_scale = 0.5

    out = scaled_mm_fp8_fused(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_down=lora_down,
        lora_up=lora_up,
        lora_scale=lora_scale,
    )
    torch.cuda.synchronize()
    ref = reference_scaled_mm_fp8_fused(
        a_f,
        b_nk,
        scale_a,
        scale_b,
        out_dtype=torch.bfloat16,
        lora_down=lora_down,
        lora_up=lora_up,
        lora_scale=lora_scale,
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.05, atol=0.2)


def test_reference_helper_cpu_math():
    """Pure torch reference stays consistent without launching the kernel."""
    torch.manual_seed(3)
    a = torch.randn(32, 32, dtype=torch.bfloat16)
    b = torch.randn(32, 32).clamp(-1, 1).to(torch.float8_e4m3fn)
    sa = torch.tensor([1.0])
    sb = torch.tensor([1.0])
    down = torch.randn(8, 32, dtype=torch.bfloat16) * 0.01
    up = torch.randn(32, 8, dtype=torch.bfloat16) * 0.01
    out = reference_scaled_mm_fp8_fused(
        a, b, sa, sb, lora_down=down, lora_up=up, lora_scale=1.0
    )
    assert out.shape == (32, 32)
    assert out.dtype == torch.bfloat16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
