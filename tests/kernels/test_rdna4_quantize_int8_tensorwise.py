#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness for gfx120x tensorwise INT8 quantize (single absmax scale)."""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.quant.rdna4_quantize_int8_tensorwise import (  # noqa: E402
    quantize_int8_tensorwise,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"int8 tensorwise requires gfx120x, got {ARCH}", allow_module_level=True)


def _torch_ref(x: torch.Tensor):
    amax = x.detach().float().abs().amax()
    scale = torch.clamp(amax / 127.0, min=1e-30)
    q = torch.round(x.float() / scale).clamp(-128, 127).to(torch.int8)
    return q, scale.reshape(())


def _hip_ref(x: torch.Tensor):
    try:
        import comfy_kitchen as ck
    except ImportError:
        return None
    q, scale = ck.quantize_int8_tensorwise(x)
    return q, scale


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(8, 64), (64, 256), (3, 7), (77, 128)])
def test_quantize_int8_tensorwise(shape, dtype):
    torch.manual_seed(20260930 + sum(shape) + (0 if dtype is torch.float32 else 1))
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    q, s = quantize_int8_tensorwise(x)
    torch.cuda.synchronize()
    assert q.shape == x.shape and q.dtype == torch.int8
    assert s.shape == () and s.dtype == torch.float32 and s.device.type == x.device.type
    qr, sr = _torch_ref(x)
    scale_maxdiff = (s.float() - sr.float()).abs().item()
    assert scale_maxdiff < 1e-5, f"scale_maxdiff={scale_maxdiff} got={s.item()} ref={sr.item()}"
    mism = (q != qr).sum().item()
    frac = mism / q.numel()
    assert frac < 0.02, f"q mismatch frac={frac} mism={mism}"
    hip = _hip_ref(x)
    if hip is not None:
        qh, sh = hip
        torch.cuda.synchronize()
        hip_scale = (s.float() - sh.float().reshape(())).abs().item()
        hip_mism = (q != qh).sum().item()
        assert hip_scale < 1e-5, f"hip scale diff={hip_scale}"
        assert hip_mism / q.numel() < 0.02, f"hip q mism={hip_mism}"


def test_supplied_scale_matches_eager():
    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    scale = torch.tensor(0.05, device="cuda", dtype=torch.float32)
    q, s = quantize_int8_tensorwise(x, scale=scale)
    # Kitchen eager divides in the activation dtype (fp16 here), not fp32.
    ref = torch.round(x / scale.to(dtype=x.dtype)).clamp(-128, 127).to(torch.int8)
    assert torch.equal(q, ref)
    assert abs(s.item() - 0.05) < 1e-8


def test_stochastic_rejected():
    x = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        quantize_int8_tensorwise(x, stochastic_rounding=1)
