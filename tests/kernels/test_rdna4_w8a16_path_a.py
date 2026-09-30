#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Correctness for gfx1201 W8A16 Path A (bf16 WMMA; no iu8 atom required)."""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.rdna4_w8a16_path_a import (  # noqa: E402
    pick_tile_config,
    w8a16_gemm_path_a,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

ARCH = str(get_rocm_arch() or "")
if not ARCH.startswith("gfx120"):
    pytest.skip(f"W8A16 Path A requires gfx120x, got {ARCH}", allow_module_level=True)


def _ref(a, b_nk, w_scale):
    # Offline dequant ref (Path A contract — not HIP dyn-quant bits).
    w = b_nk.float() * w_scale.reshape(1, -1)
    return (a.float() @ w.T).to(a.dtype)


@pytest.mark.parametrize(
    "m,n,k",
    [
        pytest.param(32, 64, 64, id="tiny-32x64x64"),
        pytest.param(64, 64, 64, id="64x64x64"),
        pytest.param(37, 70, 80, id="ragged-37x70x80"),
    ],
)
def test_w8a16_path_a_vs_offline_ref(m, n, k):
    torch.manual_seed(20260930 + m + n + k)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
    ws = (torch.rand(n, device="cuda", dtype=torch.float32) * 0.01 + 0.001).contiguous()
    out = w8a16_gemm_path_a(a, b, ws, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    ref = _ref(a, b, ws)
    torch.testing.assert_close(out, ref, rtol=4e-2, atol=4e-2)


def test_w8a16_path_a_tile_pick():
    cfg = pick_tile_config(32, 64, 64, wgps=48)
    assert cfg.bm in (64, 128, 256)
    assert cfg.bk == 64
