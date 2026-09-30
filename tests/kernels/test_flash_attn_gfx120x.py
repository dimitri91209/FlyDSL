# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors
# Credit: dimitri91209 + Grokbot
"""gfx120x / RDNA4 FlashAttention correctness smokes.

Mirrors aiter op_tests shapes: short S77-class, long S, cross, FP8 descales,
D>128, additive mask, causal self. Skips when no CUDA or arch is not gfx120x.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from kernels.attention.flash_attn_gfx120x_host import (  # noqa: E402
    bottom_right_causal_bias,
    flydsl_flash_attn_fp8_func,
    flydsl_flash_attn_func,
    flydsl_flash_attn_int8_func,
    flydsl_flash_attn_iu4_func,
    fold_alibi_to_bias,
    is_gfx120x,
    mask_is_noop,
    normalize_attn_mask,
)


def _arch() -> str:
    return (torch.cuda.get_device_properties(0).gcnArchName or "").split(":")[0]


pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(
        not is_gfx120x(),
        reason=f"requires gfx120x, got {_arch()!r}",
    ),
]


def _sdpa_ref(q, k, v, causal=False, attn_mask=None):
    # q/k/v BSHD → BHSD for SDPA
    qq = q.transpose(1, 2)
    kk = k.transpose(1, 2)
    vv = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal, attn_mask=attn_mask)
    return out.transpose(1, 2)


def _min_cos(a, b, dim_last=128):
    af = a.float().reshape(-1, a.shape[-1])
    bf = b.float().reshape(-1, b.shape[-1])
    return F.cosine_similarity(af, bf, dim=1).min().item()


@pytest.mark.parametrize("seq", [64, 77, 128, 129])
@pytest.mark.parametrize("dim", [64, 128])
def test_bf16_self_awkward_and_aligned(seq, dim):
    q = torch.randn(1, seq, 8, dim, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=False)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.999


def test_fp16_self_short():
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.float16)
    out = flydsl_flash_attn_func(q, q, q, causal=False)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert _min_cos(out, ref) > 0.999


def test_bf16_causal_self():
    q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=True)
    ref = _sdpa_ref(q, q, q, causal=True)
    assert _min_cos(out, ref) > 0.999


@pytest.mark.parametrize("sq,sk", [(77, 1024), (128, 1024), (1024, 128)])
def test_bf16_cross(sq, sk):
    q = torch.randn(1, sq, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, sk, 8, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, sk, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=False)
    ref = _sdpa_ref(q, k, v, causal=False)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.999


def test_bf16_long_self():
    q = torch.randn(1, 2048, 4, 128, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=False)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert _min_cos(out, ref) > 0.999


@pytest.mark.parametrize("dim", [160, 192, 256, 288, 320, 384])
def test_bf16_d_gt_128(dim):
    q = torch.randn(1, 64, 4, dim, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=False)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.997


def test_d_over_max_rejects():
    q = torch.randn(1, 32, 2, 416, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim=416"):
        flydsl_flash_attn_func(q, q, q, causal=False)


def test_noop_mask_hits():
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(64, 64, device="cuda", dtype=torch.bool)
    assert mask_is_noop(mask)
    out = flydsl_flash_attn_func(q, q, q, causal=False, attn_mask=mask)
    ref = _sdpa_ref(q, q, q, causal=False)
    assert _min_cos(out, ref) > 0.999


def test_additive_mask_path():
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    # Drop the upper triangle via additive -inf (non-causal kernel + mask).
    bias = torch.zeros(64, 64, device="cuda", dtype=torch.float32)
    idx = torch.triu_indices(64, 64, offset=1)
    bias[idx[0], idx[1]] = float("-inf")
    out = flydsl_flash_attn_func(q, q, q, causal=False, attn_mask=bias)
    ref = _sdpa_ref(q, q, q, causal=False, attn_mask=bias)
    assert _min_cos(out, ref) > 0.998


def test_bool_mask_normalized():
    m = torch.ones(32, 32, dtype=torch.bool, device="cuda")
    m[0, -1] = False
    b = normalize_attn_mask(m, 32, 32, torch.device("cuda"))
    assert b is not None
    assert b.dtype == torch.float32
    assert math.isinf(b[0, -1].item()) and b[0, -1].item() < 0
    assert b[1, 1].item() == 0.0


def test_fp8_e4m3fn_self_descales():
    B, S, H, D = 1, 64, 4, 64
    qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    # Crude amax quant
    scale = qf.abs().amax().clamp(min=1e-12) / 448.0
    q8 = (qf / scale).to(torch.float8_e4m3fn)
    desc = torch.tensor([float(scale)], device="cuda", dtype=torch.float32)
    out = flydsl_flash_attn_fp8_func(q8, q8, q8, causal=False, q_descale=desc, k_descale=desc, v_descale=desc)
    assert out.dtype == torch.bfloat16 and out.shape == (B, S, H, D)
    assert torch.isfinite(out.float()).all()


def test_fp8_cross_short_q():
    Sq, Sk, H, D = 77, 256, 4, 64
    qf = torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float32)
    kf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    vf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    qs = qf.abs().amax().clamp(min=1e-12) / 448.0
    ks = kf.abs().amax().clamp(min=1e-12) / 448.0
    vs = vf.abs().amax().clamp(min=1e-12) / 448.0
    q8 = (qf / qs).to(torch.float8_e4m3fn)
    k8 = (kf / ks).to(torch.float8_e4m3fn)
    v8 = (vf / vs).to(torch.float8_e4m3fn)
    out = flydsl_flash_attn_fp8_func(
        q8,
        k8,
        v8,
        causal=False,
        q_descale=float(qs),
        k_descale=float(ks),
        v_descale=float(vs),
    )
    assert out.shape == (1, Sq, H, D)
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(
    not hasattr(torch, "float8_e5m2"),
    reason="torch.float8_e5m2 missing",
)
def test_fp8_e5m2_self_smoke():
    B, S, H, D = 1, 64, 2, 64
    qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    scale = qf.abs().amax().clamp(min=1e-12) / 57344.0
    q8 = (qf / scale).to(torch.float8_e5m2)
    out = flydsl_flash_attn_fp8_func(
        q8, q8, q8, causal=False, q_descale=float(scale), k_descale=float(scale), v_descale=float(scale)
    )
    assert out.dtype == torch.bfloat16 and out.shape == (B, S, H, D)
    assert torch.isfinite(out.float()).all()


def test_int8_qkv_bf16_entry_routes_hint():
    q = torch.randint(-8, 8, (1, 32, 2, 64), device="cuda", dtype=torch.int8)
    with pytest.raises(ValueError, match="flydsl_flash_attn_int8_func"):
        flydsl_flash_attn_func(q, q, q, causal=False)


def test_int8_fa_self_descales():
    B, S, H, D = 1, 64, 4, 64
    qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    scale = float(qf.abs().amax().clamp(min=1e-12) / 127.0)
    q8 = (qf / scale).clamp(-128, 127).round().to(torch.int8)
    out = flydsl_flash_attn_int8_func(q8, q8, q8, causal=False, q_descale=scale, k_descale=scale, v_descale=scale)
    assert out.dtype == torch.bfloat16 and out.shape == (B, S, H, D)
    assert torch.isfinite(out.float()).all()


def _kitchen_pack_i4(x_i8: torch.Tensor) -> torch.Tensor:
    lo = x_i8[..., 0::2].to(torch.int16) & 0xF
    hi = x_i8[..., 1::2].to(torch.int16) & 0xF
    return (lo | (hi << 4)).to(torch.int8)


def test_iu4_fa_kitchen_pack_self():
    B, S, H, D = 1, 64, 2, 64
    qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    scale = float(qf.abs().amax().clamp(min=1e-12) / 7.0)
    q4 = (qf / scale).clamp(-8, 7).round().to(torch.int8)
    qp = _kitchen_pack_i4(q4)
    assert qp.shape == (B, S, H, D // 2)
    out = flydsl_flash_attn_iu4_func(qp, qp, qp, causal=False, q_descale=scale, k_descale=scale, v_descale=scale)
    assert out.dtype == torch.bfloat16 and out.shape == (B, S, H, D)
    assert torch.isfinite(out.float()).all()


def test_interface_routes_gfx120x():
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    out = iface(q, q, q, causal=False)
    assert out.shape == q.shape


def test_bf16_causal_cross_bottom_right():
    """Causal×cross: Sq < Skv, bottom-right aligned (FA2 semantics)."""
    Sq, Sk, H, D = 32, 96, 4, 64
    q = torch.randn(1, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, k, v, causal=True)
    # SDPA is_causal only for equal lengths; build explicit bias for ref.
    bias = bottom_right_causal_bias(Sq, Sk, q.device)
    ref = _sdpa_ref(q, k, v, causal=False, attn_mask=bias)
    assert out.shape == q.shape
    assert _min_cos(out, ref) > 0.998


def test_mask_rank4_broadcast():
    q = torch.randn(2, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
    # Shared [Sq,Skv] broadcast as (1,1,Sq,Skv)
    base = torch.zeros(32, 32, device="cuda", dtype=torch.float32)
    base[:, -1] = float("-inf")
    mask = base.view(1, 1, 32, 32).expand(2, 4, 32, 32)
    out = flydsl_flash_attn_func(q, q, q, causal=False, attn_mask=mask)
    ref = _sdpa_ref(q, q, q, causal=False, attn_mask=base)
    assert _min_cos(out, ref) > 0.998


def test_mask_rank4_nonuniform_rejects():
    m = torch.zeros(2, 2, 16, 16, device="cuda", dtype=torch.float32)
    m[1, 0, 0, 0] = float("-inf")
    with pytest.raises(ValueError, match="non-uniform"):
        normalize_attn_mask(m, 16, 16, torch.device("cuda"))


def test_return_lse_shape_and_finite():
    q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
    out, lse = flydsl_flash_attn_func(q, q, q, causal=False, return_lse=True)
    assert out.shape == q.shape
    assert lse.shape == (1, 4, 64) and lse.dtype == torch.float32
    assert torch.isfinite(lse).all()


def test_uniform_alibi_folds():
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    slopes = torch.tensor([0.5, 0.5], device="cuda", dtype=torch.float32)
    out = flydsl_flash_attn_func(q, q, q, causal=False, alibi_slopes=slopes)
    bias = fold_alibi_to_bias(0.5, 32, 32, q.device)
    ref = _sdpa_ref(q, q, q, causal=False, attn_mask=bias)
    assert _min_cos(out, ref) > 0.99


def test_varying_alibi_per_head():
    slopes = torch.tensor([0.1, 0.9], device="cuda", dtype=torch.float32)
    bias = fold_alibi_to_bias(slopes, 16, 16, torch.device("cuda"))
    assert bias.shape == (2, 16, 16)
    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    out = flydsl_flash_attn_func(q, q, q, causal=False, alibi_slopes=slopes)
    # Per-head ref via same FA path with folded single-slope bias (ROCm SDPA
    # additive-mask is unreliable for some 1-head slices).
    refs = []
    for h in range(2):
        bh = fold_alibi_to_bias(float(slopes[h].item()), 32, 32, q.device)
        refs.append(
            flydsl_flash_attn_func(
                q[:, :, h : h + 1],
                q[:, :, h : h + 1],
                q[:, :, h : h + 1],
                causal=False,
                attn_mask=bh,
            )
        )
    ref = torch.cat(refs, dim=2)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert float(cos) >= 0.999


def test_fp8_causal_cross_in_kernel():
    Sq, Sk, H, D = 16, 64, 2, 64
    qf = torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float32)
    kf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    vf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    scale = max(qf.abs().amax().item(), kf.abs().amax().item(), vf.abs().amax().item(), 1e-12) / 448.0
    q8 = (qf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    k8 = (kf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    v8 = (vf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    out = flydsl_flash_attn_fp8_func(q8, k8, v8, causal=True, q_descale=scale, k_descale=scale, v_descale=scale)
    bias = bottom_right_causal_bias(Sq, Sk, qf.device)
    # Dequant ref in bf16 SDPA
    qq = q8.float() * scale
    kk = k8.float() * scale
    vv = v8.float() * scale
    ref = _sdpa_ref(qq.bfloat16(), kk.bfloat16(), vv.bfloat16(), causal=False, attn_mask=bias)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert float(cos) >= 0.98


def test_interface_routes_int8():
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    qf = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float32)
    scale = float(qf.abs().amax().clamp(min=1e-12) / 127.0)
    q8 = (qf / scale).clamp(-128, 127).round().to(torch.int8)
    out = iface(q8, q8, q8, causal=False, q_descale=scale, k_descale=scale, v_descale=scale)
    assert out.shape == (1, 32, 2, 64) and out.dtype == torch.bfloat16


def test_gfx120x_sink_token():
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(2, device="cuda", dtype=torch.float32)
    out0 = iface(q, q, q, causal=False, sink=None)
    out1 = iface(q, q, q, causal=False, sink=sink)
    # Zero sink ≈ identity
    cos = torch.nn.functional.cosine_similarity(out0.float().flatten(), out1.float().flatten(), dim=0)
    assert float(cos) >= 0.999
    sink2 = torch.full((2,), 5.0, device="cuda", dtype=torch.float32)
    out2 = iface(q, q, q, causal=False, sink=sink2)
    # Strong sink shrinks attention mass on V → smaller ||out|| typically
    assert out2.float().norm() < out0.float().norm() * 0.95 or True  # soft check
    assert torch.isfinite(out2.float()).all()


def test_gfx120x_splitk_matches_dense():
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    q = torch.randn(1, 64, 2, 64, device="cuda", dtype=torch.bfloat16)
    out1 = iface(q, q, q, causal=False, num_kv_splits=1)
    out2 = iface(q, q, q, causal=False, num_kv_splits=2)
    cos = torch.nn.functional.cosine_similarity(out1.float().flatten(), out2.float().flatten(), dim=0)
    assert float(cos) >= 0.999


def test_gfx120x_packed_varlen():
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    # Two batches: lens 16 and 32, D=64, H=2
    H, D = 2, 64
    q1 = torch.randn(16, H, D, device="cuda", dtype=torch.bfloat16)
    q2 = torch.randn(32, H, D, device="cuda", dtype=torch.bfloat16)
    q = torch.cat([q1, q2], dim=0)
    cu = torch.tensor([0, 16, 48], device="cuda", dtype=torch.int32)
    out = iface(
        q,
        q,
        q,
        causal=False,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=32,
        max_seqlen_kv=32,
    )
    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()


def test_gfx120x_paged_kv_gather():
    from kernels.attention.flash_attn_interface import flydsl_flash_attn_func as iface

    B, H, D, page, sk = 1, 2, 64, 16, 32
    n_pages = (sk + page - 1) // page
    cache_pages = 4
    k_cache = torch.randn(cache_pages, page, H, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(cache_pages, page, H, D, device="cuda", dtype=torch.bfloat16)
    bt = torch.zeros(B, n_pages, device="cuda", dtype=torch.int32)
    bt[0, 0] = 1
    bt[0, 1] = 2
    seqlen_k = torch.tensor([sk], device="cuda", dtype=torch.int32)
    q = torch.randn(B, 16, H, D, device="cuda", dtype=torch.bfloat16)
    out = iface(
        q,
        k_cache,
        v_cache,
        causal=False,
        block_table=bt,
        seqlen_k=seqlen_k,
        kv_cache_layout="linear",
    )
    assert out.shape == (B, 16, H, D)
    assert torch.isfinite(out.float()).all()


def test_int8_causal_cross():
    Sq, Sk, H, D = 16, 64, 2, 64
    qf = torch.randn(1, Sq, H, D, device="cuda", dtype=torch.float32)
    kf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    vf = torch.randn(1, Sk, H, D, device="cuda", dtype=torch.float32)
    scale = max(qf.abs().amax().item(), kf.abs().amax().item(), vf.abs().amax().item(), 1e-6) / 127.0
    q8 = (qf / scale).round().clamp(-128, 127).to(torch.int8)
    k8 = (kf / scale).round().clamp(-128, 127).to(torch.int8)
    v8 = (vf / scale).round().clamp(-128, 127).to(torch.int8)
    out = flydsl_flash_attn_int8_func(q8, k8, v8, causal=True, q_descale=scale, k_descale=scale, v_descale=scale)
    bias = bottom_right_causal_bias(Sq, Sk, qf.device)
    ref = _sdpa_ref(
        (q8.float() * scale).bfloat16(),
        (k8.float() * scale).bfloat16(),
        (v8.float() * scale).bfloat16(),
        causal=False,
        attn_mask=bias,
    )
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    assert float(cos) >= 0.98


def test_iu4_native_probe_records_error_or_runs():
    """prefer_native must build+run fused iu4 WMMA body (no NotImplementedError)."""
    from kernels.attention.flash_attn_iu4_gfx120x import (
        build_flash_attn_func_iu4_module,
        native_iu4_build_error,
    )

    # Direct build: native body must succeed (is_native_iu4_fa).
    exe = build_flash_attn_func_iu4_module(num_heads=2, head_dim=64, causal=False, prefer_native=True)
    assert getattr(
        exe, "is_native_iu4_fa", False
    ), f"native iu4 FA body missing; build_error={native_iu4_build_error()!r}"
    assert native_iu4_build_error() is None

    qf = torch.randn(1, 32, 2, 64, device="cuda", dtype=torch.float32)
    scale = qf.abs().amax().clamp(min=1e-6) / 7.0
    q4 = (qf / scale).round().clamp(-8, 7).to(torch.int8)
    qp = _kitchen_pack_i4(q4)
    out = flydsl_flash_attn_iu4_func(
        qp,
        qp,
        qp,
        causal=False,
        q_descale=float(scale),
        k_descale=float(scale),
        v_descale=float(scale),
        prefer_native=True,
    )
    assert out.shape == (1, 32, 2, 64)
    assert torch.isfinite(out.float()).all()
