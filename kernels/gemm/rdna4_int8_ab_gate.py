# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Static Path A/B size gate for gfx1201 int8 linear (host-only).

Chooses Path A (``rdna4_w8a16_path_a`` — bf16 WMMA, in-VGPR weight dequant) vs
Path B (``rdna4_int8_linear`` — iu8 WMMA, dyn-quant acts) from a cheap **M-floor
then M×K** rule (WORKER_REFERENCE.md / playbook §11; Gemini review 2026-09-25):

  tiny  32×64×128       M=32   ≤128           → A
  mid   256×512×512     M>128, M*K=131072     → A  (≤ MK_A_MAX)
  large 1024×4096×4096  M>128, M*K=4194304    → B
  wanish 4096×3072×3072 M>128, M*K=12582912   → B

Rule (kwargs; no Comfy env names on the tip surface)
----------------------------------------------------
  1. ``force_path`` if set → that path
  2. Else if ``M <= m_a_max`` (default **128**) → **A**
  3. Else if ``M*K <= mk_a_max`` (default **500000**) → **A**
  4. Else → **B**

Path A ≠ HIP bits (no act dyn-quant). Path B ≡ HIP bits on measured shapes.
Default production without an explicit gate call remains Path B / iu8.

Credit: dimitri91209+Grokbot.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

PathName = Literal["A", "B"]

DEFAULT_M_A_MAX = 128
DEFAULT_MK_A_MAX = 500_000


def select_path(
    M: int,
    N: int,
    K: int,
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    force_path: Optional[PathName] = None,
) -> PathName:
    """Cheap static gate: ``"A" | "B"``. ``N`` unused by size selection."""
    del N  # API symmetry with MNK docs
    if force_path is not None:
        if force_path not in ("A", "B"):
            raise ValueError(f"force_path must be A|B, got {force_path!r}")
        return force_path
    m = int(M)
    k = int(K)
    if m <= int(m_a_max):
        return "A"
    if m * k <= int(mk_a_max):
        return "A"
    return "B"


def select_path_for_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    force_path: Optional[PathName] = None,
) -> PathName:
    """M from flattened x rows; N/K from weight [N,K]."""
    x2d = x.reshape(-1, x.shape[-1])
    return select_path(
        int(x2d.shape[0]),
        int(weight.shape[0]),
        int(x2d.shape[1]),
        m_a_max=m_a_max,
        mk_a_max=mk_a_max,
        force_path=force_path,
    )


def gate_rule_doc(
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
) -> dict:
    """Structured description of the active gate (smokes / docs)."""
    return {
        "m_a_max": m_a_max,
        "mk_a_max": mk_a_max,
        "default_m_a_max": DEFAULT_M_A_MAX,
        "default_mk_a_max": DEFAULT_MK_A_MAX,
        "rule": (
            f"force_path if set; else A if M <= {m_a_max}; "
            f"else A if M*K <= {mk_a_max}; else B"
        ),
        "path_a": "kernels.gemm.rdna4_w8a16_path_a (bf16 WMMA; no iu8)",
        "path_b": "kernels.gemm.rdna4_int8_linear (iu8 WMMA; tip B)",
        "smoke_anchor": {
            "tiny_expect_A": {"M": 32, "N": 128, "K": 64, "why": "M<=128"},
            "mid_expect_A": {
                "M": 256,
                "N": 512,
                "K": 512,
                "why": "M>128 but M*K<=MK_A_MAX",
            },
            "large_expect_B": {
                "M": 1024,
                "N": 4096,
                "K": 4096,
                "why": "M*K>MK_A_MAX",
            },
            "wanish_expect_B": {
                "M": 4096,
                "N": 3072,
                "K": 3072,
                "why": "M*K>MK_A_MAX",
            },
        },
    }


def _run_path_b(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Path B via tip ``rdna4_int8_linear`` raw-pointer launcher."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.tensor_shim import _run_compiled
    from kernels.gemm.rdna4_int8_linear import (
        create_wmma_int8_linear_module,
        pick_tile_config,
    )

    if a_int8.dtype != torch.int8 or b_int8.dtype != torch.int8:
        raise ValueError("path B requires int8 A/B")
    a = a_int8.contiguous()
    b = b_int8.contiguous()
    if a.data_ptr() % 16:
        a = a.clone()
    if b.data_ptr() % 16:
        b = b.clone()
    m, k = a.shape
    n = b.shape[0]
    if b.shape[1] != k:
        raise ValueError(f"inner dim mismatch: a K={k}, b K={b.shape[1]}")
    if k % 16 != 0:
        raise ValueError(f"K must be divisible by 16, got {k}")

    x_scale = x_scale.to(device=a.device, dtype=torch.float32).reshape(-1).contiguous()
    w_scale = w_scale.to(device=a.device, dtype=torch.float32).reshape(-1).contiguous()
    if x_scale.numel() != m:
        raise ValueError(f"x_scale must be [M]={m}, got {x_scale.numel()}")
    w_per_n = w_scale.numel() != 1
    if w_per_n and w_scale.numel() != n:
        raise ValueError(f"w_scale must be scalar or [N]={n}, got {w_scale.numel()}")

    store_dtype = torch.float32 if bias is not None else out_dtype
    store_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[store_dtype]
    out = torch.empty((m, n), device=a.device, dtype=store_dtype)
    cfg = pick_tile_config(m, n, k)
    launch = create_wmma_int8_linear_module(
        store_name,
        cfg,
        skip_bounds=(m % cfg.bm == 0 and n % cfg.bn == 0),
        w_scale_per_n=w_per_n,
    )

    def _ptr(t: torch.Tensor):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    _run_compiled(
        launch,
        _ptr(a),
        _ptr(b),
        _ptr(out),
        _ptr(x_scale),
        _ptr(w_scale),
        int(m),
        int(n),
        int(k),
        torch.cuda.current_stream(device=a.device),
    )
    if bias is not None:
        bias = bias.to(device=out.device, dtype=torch.float32).reshape(-1)
        if bias.numel() != n:
            raise ValueError(f"bias must be [N]={n}")
        out = (out + bias).to(out_dtype)
    elif store_dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def int8_linear_gated(
    path: PathName,
    *,
    a_int8: Optional[torch.Tensor] = None,
    b_int8: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    a_act: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dispatch to Path A or B by ``path``. Caller supplies matching tensors.

    Path ``"B"``: ``a_int8``, ``b_int8``, ``x_scale``, ``w_scale`` (needs tip B iu8).
    Path ``"A"``: ``a_act`` (bf16/fp16), ``b_int8``, ``w_scale``.
    """
    if path == "B":
        if a_int8 is None or b_int8 is None or x_scale is None or w_scale is None:
            raise ValueError("path B requires a_int8, b_int8, x_scale, w_scale")
        try:
            return _run_path_b(a_int8, b_int8, x_scale, w_scale, bias, out_dtype)
        except ImportError as e:
            raise ImportError(
                "Path B requires kernels.gemm.rdna4_int8_linear (tip "
                "gfx1201-iu8-int8 / iu8 WMMA atom). " + str(e)
            ) from e
    if path == "A":
        if a_act is None or b_int8 is None or w_scale is None:
            raise ValueError("path A requires a_act, b_int8, w_scale")
        from kernels.gemm.rdna4_w8a16_path_a import w8a16_gemm_path_a

        return w8a16_gemm_path_a(
            a_act, b_int8, w_scale, bias=bias, out_dtype=out_dtype
        )
    raise ValueError(f"unknown path {path!r}")


def int8_linear_auto(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    x_scale: Optional[torch.Tensor] = None,
    a_int8: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    force_path: Optional[PathName] = None,
) -> torch.Tensor:
    """Select path from MNK then dispatch.

    Path A uses ``x`` (bf16/fp16) directly. Path B needs pre-quantized ``a_int8``
    + ``x_scale`` (from ``quantize_int8_rowwise``); if omitted, raises with guidance.
    """
    if out_dtype is None:
        out_dtype = x.dtype
    path = select_path_for_tensors(
        x, weight, m_a_max=m_a_max, mk_a_max=mk_a_max, force_path=force_path
    )
    if path == "A":
        return int8_linear_gated(
            "A",
            a_act=x if x.ndim == 2 else x.reshape(-1, x.shape[-1]),
            b_int8=weight,
            w_scale=weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )
    if a_int8 is None or x_scale is None:
        raise ValueError(
            "Path B selected but a_int8/x_scale missing — run "
            "kernels.quant.rdna4_quantize_int8_rowwise.quantize_int8_rowwise(x) "
            "first, or force_path='A'"
        )
    return int8_linear_gated(
        "B",
        a_int8=a_int8,
        b_int8=weight,
        x_scale=x_scale,
        w_scale=weight_scale,
        bias=bias,
        out_dtype=out_dtype,
    )


__all__ = [
    "DEFAULT_M_A_MAX",
    "DEFAULT_MK_A_MAX",
    "PathName",
    "select_path",
    "select_path_for_tensors",
    "gate_rule_doc",
    "int8_linear_gated",
    "int8_linear_auto",
]
