# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors
# Credit: dimitri91209 + Grokbot

"""High-level FlyDSL Flash Attention APIs for gfx120x (RDNA4).

Ported from the lab aiter gfx1201 (HW) FA pack; FlyDSL module family is gfx120x:
  - bf16 / fp16 dense self + non-causal cross (unequal Sq/Sk)
  - causal self (equal seqlens) and causal×cross via bottom-right host bias
  - FP8 e4m3fn (+ e5m2) with per-tensor descales; self + cross (+ causal×cross)
  - KV tile pad + in-kernel ``seq_len_kv_valid`` mask
  - adaptive BLOCK_M / waves_per_eu
  - head_dim in [64, 384], ``% 32 == 0`` (LDS ≤ ~48 KiB bf16 at D=384;
    RDNA4 workgroup LDS is 64 KiB; multi-batch ceil cover)
  - optional dense additive attn bias / general mask (fp32 [Sq, Skv]);
    common Comfy ranks (B,H,Sq,Skv)/(1,1,Sq,Skv)/(B,1,Sq,Skv) reduced when
    leading dims are broadcast-singleton or identical across B/H
  - noop mask detection on host (bool all-True / additive all-zero)
  - optional ``return_lse`` via host chunked logsumexp (dense only)
  - uniform ALiBi slopes folded into the additive bias; per-head-varying
    slopes and attention-sink need gfx950 dualwave (not on Comfy RDNA4 hot path)

Int8 is not an attention QKV dtype on the Comfy/FlyDSL hot path (weights/GEMM
only); callers that pass int8 QKV get a clear ValueError.

Family name is gfx120x; hardware may report gfx1201. No paged-KV / packed
varlen / split-K path in the aiter gfx120x pack — those remain gfx950 (or
gfx1250 varlen) via ``flash_attn_interface``. Evidence: Comfy
``flydsl_attention.py`` is dense BSHD only; aiter ``fmha_kernels`` gfx1201
entry is dense equal/unequal-seq without cu_seqlens/block_table.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "flydsl_flash_attn_func",
    "flydsl_flash_attn_fp8_func",
    "flydsl_flash_attn_int8_func",
    "flydsl_flash_attn_iu4_func",
    "is_gfx120x",
    "normalize_attn_mask",
    "mask_is_noop",
    "bottom_right_causal_bias",
    "fold_alibi_to_bias",
]

_KERNEL_BLOCK_M = 128
_KERNEL_BLOCK_N = 32
_ADAPTIVE_BLOCK_M_CROSS = (16, 32, 64, 128)
_MAX_HEAD_DIM = 384  # LDS: D=384 @ prefetch1 ≈ 48.5 KiB bf16 < RDNA4 64 KiB
_MIN_HEAD_DIM = 64
_O_BUF_CACHE: dict = {}
_ARCH_OK = None
_DUMMY_BIAS_CACHE: dict = {}


def is_gfx120x(device=None) -> bool:
    """True when the device GCN arch is in the gfx120x family."""
    if device is None:
        if not torch.cuda.is_available():
            return False
        device = torch.device("cuda")
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName or ""
    except Exception:  # noqa: BLE001
        return False
    return arch.lower().split(":")[0].startswith("gfx120")


def _as_contig(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.view(-1) if t.is_contiguous() else t.reshape(-1)


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16 for the dense path, got {dtype!r}")


def _pick_block_m(seq_len_q: int, cross: bool, seq_len_kv: int = 0) -> int:
    """Choose BLOCK_M. Soft-gap sweep 2026-09-24 (idle median)."""
    if not cross:
        if seq_len_q <= 128:
            return 64
        return _KERNEL_BLOCK_M
    if seq_len_q <= 96:
        return 16
    if seq_len_q <= 128:
        return 32
    if seq_len_kv and seq_len_kv <= 128 and seq_len_q >= 512:
        return 64
    return _KERNEL_BLOCK_M


def _pick_waves_per_eu(seq_len_q: int, seq_len_kv: int, cross: bool, requested: int) -> int:
    if requested != 2:
        return requested
    if cross and seq_len_q <= 96 and seq_len_kv >= 512:
        return 4
    return requested


def _acquire_o_buf(shape, dtype, device):
    key = (
        device.index if getattr(device, "index", None) is not None else int(device),
        dtype,
        shape,
    )
    buf = _O_BUF_CACHE.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _O_BUF_CACHE[key] = buf
    return buf


def _dummy_bias(device) -> torch.Tensor:
    key = device.index if getattr(device, "index", None) is not None else int(device)
    buf = _DUMMY_BIAS_CACHE.get(key)
    if buf is None or buf.device != device:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _DUMMY_BIAS_CACHE[key] = buf
    return buf


def mask_is_noop(mask) -> bool:
    """True when mask can be ignored (None / bool all-True / additive all-zero)."""
    if mask is None:
        return True
    try:
        if not hasattr(mask, "dtype"):
            return False
        if mask.numel() == 0:
            return True
        if mask.dtype == torch.bool:
            return bool(mask.all().item())
        return bool((mask == 0).all().item())
    except Exception:  # noqa: BLE001
        return False


def normalize_attn_mask(
    mask,
    seq_len_q: int,
    seq_len_kv: int,
    device,
) -> Optional[torch.Tensor]:
    """Normalize Comfy/SDPA-style mask to dense fp32 additive bias [Sq, Skv].

    Returns None for noop masks. Bool False → -inf; True → 0. Additive masks
    are cast to fp32.

    Accepted ranks (reduced to ``[Sq, Skv]`` when leading dims broadcast):
      - ``[Sq, Skv]``
      - ``[1, Sq, Skv]`` / ``[1, 1, Sq, Skv]``
      - ``[B, 1, Sq, Skv]`` / ``[1, H, Sq, Skv]`` / ``[B, H, Sq, Skv]`` when
        every leading slice is identical (or singleton); otherwise raises so
        the caller can FALLBACK (per-batch / per-head unique masks need a
        different kernel ABI).

    Raises ValueError for shapes that cannot be reduced.
    """
    if mask is None or mask_is_noop(mask):
        return None
    m = mask
    if m.device != device:
        m = m.to(device)

    # Reduce leading dims → 2D [Sq, Skv].
    if m.dim() == 4:
        # (B, H, Sq, Skv) or broadcast variants.
        bsz, nh, sq, sk = m.shape
        if sq != seq_len_q or sk != seq_len_kv:
            raise ValueError(
                f"gfx120x FA attn_mask trailing dims {sq}x{sk} incompatible with "
                f"Sq={seq_len_q} Skv={seq_len_kv} (full shape={tuple(mask.shape)})"
            )
        flat = m.reshape(bsz * nh, sq, sk)
        if flat.shape[0] == 1:
            m = flat[0]
        else:
            # Require identical slices across B*H (broadcast intent).
            ref = flat[0]
            if not torch.equal(flat, ref.unsqueeze(0).expand_as(flat)):
                raise ValueError(
                    f"gfx120x FA attn_mask rank-4 shape {tuple(mask.shape)} has "
                    "non-uniform B/H slices; dense gfx120x bias is head/batch-shared "
                    "[Sq, Skv] only (FALLBACK or use gfx950 dualwave)"
                )
            m = ref
    elif m.dim() == 3:
        # (1, Sq, Skv) or (B, Sq, Skv) with identical rows, or (H, Sq, Skv).
        lead, sq, sk = m.shape
        if sq != seq_len_q or sk != seq_len_kv:
            # Maybe (B, H, Skv) style — not supported without Sq.
            raise ValueError(
                f"gfx120x FA attn_mask shape {tuple(mask.shape)} incompatible with " f"Sq={seq_len_q} Skv={seq_len_kv}"
            )
        if lead == 1:
            m = m[0]
        else:
            ref = m[0]
            if not torch.equal(m, ref.unsqueeze(0).expand_as(m)):
                raise ValueError(
                    f"gfx120x FA attn_mask rank-3 shape {tuple(mask.shape)} has "
                    "non-uniform leading slices; need shared [Sq, Skv]"
                )
            m = ref
    elif m.dim() == 2:
        pass
    elif m.dim() > 4:
        # Squeeze leading singletons then retry once.
        while m.dim() > 4 and m.shape[0] == 1:
            m = m.squeeze(0)
        if m.dim() != 4 and m.dim() != 2:
            raise ValueError(f"gfx120x FA attn_mask must reduce to 2D [Sq, Skv], got shape={tuple(mask.shape)}")
        return normalize_attn_mask(m, seq_len_q, seq_len_kv, device)
    else:
        raise ValueError(f"gfx120x FA attn_mask must reduce to 2D [Sq, Skv], got shape={tuple(mask.shape)}")

    if m.dim() != 2:
        raise ValueError(f"gfx120x FA attn_mask must reduce to 2D [Sq, Skv], got shape={tuple(mask.shape)}")
    if m.shape[0] != seq_len_q or m.shape[1] != seq_len_kv:
        if m.shape[0] == 1 and m.shape[1] == seq_len_kv:
            m = m.expand(seq_len_q, seq_len_kv)
        elif m.shape[0] == seq_len_q and m.shape[1] == 1:
            m = m.expand(seq_len_q, seq_len_kv)
        else:
            raise ValueError(
                f"gfx120x FA attn_mask shape {tuple(m.shape)} incompatible with " f"Sq={seq_len_q} Skv={seq_len_kv}"
            )
    if m.dtype == torch.bool:
        out = torch.zeros(m.shape, dtype=torch.float32, device=device)
        out = out.masked_fill(~m, float("-inf"))
        return _as_contig(out)
    return _as_contig(m.to(torch.float32))


def bottom_right_causal_bias(
    seq_len_q: int,
    seq_len_kv: int,
    device,
) -> torch.Tensor:
    """Bottom-right-aligned causal additive bias ``[Sq, Skv]`` (0 / -inf).

    Matches FlashAttention / FlyDSL interface: query ``i`` attends to keys
    ``j <= i + Skv - Sq``. Used for causal×cross (unequal seqlens) on the
    non-causal kernel + bias path.
    """
    q_idx = torch.arange(seq_len_q, device=device, dtype=torch.int32)[:, None]
    k_idx = torch.arange(seq_len_kv, device=device, dtype=torch.int32)[None, :]
    # j <= i + (Skv - Sq)
    allow = k_idx <= (q_idx + (seq_len_kv - seq_len_q))
    bias = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.float32, device=device)
    bias = bias.masked_fill(~allow, float("-inf"))
    return bias


def fold_alibi_to_bias(
    alibi_slopes,
    seq_len_q: int,
    seq_len_kv: int,
    device,
) -> torch.Tensor:
    """Fold ALiBi slopes into additive bias (bottom-right aligned).

    ``-slope * |i + Skv - Sq - j|``. Scalar/uniform → ``[Sq, Skv]``;
    per-head-varying (1D ``[H]``) → ``[H, Sq, Skv]``.
    """
    if alibi_slopes is None:
        raise ValueError("fold_alibi_to_bias: alibi_slopes is None")
    q_idx = torch.arange(seq_len_q, device=device, dtype=torch.float32)[:, None]
    k_idx = torch.arange(seq_len_kv, device=device, dtype=torch.float32)[None, :]
    dist = (q_idx + (seq_len_kv - seq_len_q) - k_idx).abs()

    if isinstance(alibi_slopes, (int, float)):
        slope = float(alibi_slopes)
        if slope < 0:
            raise ValueError(f"fold_alibi_to_bias: slope must be >= 0, got {slope}")
        return (-slope * dist).contiguous()

    if not isinstance(alibi_slopes, torch.Tensor):
        raise TypeError(f"fold_alibi_to_bias: expected float or Tensor, got {type(alibi_slopes).__name__}")
    s = alibi_slopes.detach().float()
    if s.numel() == 0:
        raise ValueError("fold_alibi_to_bias: empty alibi_slopes")
    if (s < 0).any():
        raise ValueError("fold_alibi_to_bias: slopes must be >= 0")
    if s.dim() == 2:
        if not bool(torch.allclose(s, s[:1].expand_as(s))):
            raise ValueError(
                "gfx120x FA: batch-varying alibi_slopes not supported " f"(got {tuple(alibi_slopes.shape)})"
            )
        s = s[0]
    s = s.reshape(-1)
    if s.numel() == 1 or bool(torch.allclose(s, s[:1].expand_as(s))):
        return (-float(s[0].item()) * dist).contiguous()
    return (-s[:, None, None] * dist[None, :, :]).contiguous()


def _merge_bias(*parts) -> Optional[torch.Tensor]:
    """Elementwise-add non-None fp32 bias tensors (same shape)."""
    acc = None
    for p in parts:
        if p is None:
            continue
        acc = p if acc is None else (acc + p)
    return acc


def _host_row_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    bias: Optional[torch.Tensor],
    chunk_kv: int = 512,
) -> torch.Tensor:
    """Chunked host logsumexp of ``sm_scale * q@k^T (+ bias)`` → ``[B, H, Sq]``.

    Used when ``return_lse=True`` on gfx120x (no in-kernel LSE epilogue yet).
    Matches the natural-log, scale-folded contract of ``flash_attn_interface``.
    """
    import math as _math

    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    scale = 1.0 / _math.sqrt(D)
    qf = q.float().transpose(1, 2)  # B H Sq D
    kf = k.float().transpose(1, 2)  # B H Skv D
    lse = torch.full((B, H, Sq), float("-inf"), device=q.device, dtype=torch.float32)
    for start in range(0, Skv, chunk_kv):
        end = min(start + chunk_kv, Skv)
        scores = torch.matmul(qf, kf[:, :, start:end, :].transpose(-1, -2)) * scale
        if bias is not None:
            scores = scores + bias[:, start:end].to(dtype=torch.float32)
        if causal:
            q_idx = torch.arange(Sq, device=q.device)[None, None, :, None]
            k_idx = torch.arange(start, end, device=q.device)[None, None, None, :]
            allow = k_idx <= (q_idx + (Skv - Sq))
            scores = scores.masked_fill(~allow, float("-inf"))
        lse = torch.logaddexp(lse, torch.logsumexp(scores, dim=-1))
    return lse


@lru_cache(maxsize=64)
def _get_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    waves_per_eu: int,
    daz: bool,
    block_m: int = 128,
    has_attn_bias: bool = False,
):
    from kernels.attention.flash_attn_gfx120x import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        has_attn_bias=has_attn_bias,
    )


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    out: torch.Tensor | None = None,
    attn_mask=None,
    bias=None,
    alibi_slopes=None,
    return_lse: bool = False,
) -> torch.Tensor:
    """Run FlyDSL Flash Attention on RDNA4 (gfx120x family).

    Args:
        q, k, v: BSHD ``[B, S, H, D]`` bf16/fp16. Sq may differ from Sk (cross).
        causal: bottom-right aligned. Equal seqlens use the in-kernel causal
            path; unequal (causal×cross) uses in-kernel bottom-right masking.
        attn_mask / bias: optional general mask. Noop is ignored; otherwise
            normalized to fp32 additive ``[Sq, Skv]`` and applied in-kernel.
        alibi_slopes: optional uniform ALiBi slopes folded into the bias;
            per-head-varying slopes fold to [H,Sq,Skv] and dispatch per head.
        return_lse: when True, return ``(out, lse)`` with fp32 ``[B, H, Sq]``
            from host chunked logsumexp (no in-kernel LSE epilogue yet).
        head_dim: must be in ``[64, 384]`` and ``% 32 == 0``.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    if q.dtype in (torch.int8, torch.uint8):
        raise ValueError(
            "flydsl_flash_attn_func: int8 QKV belongs on flydsl_flash_attn_int8_func "
            "(iu8 WMMA + descales); bf16 path is bf16/fp16 only. "
            "Packed native int4 QKV → flydsl_flash_attn_iu4_func."
        )
    global _ARCH_OK
    if _ARCH_OK is not True:
        if not is_gfx120x(q.device):
            try:
                arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
            except Exception:  # noqa: BLE001
                arch = ""
            raise ValueError(f"flydsl_flash_attn_func requires gfx120x, got {arch!r}")
        _ARCH_OK = True
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensors, got q={tuple(q.shape)} " f"k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if not (
        q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[2] == k.shape[2] == v.shape[2]
        and q.shape[3] == k.shape[3] == v.shape[3]
    ):
        raise ValueError(
            "flydsl_flash_attn_func: q/k/v must share batch, num_heads, "
            f"head_dim; got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"flydsl_flash_attn_func: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    batch, seq_len_q_real, num_heads, head_dim = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real
    # Causal self + causal×cross: in-kernel bottom-right (host bias fold retired).
    causal_cross = False
    kernel_causal = bool(causal)
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0:
        raise ValueError(f"kernel requires head_dim >= {_MIN_HEAD_DIM} and head_dim % 32 == 0, got {head_dim}")
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError(
            f"flydsl_flash_attn_func: head_dim={head_dim} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA (LDS / register tile budget; "
            f"RDNA4 WG LDS 64 KiB holds D<=384 @ BLOCK_N=32 prefetch1)."
        )

    # Merge attn_mask / bias aliases + optional causal×cross / ALiBi folds.
    mask_in = attn_mask if attn_mask is not None else bias
    bias_t = normalize_attn_mask(mask_in, seq_len_q_real, seq_len_kv_real, q.device)
    causal_bias = None
    if causal_cross:
        causal_bias = bottom_right_causal_bias(seq_len_q_real, seq_len_kv_real, q.device)
    alibi_bias = None
    if alibi_slopes is not None:
        alibi_bias = fold_alibi_to_bias(alibi_slopes, seq_len_q_real, seq_len_kv_real, q.device)
    bias_t = _merge_bias(bias_t, causal_bias, alibi_bias)
    if bias_t is not None and bias_t.dim() == 3:
        if int(bias_t.shape[0]) != num_heads:
            raise ValueError(f"gfx120x FA per-head bias H={int(bias_t.shape[0])} != num_heads={num_heads}")
        outs = []
        for h in range(num_heads):
            outs.append(
                flydsl_flash_attn_func(
                    q[:, :, h : h + 1, :],
                    k[:, :, h : h + 1, :],
                    v[:, :, h : h + 1, :],
                    causal=causal,
                    waves_per_eu=waves_per_eu,
                    daz=daz,
                    stream=stream,
                    attn_mask=bias_t[h],
                    bias=None,
                    alibi_slopes=None,
                    return_lse=False,
                    out=None,
                )
            )
        o_cat = torch.cat(outs, dim=2)
        if return_lse:
            lse_parts = [
                _host_row_lse(
                    q[:, :, h : h + 1, :],
                    k[:, :, h : h + 1, :],
                    causal=kernel_causal,
                    bias=bias_t[h],
                )
                for h in range(num_heads)
            ]
            return o_cat, torch.cat(lse_parts, dim=1)
        return o_cat
    has_bias = bias_t is not None

    dtype_str = _torch_dtype_to_str(q.dtype)
    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    seq_len_q_launch = seq_len_q_real
    if cross:
        seq_len_kv_pad = ((seq_len_kv_real + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
    else:
        seq_len_kv_pad = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
        seq_len_kv_pad = ((seq_len_kv_pad + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N

    n_pad_kv = seq_len_kv_pad - seq_len_kv_real
    q_p = _as_contig(q)
    if n_pad_kv > 0:
        k_p = F.pad(_as_contig(k), (0, 0, 0, 0, 0, n_pad_kv))
        v_p = F.pad(_as_contig(v), (0, 0, 0, 0, 0, n_pad_kv))
        if has_bias:
            bias_t = F.pad(bias_t, (0, n_pad_kv))
    else:
        k_p = _as_contig(k)
        v_p = _as_contig(v)

    bias_arg = _as_contig(bias_t) if has_bias else _dummy_bias(q.device)

    o_shape = (batch, seq_len_q_launch, num_heads, head_dim)
    if out is not None:
        if tuple(out.shape) != o_shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(
                f"flydsl_flash_attn_func: out shape/dtype/device mismatch: "
                f"got {tuple(out.shape)}/{out.dtype}/{out.device}, "
                f"want {o_shape}/{q.dtype}/{q.device}"
            )
        o_p = out
    else:
        o_p = _acquire_o_buf(o_shape, q.dtype, q.device)

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        exe = _get_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=kernel_causal,
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            has_attn_bias=has_bias,
        )
        exe(
            _flat(q_p),
            _flat(k_p),
            _flat(v_p),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            seq_len_kv_real,
            _flat(bias_arg),
            stream=launch_stream,
        )

    if return_lse:
        # Host LSE uses the *unpadded* K and the pre-pad merged bias (no KV pad
        # columns). Causal×cross / equal causal handled inside _host_row_lse.
        lse_bias = bias_t
        if lse_bias is not None and n_pad_kv > 0:
            lse_bias = lse_bias[:, :seq_len_kv_real]
        lse = _host_row_lse(
            q_p[:, :seq_len_q_real],
            k[:, :seq_len_kv_real] if k.shape[1] >= seq_len_kv_real else k,
            # Equal-seq causal stays a flag; causal×cross already folded into bias.
            causal=kernel_causal,
            bias=lse_bias,
        )
        return o_p, lse
    return o_p


# ---------------------------------------------------------------------------
# FP8 (E4M3FN / E5M2) — self + cross, descales, seq_len_kv_valid
# ---------------------------------------------------------------------------

_FP8_KERNEL_BLOCK_M = 128


@lru_cache(maxsize=32)
def _get_fp8_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    dtype_str: str = "fp8_e4m3fn",
    block_m: int = 128,
):
    from kernels.attention.flash_attn_fp8_gfx120x import (
        build_flash_attn_func_fp8_module,
    )

    return build_flash_attn_func_fp8_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
    )


def _fp8_descale_to_float(name: str, scale, device) -> float:
    import math as _math

    if scale is None:
        return 1.0
    if isinstance(scale, (int, float)):
        val = float(scale)
    elif isinstance(scale, torch.Tensor):
        if not scale.is_cuda:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be a CUDA tensor, " f"got device={scale.device}")
        if scale.device != device:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be on {device}, got {scale.device}")
        if scale.dtype != torch.float32:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be float32, got {scale.dtype}")
        if scale.numel() != 1:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: {name} must have numel==1 "
                f"(per-tensor), got shape={tuple(scale.shape)}"
            )
        val = float(scale.reshape(-1)[0].item())
    else:
        raise TypeError(
            f"flydsl_flash_attn_fp8_func: {name} must be None, float, or "
            f"float32 CUDA tensor[1], got {type(scale).__name__}"
        )
    if not (_math.isfinite(val) and val > 0.0):
        raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be positive and finite, got {val}")
    return val


def _fp8_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float8_e4m3fn:
        return "fp8_e4m3fn"
    if dtype == torch.float8_e5m2:
        return "fp8_e5m2"
    raise ValueError(f"flydsl_flash_attn_fp8_func expects float8_e4m3fn or float8_e5m2, got {dtype}")


def flydsl_flash_attn_fp8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FlyDSL FP8 Flash Attention on RDNA4 (gfx120x).

    Q/K/V: ``float8_e4m3fn`` or ``float8_e5m2`` BSHD. Supports self and
    non-causal cross (unequal Sq/Sk) via ``seq_len_kv`` / ``seq_len_kv_valid``.
    Output is bf16. Descales match the gfx950 per-tensor contract.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_fp8_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    if not is_gfx120x(q.device):
        try:
            arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
        except Exception:  # noqa: BLE001
            arch = ""
        raise ValueError(f"flydsl_flash_attn_fp8_func requires gfx120x, got {arch!r}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"flydsl_flash_attn_fp8_func: q/k/v dtype must match, got " f"{q.dtype}/{k.dtype}/{v.dtype}")
    dtype_str = _fp8_dtype_str(q.dtype)
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensors, got q={tuple(q.shape)} " f"k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (
        q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[2] == k.shape[2] == v.shape[2]
        and q.shape[3] == k.shape[3] == v.shape[3]
    ):
        raise ValueError(
            "flydsl_flash_attn_fp8_func: q/k/v must share batch, num_heads, head_dim; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"flydsl_flash_attn_fp8_func: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    batch, seq_len_q_real, num_heads, head_dim = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real
    # Causal×cross: in-kernel bottom-right (no dequant→bf16 FALLBACK).
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0:
        raise ValueError(f"kernel requires head_dim >= {_MIN_HEAD_DIM} and head_dim % 32 == 0, got {head_dim}")
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: head_dim={head_dim} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA (LDS / register tile budget)."
        )

    qd = _fp8_descale_to_float("q_descale", q_descale, q.device)
    kd = _fp8_descale_to_float("k_descale", k_descale, q.device)
    vd = _fp8_descale_to_float("v_descale", v_descale, q.device)

    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    seq_len_q_launch = seq_len_q_real
    if cross:
        seq_len_kv_pad = ((seq_len_kv_real + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
    else:
        seq_len_kv_pad = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
        seq_len_kv_pad = ((seq_len_kv_pad + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N

    n_pad_kv = seq_len_kv_pad - seq_len_kv_real
    q_p = _as_contig(q)
    if n_pad_kv > 0:
        k_p = F.pad(_as_contig(k), (0, 0, 0, 0, 0, n_pad_kv))
        v_p = F.pad(_as_contig(v), (0, 0, 0, 0, 0, n_pad_kv))
    else:
        k_p = _as_contig(k)
        v_p = _as_contig(v)

    o_shape = (batch, seq_len_q_launch, num_heads, head_dim)
    if out is not None:
        if tuple(out.shape) != o_shape or out.dtype != torch.bfloat16 or out.device != q.device:
            raise ValueError(f"flydsl_flash_attn_fp8_func: out must be bf16 {o_shape} on {q.device}")
        o_p = out
    else:
        o_p = torch.empty(o_shape, dtype=torch.bfloat16, device=q.device)

    if sm_scale is not None:
        import math as _math

        default = 1.0 / _math.sqrt(head_dim)
        if abs(sm_scale - default) > 1e-7:
            raise ValueError(
                "flydsl_flash_attn_fp8_func only supports default " f"sm_scale=1/sqrt(D) ({default}), got {sm_scale}"
            )

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        exe = _get_fp8_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            waves_per_eu=waves_per_eu,
            daz=daz,
            dtype_str=dtype_str,
            block_m=block_m,
        )
        exe(
            _flat(q_p),
            _flat(k_p),
            _flat(v_p),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            seq_len_kv_real,
            qd,
            kd,
            vd,
            stream=launch_stream,
        )

    return o_p


# ---------------------------------------------------------------------------
# Int8 (iu8 WMMA) — self + cross, descales, seq_len_kv_valid (FP8-patterned)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_int8_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    block_m: int = 128,
):
    from kernels.attention.flash_attn_int8_gfx120x import (
        build_flash_attn_func_int8_module,
    )

    return build_flash_attn_func_int8_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="int8",
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
    )


def flydsl_flash_attn_int8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """FlyDSL int8 Flash Attention on RDNA4 (iu8 WMMA).

    Q/K/V: ``torch.int8`` BSHD. Per-tensor descales match the FP8 contract.
    Output bf16. Self + cross; causal/causal×cross via in-kernel bottom-right
    (no dequant→bf16 FALLBACK). Patterned on ``flydsl_flash_attn_fp8_func``.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_int8_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    if not is_gfx120x(q.device):
        try:
            arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
        except Exception:  # noqa: BLE001
            arch = ""
        raise ValueError(f"flydsl_flash_attn_int8_func requires gfx120x, got {arch!r}")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise ValueError(f"flydsl_flash_attn_int8_func expects torch.int8 QKV, got " f"{q.dtype}/{k.dtype}/{v.dtype}")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensors, got q={tuple(q.shape)} " f"k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (
        q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[2] == k.shape[2] == v.shape[2]
        and q.shape[3] == k.shape[3] == v.shape[3]
    ):
        raise ValueError(
            "flydsl_flash_attn_int8_func: q/k/v must share batch, num_heads, head_dim; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"flydsl_flash_attn_int8_func: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    batch, seq_len_q_real, num_heads, head_dim = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real
    # Causal×cross: in-kernel bottom-right (no dequant→bf16 FALLBACK).
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0:
        raise ValueError(f"kernel requires head_dim >= {_MIN_HEAD_DIM} and head_dim % 32 == 0, got {head_dim}")
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError(
            f"flydsl_flash_attn_int8_func: head_dim={head_dim} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA."
        )

    qd = _fp8_descale_to_float("q_descale", q_descale, q.device)
    kd = _fp8_descale_to_float("k_descale", k_descale, q.device)
    vd = _fp8_descale_to_float("v_descale", v_descale, q.device)

    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    seq_len_q_launch = seq_len_q_real
    if cross:
        seq_len_kv_pad = ((seq_len_kv_real + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
    else:
        seq_len_kv_pad = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
        seq_len_kv_pad = ((seq_len_kv_pad + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N

    n_pad_kv = seq_len_kv_pad - seq_len_kv_real
    q_p = _as_contig(q)
    if n_pad_kv > 0:
        k_p = F.pad(_as_contig(k), (0, 0, 0, 0, 0, n_pad_kv))
        v_p = F.pad(_as_contig(v), (0, 0, 0, 0, 0, n_pad_kv))
    else:
        k_p = _as_contig(k)
        v_p = _as_contig(v)

    o_shape = (batch, seq_len_q_launch, num_heads, head_dim)
    if out is not None:
        if tuple(out.shape) != o_shape or out.dtype != torch.bfloat16 or out.device != q.device:
            raise ValueError(f"flydsl_flash_attn_int8_func: out must be bf16 {o_shape} on {q.device}")
        o_p = out
    else:
        o_p = torch.empty(o_shape, dtype=torch.bfloat16, device=q.device)

    if sm_scale is not None:
        import math as _math

        default = 1.0 / _math.sqrt(head_dim)
        if abs(sm_scale - default) > 1e-7:
            raise ValueError(
                "flydsl_flash_attn_int8_func only supports default " f"sm_scale=1/sqrt(D) ({default}), got {sm_scale}"
            )

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        exe = _get_int8_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
        )
        exe(
            _flat(q_p),
            _flat(k_p),
            _flat(v_p),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            seq_len_kv_real,
            qd,
            kd,
            vd,
            stream=launch_stream,
        )

    return o_p


# ---------------------------------------------------------------------------
# Native int4 (iu4 WMMA) — kitchen-packed QKV, FP8-patterned descales
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_iu4_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    block_m: int = 128,
    prefer_native: bool = True,
):
    from kernels.attention.flash_attn_iu4_gfx120x import (
        build_flash_attn_func_iu4_module,
    )

    return build_flash_attn_func_iu4_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="iu4",
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        prefer_native=prefer_native,
    )


def _kitchen_unpack_iu4_to_i8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack kitchen int4 bytes ``[..., D//2]`` → signed int8 ``[..., D]``."""
    lo = (packed.to(torch.int16) & 0xF).to(torch.int8)
    hi = ((packed.to(torch.int16) >> 4) & 0xF).to(torch.int8)
    # sign-extend nibbles
    lo = torch.where(lo >= 8, lo - 16, lo.to(torch.int16)).to(torch.int8)
    hi = torch.where(hi >= 8, hi - 16, hi.to(torch.int16)).to(torch.int8)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def flydsl_flash_attn_iu4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    out: torch.Tensor | None = None,
    head_dim: int | None = None,
    prefer_native: bool = True,
) -> torch.Tensor:
    """FlyDSL native int4 Flash Attention on RDNA4 (kitchen-pack ABI).

    Q/K/V: kitchen-packed ``torch.int8`` ``[B, S, H, D//2]`` (two signed
    nibbles per byte, low first — same pack as ``rdna4_iu4_gemm``, not AWQ).
    Logical ``head_dim`` defaults to ``2 * q.shape[-1]``.

    Prefers in-kernel iu4 WMMA FA (i32 kitchen-pack load path from
    ``rdna4_iu4_gemm``). On toolchain refusal falls back to kitchen-unpack →
    iu8 FA (same descales / softmax / bf16 O). Pass ``prefer_native=False``
    to skip the native path.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_iu4_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    if not is_gfx120x(q.device):
        try:
            arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
        except Exception:  # noqa: BLE001
            arch = ""
        raise ValueError(f"flydsl_flash_attn_iu4_func requires gfx120x, got {arch!r}")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise ValueError(
            f"flydsl_flash_attn_iu4_func expects kitchen-packed int8 QKV, got " f"{q.dtype}/{k.dtype}/{v.dtype}"
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4D packed BSHD, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (
        q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[2] == k.shape[2] == v.shape[2]
        and q.shape[3] == k.shape[3] == v.shape[3]
    ):
        raise ValueError(
            "flydsl_flash_attn_iu4_func: q/k/v must share batch, num_heads, packed_dim; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"flydsl_flash_attn_iu4_func: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    d_pack = int(q.shape[3])
    logical_d = int(head_dim) if head_dim is not None else d_pack * 2
    if logical_d != d_pack * 2:
        raise ValueError(
            f"flydsl_flash_attn_iu4_func: head_dim={logical_d} must equal 2*packed_last " f"({d_pack * 2})"
        )
    if logical_d < _MIN_HEAD_DIM or logical_d % 32 != 0:
        raise ValueError(f"kernel requires head_dim >= {_MIN_HEAD_DIM} and head_dim % 32 == 0, got {logical_d}")
    if logical_d > _MAX_HEAD_DIM:
        raise ValueError(
            f"flydsl_flash_attn_iu4_func: head_dim={logical_d} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA."
        )

    batch, seq_len_q_real, num_heads, _ = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real

    qd = _fp8_descale_to_float("q_descale", q_descale, q.device)
    kd = _fp8_descale_to_float("k_descale", k_descale, q.device)
    vd = _fp8_descale_to_float("v_descale", v_descale, q.device)

    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    if sm_scale is not None:
        import math as _math

        default = 1.0 / _math.sqrt(logical_d)
        if abs(sm_scale - default) > 1e-7:
            raise ValueError(
                "flydsl_flash_attn_iu4_func only supports default " f"sm_scale=1/sqrt(D) ({default}), got {sm_scale}"
            )

    # Prefer true in-kernel iu4 WMMA FA when the fused body builds.
    if prefer_native:
        try:
            exe = _get_iu4_kernel(
                num_heads=num_heads,
                head_dim=logical_d,
                causal=bool(causal),
                waves_per_eu=int(waves_per_eu),
                daz=bool(daz),
                block_m=block_m,
                prefer_native=True,
            )
        except Exception:
            exe = None
        if exe is not None and getattr(exe, "is_native_iu4_fa", False):
            seq_len_q_launch = seq_len_q_real
            if cross:
                seq_len_kv_pad = ((seq_len_kv_real + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
            else:
                seq_len_kv_pad = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
                seq_len_kv_pad = ((seq_len_kv_pad + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
            n_pad_kv = seq_len_kv_pad - seq_len_kv_real
            q_p = _as_contig(q)
            if n_pad_kv > 0:
                k_p = F.pad(_as_contig(k), (0, 0, 0, 0, 0, n_pad_kv))
                v_p = F.pad(_as_contig(v), (0, 0, 0, 0, 0, n_pad_kv))
            else:
                k_p = _as_contig(k)
                v_p = _as_contig(v)
            o_shape = (batch, seq_len_q_launch, num_heads, logical_d)
            if out is not None:
                if tuple(out.shape) != o_shape or out.dtype != torch.bfloat16 or out.device != q.device:
                    raise ValueError(f"flydsl_flash_attn_iu4_func: out must be bf16 {o_shape} on {q.device}")
                o_p = out
            else:
                o_p = torch.empty(o_shape, dtype=torch.bfloat16, device=q.device)
            with torch.cuda.device(q.device.index):
                launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
                if launch_stream.device != q.device:
                    raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
                exe(
                    _flat(q_p),
                    _flat(k_p),
                    _flat(v_p),
                    _flat(o_p),
                    batch,
                    seq_len_q_launch,
                    seq_len_kv_pad,
                    seq_len_kv_real,
                    qd,
                    kd,
                    vd,
                    stream=launch_stream,
                )
            return o_p

    # Fallback: kitchen-unpack → iu8 FA.
    q8 = _kitchen_unpack_iu4_to_i8(_as_contig(q))
    k8 = _kitchen_unpack_iu4_to_i8(_as_contig(k))
    v8 = _kitchen_unpack_iu4_to_i8(_as_contig(v))
    return flydsl_flash_attn_int8_func(
        q8,
        k8,
        v8,
        causal=causal,
        sm_scale=sm_scale,
        waves_per_eu=waves_per_eu,
        daz=daz,
        stream=stream,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        out=out,
    )
