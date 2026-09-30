# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X AWQ W4A16 dequant + fused GEMV (packed uint4, in-reg dequant).

Kitchen wire (``TensorCoreAWQW4A16Layout`` / eager ``gemv_awq_w4a16``):

* ``qweight[N, K//2]`` int8 — **unsigned** nibble pack: low = even col, high = odd
  (``(u0 & 0xF) | ((u1 & 0xF) << 4)``); indices in ``[0, 15]``.
* ``wscales[K//G, N]`` bf16/fp16 — per-group scales (G default 64).
* ``wzeros[K//G, N]`` same dtype — per-group fp zero points.

Dequant (per element)::

    W[n, k] = (qweight_u4[n, k] - 8) * wscales[k // G, n] + wzeros[k // G, n]

Hot kitchen path is ``gemv_awq_w4a16`` (modulation linears, small batch).
This module ships:

* FlyDSL ``dequant_awq_w4a16_weight`` — device unpack → bf16/fp16 W
* FlyDSL ``build_awq_gemv_module`` / fused ``gemv_awq_w4a16`` — keeps uint4
  packed; applies ``(q-8)*s+z`` in-register while accumulating ``x @ W.T``.
  Tiled path: one block per ``n_tile`` output cols; stage X in LDS; decode
  **8 weights / i32** with one scale+zero (kitchen HIP contract); register
  block over ``max_m`` in ``{1,4}`` for gemv shapes ``[1|4, N, K]``

No Comfy imports. Credit: dimitri91209 + Grokbot.
"""

import math
from functools import lru_cache
from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from .rdna4_common import buf_copy_load, buf_copy_store, ptr_buf_tensor

_DEFAULT_GROUP = 64
WARP = 32
_MAX_K = 16384
KERNEL_NAME = "awq_dequant_w4a16_gfx120x"


def _kernel_signature(**params: object) -> str:
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


def _block_threads(k: int) -> int:
    packed = k // 2
    if packed <= 64:
        return 64
    if packed <= 128:
        return 128
    if packed <= 256:
        return 256
    return 256


def _ptr(t):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def unpack_uint4_row_major(packed):
    """(..., K//2) int8 → (..., K) int8 in [0, 15] (kitchen eager codec)."""
    import torch

    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def reference_dequant_awq_w4a16(qweight, wscales, wzeros, group_size: int = _DEFAULT_GROUP):
    """Torch reference matching kitchen eager / layout.dequantize."""

    n, k_half = qweight.shape
    k = k_half * 2
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    compute_dtype = wscales.dtype
    w_uint = unpack_uint4_row_major(qweight).to(compute_dtype)
    w_groups = w_uint.view(n, k // group_size, group_size)
    scales_ng = wscales.t().unsqueeze(-1)
    zeros_ng = wzeros.t().unsqueeze(-1)
    return ((w_groups - 8.0) * scales_ng + zeros_ng).view(n, k)


@lru_cache(maxsize=64)
def build_awq_dequant_w4a16_module(
    k: int,
    group_size: int = _DEFAULT_GROUP,
    out_dtype: str = "bfloat16",
    block_threads: Optional[int] = None,
):
    """Unpack AWQ uint4 + apply group scales/zeros → bf16/fp16 weight row."""
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size < 8 or group_size % 2 != 0 or k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide K={k}, be even, and >=8")
    if k > _MAX_K:
        raise ValueError(f"K={k} exceeds budget {_MAX_K}")
    if out_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"out_dtype must be bfloat16|float16, got {out_dtype}")

    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads > WARP and block_threads % WARP != 0:
        raise ValueError(f"block_threads={block_threads} must be multiple of {WARP}")

    packed_k = k // 2
    groups = k // group_size
    pack_steps = (packed_k + block_threads - 1) // block_threads
    OutTy = fx.BFloat16 if out_dtype == "bfloat16" else fx.Float16
    out_bytes = 2
    sig = _kernel_signature(
        block=block_threads,
        group_size=group_size,
        k=k,
        dtype=out_dtype,
        op="awq_dequant",
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dequant_kernel(
        Qdata: fx.Pointer,
        Scales: fx.Pointer,  # (N, groups) row-major f32
        Zeros: fx.Pointer,  # (N, groups) row-major f32
        Out: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
    ):
        bid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        active_row = bid < n_rows

        q_buf = ptr_buf_tensor(
            Qdata,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(packed_k),
        )
        s_buf = ptr_buf_tensor(
            Scales,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(groups) * fx.Int64(4),
        )
        z_buf = ptr_buf_tensor(
            Zeros,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(groups) * fx.Int64(4),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(K) * fx.Int64(out_bytes),
        )

        for step in range_constexpr(pack_steps):
            pc = tid + fx.Int32(step * block_threads)
            inb = active_row & (pc < fx.Int32(packed_k))
            gidx = fx.Int64(bid) * fx.Int64(packed_k) + fx.Int64(pc)
            safe_g = inb.select(gidx, fx.Int64(0))
            raw = buf_copy_load(q_buf, safe_g, elem=fx.Int8, unit_elems=1)
            packed = fx.Int32(raw) & fx.Int32(0xFF)
            lo = packed & fx.Int32(0xF)
            hi = (packed >> fx.Int32(4)) & fx.Int32(0xF)

            col0 = pc * fx.Int32(2)
            g = col0 // fx.Int32(group_size)
            sidx = fx.Int64(bid) * fx.Int64(groups) + fx.Int64(g)
            safe_s = inb.select(sidx, fx.Int64(0))
            s = fx.Float32(buf_copy_load(s_buf, safe_s, elem=fx.Float32, unit_elems=1))
            z = fx.Float32(buf_copy_load(z_buf, safe_s, elem=fx.Float32, unit_elems=1))

            # W = (u4 - 8) * scale + zero
            v0 = (fx.Float32(lo) - fx.Float32(8.0)) * s + z
            v1 = (fx.Float32(hi) - fx.Float32(8.0)) * s + z

            o0 = fx.Int64(bid) * fx.Int64(K) + fx.Int64(col0)
            o1 = o0 + fx.Int64(1)
            # OOB store contract matches rdna4_asym_w4a8 (inactive → index 0).
            out_idx0 = inb.select(o0, fx.Int64(0))
            out_idx1 = inb.select(o1, fx.Int64(0))
            buf_copy_store(o_buf, out_idx0, OutTy(v0), elem=OutTy, unit_elems=1)
            buf_copy_store(o_buf, out_idx1, OutTy(v1), elem=OutTy, unit_elems=1)

    dequant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Qdata: fx.Pointer,
        Scales: fx.Pointer,
        Zeros: fx.Pointer,
        Out: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
        stream: fx.Stream,
    ):
        dequant_kernel(Qdata, Scales, Zeros, Out, n_rows, K).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def dequant_awq_w4a16_weight(
    qweight,
    wscales,
    wzeros,
    group_size: int = _DEFAULT_GROUP,
    *,
    stream=None,
):
    """FlyDSL AWQ W4A16 weight dequant → bf16/fp16 ``(N, K)``.

    ``wscales`` / ``wzeros`` keep kitchen layout ``(K//G, N)``; converted to
    row-major ``(N, G)`` f32 for the kernel.
    """
    import torch

    from kernels.common.tensor_shim import _run_compiled

    if qweight.dim() != 2 or qweight.dtype != torch.int8:
        raise ValueError("qweight must be 2D int8")
    if qweight.device.type != "cuda":
        raise ValueError("qweight must be on CUDA/ROCm")
    n, k_half = qweight.shape
    k = k_half * 2
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    groups = k // group_size
    if tuple(wscales.shape) != (groups, n):
        raise ValueError(f"wscales must be {(groups, n)}, got {tuple(wscales.shape)}")
    if tuple(wzeros.shape) != (groups, n):
        raise ValueError(f"wzeros must be {(groups, n)}, got {tuple(wzeros.shape)}")
    if wscales.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"wscales dtype {wscales.dtype} unsupported")
    if wzeros.dtype != wscales.dtype:
        wzeros = wzeros.to(wscales.dtype)

    out_dtype = "bfloat16" if wscales.dtype == torch.bfloat16 else "float16"
    # Kernel consumes (N, groups) f32
    scales_ng = wscales.t().contiguous().float()
    zeros_ng = wzeros.t().contiguous().float()
    out = torch.empty((n, k), device=qweight.device, dtype=wscales.dtype)
    launch = build_awq_dequant_w4a16_module(k=k, group_size=group_size, out_dtype=out_dtype)
    if stream is None:
        stream = torch.cuda.current_stream()
    _run_compiled(
        launch,
        _ptr(qweight.contiguous()),
        _ptr(scales_ng),
        _ptr(zeros_ng),
        _ptr(out),
        n,
        k,
        stream,
    )
    return out


@lru_cache(maxsize=64)
def build_awq_gemv_module(
    k: int,
    group_size: int = _DEFAULT_GROUP,
    act_dtype: str = "bfloat16",
    max_m: int = 1,
    n_tile: int = 1,
    block_threads: Optional[int] = None,
):
    """Fused AWQ W4A16 GEMV — i32×8 decode, optional LDS-X + N-tile.

    Grid ``ceil(N / n_tile)``. Kitchen HIP contract: decode eight weights per
    i32 chunk, one scale/zero (``group_size % 8 == 0``). When ``n_tile > 1``,
    stage ``X[0:max_m, :]`` in LDS and reuse across the N-tile (cuts X traffic).
    When ``n_tile == 1``, read X from GMEM (LDS fill would be pure overhead).
    """
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size <= 0 or group_size % 8 != 0 or k % group_size != 0:
        raise ValueError(f"group_size={group_size} must be a positive multiple of 8 and divide K={k}")
    if k > _MAX_K:
        raise ValueError(f"K={k} exceeds budget {_MAX_K}")
    if act_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"act_dtype must be bfloat16|float16, got {act_dtype}")
    if max_m not in (1, 4):
        raise ValueError(f"max_m must be 1 or 4, got {max_m}")
    if n_tile not in (1, 4, 8):
        raise ValueError(f"n_tile must be 1, 4, or 8, got {n_tile}")
    packed_k = k // 2
    if packed_k % 4 != 0:
        raise ValueError(f"packed_k={packed_k} must be multiple of 4 for i32 decode")

    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads > WARP and block_threads % WARP != 0:
        raise ValueError(f"block_threads={block_threads} must be multiple of {WARP}")

    groups = k // group_size
    vec_k = packed_k // 4
    vec_steps = (vec_k + block_threads - 1) // block_threads
    use_lds = n_tile > 1
    x_fill_steps = (max_m * k + block_threads - 1) // block_threads if use_lds else 0
    ActTy = fx.BFloat16 if act_dtype == "bfloat16" else fx.Float16
    act_bytes = 2
    reduction_slots = block_threads // WARP
    n_acc = max_m * n_tile
    red_elems = max(reduction_slots * n_acc, n_acc)
    RedTy = fx.Array[fx.Float32, red_elems, 16]
    # Always declare X LDS (size 1 if unused) so struct shape is valid.
    x_lds_elems = max_m * k if use_lds else 1
    XLdsTy = fx.Array[ActTy, x_lds_elems, 16]
    sig = _kernel_signature(
        block=block_threads,
        group_size=group_size,
        k=k,
        dtype=act_dtype,
        max_m=max_m,
        n_tile=n_tile,
        op="awq_gemv_tile",
    )

    @fx.struct
    class SharedStorage:
        x_tile: XLdsTy
        reduction_buffer: RedTy

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def gemv_kernel(
        X: fx.Pointer,
        Qdata: fx.Pointer,
        Scales: fx.Pointer,
        Zeros: fx.Pointer,
        Bias: fx.Pointer,
        Out: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        has_bias: fx.Int32,
    ):
        bid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        n0 = bid * fx.Int32(n_tile)
        c0 = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        x_lds = lds.x_tile.view(fx.make_layout(x_lds_elems, 1))
        red = lds.reduction_buffer.view(fx.make_layout(red_elems, 1))

        def wave_reduce_sum(w0):
            w = w0
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w = w + gpu.shuffle_xor(w, off, WARP)
            return w

        x_buf = ptr_buf_tensor(
            X,
            elem=ActTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(M) * fx.Int64(k) * fx.Int64(act_bytes),
        )
        q_i32_buf = ptr_buf_tensor(
            Qdata,
            elem=fx.Int32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(N) * fx.Int64(packed_k),
        )
        s_buf = ptr_buf_tensor(
            Scales,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(N) * fx.Int64(groups) * fx.Int64(4),
        )
        z_buf = ptr_buf_tensor(
            Zeros,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(N) * fx.Int64(groups) * fx.Int64(4),
        )
        b_buf = ptr_buf_tensor(
            Bias,
            elem=ActTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(N) * fx.Int64(act_bytes),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=ActTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(M) * fx.Int64(N) * fx.Int64(act_bytes),
        )

        # Optional LDS stage of X (only when n_tile>1 so X is reused).
        if use_lds:
            for step in range_constexpr(x_fill_steps):
                idx = tid + fx.Int32(step * block_threads)
                in_x = idx < fx.Int32(max_m * k)
                mi = idx // fx.Int32(k)
                ki = idx - mi * fx.Int32(k)
                row_ok = mi < M
                gidx = fx.Int64(mi) * fx.Int64(k) + fx.Int64(ki)
                safe_g = (in_x & row_ok).select(gidx, fx.Int64(0))
                xv = fx.Float32(buf_copy_load(x_buf, safe_g, elem=ActTy, unit_elems=1))
                store_v = (in_x & row_ok).select(ActTy(xv), ActTy(c0))
                if in_x:
                    fx.memref_store(store_v, x_lds, idx)
            gpu.barrier()

        partials = [c0 for _ in range(n_acc)]

        for step in range_constexpr(vec_steps):
            vi = tid + fx.Int32(step * block_threads)
            in_k = vi < fx.Int32(vec_k)
            col0 = vi * fx.Int32(8)
            g = col0 // fx.Int32(group_size)

            # Gather 8 X values per m-row (LDS or GMEM).
            x_m = []
            for mi in range_constexpr(max_m):
                xs = []
                for t in range_constexpr(8):
                    if use_lds:
                        base = fx.Int32(mi * k) + col0
                        xi = in_k.select(base + fx.Int32(t), fx.Int32(0))
                        xs.append(fx.Float32(x_lds[xi]))
                    else:
                        gidx = fx.Int64(mi) * fx.Int64(k) + fx.Int64(col0 + fx.Int32(t))
                        m_ok = fx.Int32(mi) < M
                        safe = (in_k & m_ok).select(gidx, fx.Int64(0))
                        xs.append(fx.Float32(buf_copy_load(x_buf, safe, elem=ActTy, unit_elems=1)))
                x_m.append(xs)

            for nt in range_constexpr(n_tile):
                n = n0 + fx.Int32(nt)
                active = in_k & (n < N)
                qidx = fx.Int64(n) * fx.Int64(vec_k) + fx.Int64(vi)
                safe_q = active.select(qidx, fx.Int64(0))
                raw = fx.Int32(buf_copy_load(q_i32_buf, safe_q, elem=fx.Int32, unit_elems=1))
                sidx = fx.Int64(n) * fx.Int64(groups) + fx.Int64(g)
                safe_s = active.select(sidx, fx.Int64(0))
                s = fx.Float32(buf_copy_load(s_buf, safe_s, elem=fx.Float32, unit_elems=1))
                z = fx.Float32(buf_copy_load(z_buf, safe_s, elem=fx.Float32, unit_elems=1))
                for mi in range_constexpr(max_m):
                    m_ok = fx.Int32(mi) < M
                    acc_i = mi * n_tile + nt
                    dot = c0
                    for t in range_constexpr(8):
                        nib = (raw >> fx.Int32(4 * t)) & fx.Int32(0xF)
                        w = (fx.Float32(nib) - fx.Float32(8.0)) * s + z
                        dot = dot + x_m[mi][t] * w
                    contrib = (active & m_ok).select(dot, c0)
                    partials[acc_i] = partials[acc_i] + contrib

        # Batched multi-acc reduce (2 barriers total).
        lane = tid % fx.Int32(WARP)
        wave = tid // fx.Int32(WARP)
        wave_sums = []
        for acc_i in range_constexpr(n_acc):
            wave_sums.append(wave_reduce_sum(partials[acc_i]))
        if lane == 0:
            for acc_i in range_constexpr(n_acc):
                fx.memref_store(wave_sums[acc_i], red, wave * fx.Int32(n_acc) + fx.Int32(acc_i))
        gpu.barrier()
        if wave == 0:
            for acc_i in range_constexpr(n_acc):
                lane_ok = lane < fx.Int32(reduction_slots)
                lane_safe = lane_ok.select(lane, fx.Int32(0))
                vv = red[lane_safe * fx.Int32(n_acc) + fx.Int32(acc_i)]
                ww = lane_ok.select(vv, c0)
                ww = wave_reduce_sum(ww)
                if lane == 0:
                    fx.memref_store(ww, red, fx.Int32(acc_i))
        gpu.barrier()
        if tid == 0:
            for mi in range_constexpr(max_m):
                for nt in range_constexpr(n_tile):
                    acc_i = mi * n_tile + nt
                    acc = fx.Float32(red[fx.Int32(acc_i)])
                    n = n0 + fx.Int32(nt)
                    row_ok = fx.Int32(mi) < M
                    col_ok = n < N
                    if has_bias != 0:
                        b = fx.Float32(
                            buf_copy_load(
                                b_buf,
                                (row_ok & col_ok).select(fx.Int64(n), fx.Int64(0)),
                                elem=ActTy,
                                unit_elems=1,
                            )
                        )
                        acc = acc + (row_ok & col_ok).select(b, c0)
                    oidx = fx.Int64(mi) * fx.Int64(N) + fx.Int64(n)
                    safe_o = (row_ok & col_ok).select(oidx, fx.Int64(0))
                    if row_ok & col_ok:
                        buf_copy_store(o_buf, safe_o, ActTy(acc), elem=ActTy, unit_elems=1)

    gemv_kernel.__name__ = f"{KERNEL_NAME}_gemv_{sig}"

    @flyc.jit
    def launch(
        X: fx.Pointer,
        Qdata: fx.Pointer,
        Scales: fx.Pointer,
        Zeros: fx.Pointer,
        Bias: fx.Pointer,
        Out: fx.Pointer,
        M: fx.Int32,
        N: fx.Int32,
        has_bias: fx.Int32,
        stream: fx.Stream,
    ):
        grid_n = (N + n_tile - 1) // n_tile
        gemv_kernel(X, Qdata, Scales, Zeros, Bias, Out, M, N, has_bias).launch(
            grid=(grid_n, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_gemv_{sig}"
    return launch


def gemv_awq_w4a16(
    x,
    qweight,
    wscales,
    wzeros,
    bias=None,
    group_size: int = _DEFAULT_GROUP,
    *,
    stream=None,
):
    """Kitchen-API ``gemv_awq_w4a16``: fused FlyDSL GEMV (packed W, in-reg dequant).

    Layout matches kitchen: ``qweight[N,K//2]`` uint4, ``wscales/wzeros[K/G,N]``.
    Tile pick: ``m==1`` → ``max_m=1, n_tile=8`` (LDS-X reuse); ``m∈(1,4]`` →
    ``max_m=4, n_tile=1`` (multi-row register block, GMEM X); M>4 host-chunks.
    """
    import torch

    from kernels.common.tensor_shim import _run_compiled

    if x.dim() < 1:
        raise ValueError("x must be at least 1D")
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    m, k = x2d.shape
    if qweight.dim() != 2 or qweight.dtype != torch.int8:
        raise ValueError("qweight must be 2D int8")
    if qweight.device.type != "cuda" or x2d.device.type != "cuda":
        raise ValueError("x/qweight must be on CUDA/ROCm")
    n, k_half = qweight.shape
    if k_half * 2 != k:
        raise ValueError(f"qweight K//2={k_half} inconsistent with x K={k}")
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    if group_size <= 0 or group_size % 8 != 0:
        raise ValueError(f"group_size must be a positive multiple of 8, got {group_size}")
    groups = k // group_size
    if tuple(wscales.shape) != (groups, n):
        raise ValueError(f"wscales must be {(groups, n)}, got {tuple(wscales.shape)}")
    if tuple(wzeros.shape) != (groups, n):
        raise ValueError(f"wzeros must be {(groups, n)}, got {tuple(wzeros.shape)}")
    if wscales.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"wscales dtype {wscales.dtype} unsupported")
    if x2d.dtype != wscales.dtype:
        x2d = x2d.to(wscales.dtype)
    if wzeros.dtype != wscales.dtype:
        wzeros = wzeros.to(wscales.dtype)

    act_dtype = "bfloat16" if wscales.dtype == torch.bfloat16 else "float16"
    scales_ng = wscales.t().contiguous().float()
    zeros_ng = wzeros.t().contiguous().float()
    out = torch.empty((m, n), device=x2d.device, dtype=wscales.dtype)
    if bias is None:
        bias_buf = torch.zeros((n,), device=x2d.device, dtype=wscales.dtype)
        has_bias = 0
    else:
        if tuple(bias.shape) != (n,):
            raise ValueError(f"bias must be {(n,)}, got {tuple(bias.shape)}")
        bias_buf = bias.to(device=x2d.device, dtype=wscales.dtype).contiguous()
        has_bias = 1

    if stream is None:
        stream = torch.cuda.current_stream()
    qw = qweight.contiguous()

    def _launch_chunk(x_chunk, out_chunk, mm: int):
        if mm == 1:
            max_m, n_tile = 1, (8 if n >= 8 else 1)
        else:
            max_m, n_tile = 4, 1
        launch = build_awq_gemv_module(k=k, group_size=group_size, act_dtype=act_dtype, max_m=max_m, n_tile=n_tile)
        # Pad M up to max_m when needed (kernel reads runtime M for masks).
        if mm < max_m:
            x_pad = torch.zeros((max_m, k), device=x_chunk.device, dtype=x_chunk.dtype)
            x_pad[:mm].copy_(x_chunk)
            o_pad = torch.empty((max_m, n), device=out_chunk.device, dtype=out_chunk.dtype)
            _run_compiled(
                launch,
                _ptr(x_pad),
                _ptr(qw),
                _ptr(scales_ng),
                _ptr(zeros_ng),
                _ptr(bias_buf),
                _ptr(o_pad),
                mm,
                n,
                has_bias,
                stream,
            )
            out_chunk.copy_(o_pad[:mm])
        else:
            _run_compiled(
                launch,
                _ptr(x_chunk),
                _ptr(qw),
                _ptr(scales_ng),
                _ptr(zeros_ng),
                _ptr(bias_buf),
                _ptr(out_chunk),
                mm,
                n,
                has_bias,
                stream,
            )

    if m <= 4:
        _launch_chunk(x2d, out, m)
    else:
        for m0 in range(0, m, 4):
            mm = min(4, m - m0)
            _launch_chunk(x2d[m0 : m0 + mm], out[m0 : m0 + mm], mm)

    return out.reshape(*orig_shape[:-1], n)
