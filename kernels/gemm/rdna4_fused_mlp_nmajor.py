# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Zero-LDS N-major float GEMM building block + SwiGLU MLP host (gfx120x).

RDNA4 WMMA D is M-major in VGPRs and is a poor next-GEMM A without an LDS
transpose. Swapping A/B on the first WMMA lands D N-major so the next MMA can
consume it in-register. See GPUOpen WMMA guide part 1 (RDNA 4):
https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/

What this module provides
-------------------------
* ``build_wmma_tile_module(swap_ab=...)`` -- single-wave 16x16x16 bf16/fp16
  WMMA (Sequence ABI, matching upstream ``scaled_mm``). ``swap_ab=False`` matches
  ``A @ B.T``. ``swap_ab=True`` + standard Wave32 store yields the transpose on
  square tiles (layout probe for the A/B swap).
* ``fused_gemm_tn`` / ``build_fused_gemm_tn_module`` -- zero-LDS two-GEMM fuse:
  first WMMA swapped (N-major D0 in VGPR) -> narrow f32->bf16/fp16 -> second
  WMMA **unswapped** consumes D0 as A -> standard M-major store. Empirically
  ``max_err=0`` vs ``(A0@B0.T)@B1.T`` on gfx120x. GPUOpen's sample swaps both
  builtins; this path needs swap on GEMM0 only under the FlyDSL Sequence ABI.
* ``gemm_bf16_nmajor`` -- host-tiled ``A @ B.T`` via the unswapped tile
  (building block for the MLP host). Not a silent production layout change.
* ``fused_swiglu_mlp_inreg`` / ``build_fused_swiglu_mlp_module`` -- in-register
  SwiGLU fuse for one-wave 16x16 panels. Host tiles along M/N when
  ``K == FFN == 16`` and ``M, N_out`` are multiples of 16. Per tile: GEMM0_gate
  + GEMM0_up (A/B-swapped) -> SiLU(gate)*up in registers -> narrow -> GEMM1
  unswapped -> M-major store. Layout: ``silu(x @ Wgate.T) * (x @ Wup.T)`` then
  ``@ Wdown.T`` with weights ``[N, K]``.
* ``fused_swiglu_mlp_nmajor`` -- thin host chain (separate launches) for other
  sizes. Prefer ``fused_swiglu_mlp_inreg`` when ``K == FFN == 16``.

Known limits
------------
* Multi-wave in-kernel / LDS-pipelined N-major production GEMM is not here yet.
* In-reg SwiGLU for ``K != 16`` or ``FFN != 16`` needs a full-K reduce before
  SiLU (or a mid GMEM spill); use ``fused_swiglu_mlp_nmajor`` for those shapes.
* Host-tiled in-reg covers ``K == FFN == 16`` with ``M, N_out`` multiples of 16
  (one fused 16x16 launch per (M,N) tile; N>16 recomputes GEMM0+SiLU per N tile).
* Production ``scaled_mm_fp8`` / ``int8_linear`` defaults are untouched.
* Speed vs HIP: honest miss (no kitchen fused-MLP HIP baseline). Microbench vs
  unfused FlyDSL gemm+SiLU is informational only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from kernels.common.tensor_shim import _run_compiled

WM = WN = WK = 16
WAVE = 32
_TILE = 16


def _ptr(t: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _zeros8():
    return [
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
    ]


@lru_cache(maxsize=8)
def build_wmma_tile_module(dtype_name: str = "bfloat16", *, swap_ab: bool = False):
    """One wave, one 16×16×16 WMMA tile. ``swap_ab`` selects GPUOpen A/B swap."""
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    Elem = fx.BFloat16 if dtype_name == "bfloat16" else fx.Float16
    M = N = K = _TILE
    do_swap = bool(swap_ab)

    if do_swap:

        @flyc.kernel
        def wmma_tile_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            bA = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(A)),
                fx.make_layout((M, K), (K, 1)),
            )
            bB = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(B)),
                fx.make_layout((N, K), (K, 1)),
            )
            bC = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(C)),
                fx.make_layout((M, N), (N, 1)),
            )
            mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, Elem, fx.Float32))
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
            thr_mma = tiled_mma.thr_slice(tid)
            frag_A = thr_mma.make_fragment_A(bA)
            frag_B = thr_mma.make_fragment_B(bB)
            frag_C = thr_mma.make_fragment_C(bC)
            copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
            copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
            copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
            thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
            thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
            thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
            fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
            fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
            a_rm = fx.make_rmem_tensor(8, Elem)
            b_rm = fx.make_rmem_tensor(8, Elem)
            c_rm = fx.make_rmem_tensor(8, fx.Float32)
            a_rm.store(Vec(frag_A.load()))
            b_rm.store(Vec(frag_B.load()))
            c_rm.store(Vec.from_elements(_zeros8(), fx.Float32))
            # GPUOpen: wmma(B, A, C) → D N-major; std M-major store → transpose.
            fx.gemm(mma_atom, c_rm, [b_rm], [a_rm], c_rm)
            frag_C.store(Vec(c_rm.load()))
            fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    else:

        @flyc.kernel
        def wmma_tile_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            bA = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(A)),
                fx.make_layout((M, K), (K, 1)),
            )
            bB = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(B)),
                fx.make_layout((N, K), (K, 1)),
            )
            bC = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(C)),
                fx.make_layout((M, N), (N, 1)),
            )
            mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, Elem, fx.Float32))
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
            thr_mma = tiled_mma.thr_slice(tid)
            frag_A = thr_mma.make_fragment_A(bA)
            frag_B = thr_mma.make_fragment_B(bB)
            frag_C = thr_mma.make_fragment_C(bC)
            copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
            copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
            copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
            thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
            thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
            thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
            fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
            fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
            a_rm = fx.make_rmem_tensor(8, Elem)
            b_rm = fx.make_rmem_tensor(8, Elem)
            c_rm = fx.make_rmem_tensor(8, fx.Float32)
            a_rm.store(Vec(frag_A.load()))
            b_rm.store(Vec(frag_B.load()))
            c_rm.store(Vec.from_elements(_zeros8(), fx.Float32))
            fx.gemm(mma_atom, c_rm, [a_rm], [b_rm], c_rm)
            frag_C.store(Vec(c_rm.load()))
            fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    wmma_tile_kernel.__name__ = f"wmma_tile_{dtype_name}_swap{int(do_swap)}"

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        wmma_tile_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)

    launch.__name__ = f"launch_wmma_tile_{dtype_name}_swap{int(do_swap)}"
    return launch


@lru_cache(maxsize=4)
def build_fused_gemm_tn_module(dtype_name: str = "bfloat16"):
    """Zero-LDS fused ``C1 = (A0 @ B0.T) @ B1.T`` for 16×16×16 panels.

    GEMM0: A/B swapped → D0 N-major in VGPR. Narrow to act dtype. GEMM1:
    unswapped, D0 as A. Standard M-major store. No LDS transpose of D0.
    """
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    Elem = fx.BFloat16 if dtype_name == "bfloat16" else fx.Float16
    M = N = K = _TILE

    @flyc.kernel
    def fused_gemm_tn_kernel(A0: fx.Tensor, B0: fx.Tensor, B1: fx.Tensor, C1: fx.Tensor):
        tid = fx.thread_idx.x
        bA0 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(A0)),
            fx.make_layout((M, K), (K, 1)),
        )
        bB0 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(B0)),
            fx.make_layout((N, K), (K, 1)),
        )
        bB1 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(B1)),
            fx.make_layout((N, K), (K, 1)),
        )
        bC1 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(C1)),
            fx.make_layout((M, N), (N, 1)),
        )
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, Elem, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A0 = thr_mma.make_fragment_A(bA0)
        frag_B0 = thr_mma.make_fragment_B(bB0)
        frag_B1 = thr_mma.make_fragment_B(bB1)
        frag_C1 = thr_mma.make_fragment_C(bC1)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
        fx.copy(copy_a, thr_copy_A.partition_S(bA0), thr_copy_A.retile(frag_A0))
        fx.copy(copy_b, thr_copy_B.partition_S(bB0), thr_copy_B.retile(frag_B0))
        fx.copy(copy_b, thr_copy_B.partition_S(bB1), thr_copy_B.retile(frag_B1))

        a0 = fx.make_rmem_tensor(8, Elem)
        b0 = fx.make_rmem_tensor(8, Elem)
        c0 = fx.make_rmem_tensor(8, fx.Float32)
        a0.store(Vec(frag_A0.load()))
        b0.store(Vec(frag_B0.load()))
        c0.store(Vec.from_elements(_zeros8(), fx.Float32))
        # GEMM0 swapped → N-major D0 (no LDS transpose).
        fx.gemm(mma_atom, c0, [b0], [a0], c0)

        a1 = fx.make_rmem_tensor(8, Elem)
        c0v = Vec(c0.load())
        narrow = []
        for i in range_constexpr(8):
            narrow.append(fx.Float32(c0v[i]).to(Elem))
        a1.store(Vec.from_elements(narrow, Elem))

        b1 = fx.make_rmem_tensor(8, Elem)
        b1.store(Vec(frag_B1.load()))
        c1 = fx.make_rmem_tensor(8, fx.Float32)
        c1.store(Vec.from_elements(_zeros8(), fx.Float32))
        # GEMM1 unswapped: N-major D0 is already legal as A for Sequence ABI.
        fx.gemm(mma_atom, c1, [a1], [b1], c1)

        frag_C1.store(Vec(c1.load()))
        fx.copy(copy_c, thr_copy_C.retile(frag_C1), thr_copy_C.partition_S(bC1))

    fused_gemm_tn_kernel.__name__ = f"fused_gemm_tn_{dtype_name}"

    @flyc.jit
    def launch(
        A0: fx.Tensor,
        B0: fx.Tensor,
        B1: fx.Tensor,
        C1: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        fused_gemm_tn_kernel(A0, B0, B1, C1).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)

    launch.__name__ = f"launch_fused_gemm_tn_{dtype_name}"
    return launch


@lru_cache(maxsize=4)
def build_fused_swiglu_mlp_module(dtype_name: str = "bfloat16"):
    """Zero-LDS fused SwiGLU MLP for 16×16 panels (in-register SiLU×mul).

    ``Y = (silu(A0 @ Bg.T) * (A0 @ Bu.T)) @ Bd.T`` with no LDS/GMEM spill of
    the mid activation. GEMM0_gate and GEMM0_up swap A/B (N-major D in VGPR);
    Sequence ABI keeps GEMM1 **unswapped** (mid already legal as A).
    """
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    Elem = fx.BFloat16 if dtype_name == "bfloat16" else fx.Float16
    M = N = K = _TILE

    @flyc.kernel
    def fused_swiglu_mlp_kernel(
        A0: fx.Tensor,
        Bg: fx.Tensor,
        Bu: fx.Tensor,
        Bd: fx.Tensor,
        C1: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        bA0 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(A0)),
            fx.make_layout((M, K), (K, 1)),
        )
        bBg = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(Bg)),
            fx.make_layout((N, K), (K, 1)),
        )
        bBu = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(Bu)),
            fx.make_layout((N, K), (K, 1)),
        )
        bBd = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(Bd)),
            fx.make_layout((N, K), (K, 1)),
        )
        bC1 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(C1)),
            fx.make_layout((M, N), (N, 1)),
        )
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, Elem, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A0 = thr_mma.make_fragment_A(bA0)
        frag_Bg = thr_mma.make_fragment_B(bBg)
        frag_Bu = thr_mma.make_fragment_B(bBu)
        frag_Bd = thr_mma.make_fragment_B(bBd)
        frag_C1 = thr_mma.make_fragment_C(bC1)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
        fx.copy(copy_a, thr_copy_A.partition_S(bA0), thr_copy_A.retile(frag_A0))
        fx.copy(copy_b, thr_copy_B.partition_S(bBg), thr_copy_B.retile(frag_Bg))
        fx.copy(copy_b, thr_copy_B.partition_S(bBu), thr_copy_B.retile(frag_Bu))
        fx.copy(copy_b, thr_copy_B.partition_S(bBd), thr_copy_B.retile(frag_Bd))

        a0 = fx.make_rmem_tensor(8, Elem)
        bg = fx.make_rmem_tensor(8, Elem)
        bu = fx.make_rmem_tensor(8, Elem)
        cg = fx.make_rmem_tensor(8, fx.Float32)
        cu = fx.make_rmem_tensor(8, fx.Float32)
        a0.store(Vec(frag_A0.load()))
        bg.store(Vec(frag_Bg.load()))
        bu.store(Vec(frag_Bu.load()))
        cg.store(Vec.from_elements(_zeros8(), fx.Float32))
        cu.store(Vec.from_elements(_zeros8(), fx.Float32))
        # GEMM0 gate + up: swapped → N-major D in VGPR (no LDS of mid).
        fx.gemm(mma_atom, cg, [bg], [a0], cg)
        fx.gemm(mma_atom, cu, [bu], [a0], cu)

        # In-register SiLU(gate)*up (same formula as rdna4_swiglu).
        one = fx.Float32(1.0)
        neg_log2e = fx.Float32(-1.4426950408889634)
        cgv = Vec(cg.load())
        cuv = Vec(cu.load())
        mid = []
        for i in range_constexpr(8):
            g = fx.Float32(cgv[i])
            u = fx.Float32(cuv[i])
            sigv = one / (one + fmath.exp2(g * neg_log2e))
            y = g * sigv * u
            mid.append(y.to(Elem))
        a1 = fx.make_rmem_tensor(8, Elem)
        a1.store(Vec.from_elements(mid, Elem))

        bd = fx.make_rmem_tensor(8, Elem)
        bd.store(Vec(frag_Bd.load()))
        c1 = fx.make_rmem_tensor(8, fx.Float32)
        c1.store(Vec.from_elements(_zeros8(), fx.Float32))
        # GEMM1 unswapped: N-major mid is already legal as A for Sequence ABI.
        fx.gemm(mma_atom, c1, [a1], [bd], c1)

        frag_C1.store(Vec(c1.load()))
        fx.copy(copy_c, thr_copy_C.retile(frag_C1), thr_copy_C.partition_S(bC1))

    fused_swiglu_mlp_kernel.__name__ = f"fused_swiglu_mlp_{dtype_name}"

    @flyc.jit
    def launch(
        A0: fx.Tensor,
        Bg: fx.Tensor,
        Bu: fx.Tensor,
        Bd: fx.Tensor,
        C1: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        fused_swiglu_mlp_kernel(A0, Bg, Bu, Bd, C1).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)

    launch.__name__ = f"launch_fused_swiglu_mlp_{dtype_name}"
    return launch


def _dtype_name(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bfloat16"
    if dt == torch.float16:
        return "float16"
    raise ValueError(f"expected bf16/fp16, got {dt}")


def _run_tile(
    a_tile: torch.Tensor,
    b_tile: torch.Tensor,
    *,
    swap_ab: bool = False,
) -> torch.Tensor:
    out = torch.zeros((_TILE, _TILE), device=a_tile.device, dtype=torch.float32)
    launch = build_wmma_tile_module(_dtype_name(a_tile.dtype), swap_ab=swap_ab)
    _run_compiled(
        launch,
        a_tile.contiguous(),
        b_tile.contiguous(),
        out,
        torch.cuda.current_stream(device=a_tile.device),
    )
    return out


def fused_gemm_tn(
    a0: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """``C1 = (A0 @ B0.T) @ B1.T`` via zero-LDS fused WMMA (16×16 panels only)."""
    for t, name in ((a0, "a0"), (b0, "b0"), (b1, "b1")):
        if t.shape != (_TILE, _TILE):
            raise ValueError(f"{name} must be [16,16], got {tuple(t.shape)}")
    if out_dtype is None:
        out_dtype = a0.dtype
    out = torch.zeros((_TILE, _TILE), device=a0.device, dtype=torch.float32)
    launch = build_fused_gemm_tn_module(_dtype_name(a0.dtype))
    _run_compiled(
        launch,
        a0.contiguous(),
        b0.contiguous(),
        b1.contiguous(),
        out,
        torch.cuda.current_stream(device=a0.device),
    )
    return out.to(out_dtype)


def gemm_bf16_nmajor(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K].T via host-tiled 16×16 WMMA (unswapped tiles).

    Named ``nmajor`` for the zero-LDS family this module belongs to; the
    single-GEMM path uses the layout-correct unswapped tile. See
    ``fused_gemm_tn`` for the A/B-swap N-major fuse. Pads M/N/K up to 16.
    """
    if a.ndim != 2 or b_nk.ndim != 2:
        raise ValueError("expects A[M,K], B[N,K]")
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"A must be bf16/fp16, got {a.dtype}")
    if b_nk.dtype != a.dtype:
        raise ValueError(f"B dtype {b_nk.dtype} must match A {a.dtype}")
    m, k = a.shape
    n, k2 = b_nk.shape
    if k != k2:
        raise ValueError(f"K mismatch {k} vs {k2}")
    if out_dtype is None:
        out_dtype = a.dtype

    mp = (m + _TILE - 1) // _TILE * _TILE
    np_ = (n + _TILE - 1) // _TILE * _TILE
    kp = (k + _TILE - 1) // _TILE * _TILE
    if mp != m or np_ != n or kp != k:
        a_pad = torch.zeros((mp, kp), device=a.device, dtype=a.dtype)
        b_pad = torch.zeros((np_, kp), device=a.device, dtype=a.dtype)
        a_pad[:m, :k] = a
        b_pad[:n, :k] = b_nk
    else:
        a_pad, b_pad = a.contiguous(), b_nk.contiguous()

    acc = torch.zeros((mp, np_), device=a.device, dtype=torch.float32)
    for i in range(0, mp, _TILE):
        for j in range(0, np_, _TILE):
            tile_acc = torch.zeros((_TILE, _TILE), device=a.device, dtype=torch.float32)
            for k0 in range(0, kp, _TILE):
                tile_acc += _run_tile(
                    a_pad[i : i + _TILE, k0 : k0 + _TILE],
                    b_pad[j : j + _TILE, k0 : k0 + _TILE],
                    swap_ab=False,
                )
            acc[i : i + _TILE, j : j + _TILE] = tile_acc

    result = acc[:m, :n].to(out_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    try:
        from kernels.quant.rdna4_swiglu import build_silu_mul_module

        out = torch.empty_like(gate)
        launch = build_silu_mul_module(dtype=_dtype_name(gate.dtype))
        _run_compiled(
            launch,
            _ptr(gate),
            _ptr(up),
            _ptr(out),
            int(gate.numel()),
            torch.cuda.current_stream(device=gate.device),
        )
        return out
    except Exception:
        return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def fused_swiglu_mlp_inreg(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """In-register SwiGLU MLP for ``K == FFN == 16`` panels (host-tiles M/N).

    Computes ``silu(x @ Wgate.T) * (x @ Wup.T)`` then ``@ Wdown.T`` with
    weights ``[N, K]`` (linear layout). Each 16×16 (M,N) tile keeps mid
    in VGPRs (no LDS/GMEM spill). Requires ``K == FFN == 16`` and ``M``,
    ``N_out`` multiples of 16 (e.g. 32×16). Larger ``K``/``FFN`` need an
    in-kernel K/FFN loop — use ``fused_swiglu_mlp_nmajor`` instead.
    """
    if out_dtype is None:
        out_dtype = x.dtype
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).contiguous()
    m, k = x2d.shape
    ffn, k_g = w_gate.shape
    ffn_u, k_u = w_up.shape
    n_out, k_d = w_down.shape
    if w_gate.dtype != x.dtype or w_up.dtype != x.dtype or w_down.dtype != x.dtype:
        raise ValueError("weight dtypes must match x")
    if k != _TILE or k_g != _TILE or k_u != _TILE or k_d != _TILE:
        raise ValueError(
            "fused_swiglu_mlp_inreg requires K == 16 "
            f"(got xK={k} gateK={k_g} upK={k_u} downK={k_d}); "
            "use fused_swiglu_mlp_nmajor for other sizes"
        )
    if ffn != _TILE or ffn_u != _TILE:
        raise ValueError(
            "fused_swiglu_mlp_inreg requires FFN == 16 "
            f"(got gate={ffn} up={ffn_u}); "
            "use fused_swiglu_mlp_nmajor for other sizes"
        )
    if m % _TILE != 0 or n_out % _TILE != 0:
        raise ValueError(
            "fused_swiglu_mlp_inreg requires M and N_out multiples of 16 "
            f"(got M={m} N_out={n_out}); "
            "use fused_swiglu_mlp_nmajor for other sizes"
        )

    launch = build_fused_swiglu_mlp_module(_dtype_name(x.dtype))
    stream = torch.cuda.current_stream(device=x.device)
    w_gate_c = w_gate.contiguous()
    w_up_c = w_up.contiguous()
    w_down_c = w_down.contiguous()
    out_f32 = torch.zeros((m, n_out), device=x.device, dtype=torch.float32)
    # Host-tile along M and N: each launch is the true in-reg 16×16 fuse.
    # N>16 recomputes GEMM0+SiLU per N tile (mid not shared across launches).
    for i0 in range(0, m, _TILE):
        x_tile = x2d[i0 : i0 + _TILE, :].contiguous()
        for j0 in range(0, n_out, _TILE):
            wd_tile = w_down_c[j0 : j0 + _TILE, :].contiguous()
            tile_out = torch.zeros((_TILE, _TILE), device=x.device, dtype=torch.float32)
            _run_compiled(
                launch,
                x_tile,
                w_gate_c,
                w_up_c,
                wd_tile,
                tile_out,
                stream,
            )
            out_f32[i0 : i0 + _TILE, j0 : j0 + _TILE] = tile_out
    return out_f32.to(out_dtype).reshape(*orig[:-1], n_out)


def fused_swiglu_mlp_nmajor(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Thin SwiGLU MLP host on tiled WMMA gemm + silu_mul.

    Separate launches. For exact 16×16 panels prefer
    ``fused_swiglu_mlp_inreg`` (true in-register SiLU×mul, no mid spill).
    Weights are ``[N, K]`` (same as linear ``B[N,K]``).
    """
    if out_dtype is None:
        out_dtype = x.dtype
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).contiguous()
    gate = gemm_bf16_nmajor(x2d, w_gate, out_dtype=x.dtype)
    up = gemm_bf16_nmajor(x2d, w_up, out_dtype=x.dtype)
    mid = _silu_mul(gate, up)
    y = gemm_bf16_nmajor(mid, w_down, out_dtype=out_dtype)
    return y.reshape(*orig[:-1], w_down.shape[0])


def reference_swiglu_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Eager reference: two GEMMs + SiLU×mul + down (f32 math)."""
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).float()
    gate = x2d @ w_gate.float().T
    up = x2d @ w_up.float().T
    mid = F.silu(gate) * up
    y = mid @ w_down.float().T
    return y.reshape(*orig[:-1], w_down.shape[0])
