# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
################################################################################
# WARNING — EXPERIMENTAL SIZE/SHAPE GATE FOR GFX120X
#
# THIS MODULE IS AN EXPERIMENTAL SIZE/SHAPE GATE MEASURED LOCALLY ON GFX120X
# (RDNA4). BREAKPOINTS WERE MEASURED IN IDLE MEASUREMENTS; THIS IS NOT A GENERAL
# FLYDSL AUTOTUNE API. PRODUCT INTEGRATIONS MAY STILL OWN PRODUCTION WIRING AND
# FALLBACKS. CALLERS MUST NOT TREAT THIS AS A PRODUCTION DEFAULT WITHOUT READING
# THE GATE RULES BELOW AND THE MATCHING IDLE RESULTS.
#
# PLAIN KERNEL SIBLING: kernels/gemm/rdna4_scaled_mm_fp8.py REMAINS THE DEFAULT PATH
# SURFACE. THIS *_auto MODULE IS ADDITIVE / OPT-IN ONLY.
################################################################################
"""Size-dispatcher host for gfx120x FP8 ``scaled_mm`` (tile pick -> plain kernel).

Wraps measured tile breakpoints in ``pick_tile_config`` (skinny BK on tiny
shapes; deep-K on fat K>=512). Host ``scaled_mm_fp8_auto`` calls plain
``build_scaled_mm_fp8_module``. On wanish ``[1024,5120,5120]`` the picker moves
plain PARITY (~x1.00) to about x3.97 WIN -- see
``docs/gfx120x_idle_speed_vs_hip.md``.

Shorthand: gate / ``_gated`` aliases are fine; this is a measured size rule,
not a general FlyDSL autotune API.

Credit: dimitri91209 + Grokbot.
"""

from typing import Literal, Optional

import torch

from kernels.common.dispatch_mode import get_dispatch_mode
from kernels.common.gfx120x_arch import require_gfx120x

from kernels.gemm.rdna4_scaled_mm_fp8 import (
    TileConfig,
    build_scaled_mm_fp8_module,
    pick_tile_config,
)

KERNEL_NAME = "rdna4_scaled_mm_fp8_auto"

TilePath = Literal[
    "skinny_bk128",
    "skinny_bk64",
    "fat_deep_k_a",
    "fat_deep_k_b",
    "fat_bk64",
]

# Documented smoke anchors (lab / FULL_SUITE / PARITY_CONFIRM 2026-09-30).
_SMOKE_ANCHORS = {
    "tiny_k128_expect_skinny_bk128": {
        "M": 32,
        "N": 128,
        "K": 128,
        "why": "skinny + K>=128 → 64x64x128",
    },
    "tiny_k64_expect_skinny_bk64": {
        "M": 32,
        "N": 128,
        "K": 64,
        "why": "skinny + K<128 → 64x64x64",
    },
    "mid_fat_underfill_expect_skinny_bk128": {
        "M": 128,
        "N": 512,
        "K": 512,
        "why": "underfilled fat grid → plain K>=128 fallback 64x64x128",
    },
    "wanish_expect_fat_deep_k": {
        "M": 1024,
        "N": 5120,
        "K": 5120,
        "why": "fat deep-K; inform-loser vs HIP still ships Fly",
    },
}

def _tile_path_name(cfg: TileConfig, M: int, N: int, K: int, wgps: int) -> TilePath:
    """Map a concrete TileConfig to a stable gate-table label (by tile identity)."""
    del M, N, K, wgps  # labels follow cfg; MNK already applied in pick_tile_config
    if cfg.bm == 64 and cfg.bn == 64 and cfg.bk == 128:
        return "skinny_bk128"
    if cfg.bm == 64 and cfg.bn == 64 and cfg.bk == 64:
        return "skinny_bk64"
    if cfg.bm == 128 and cfg.bn == 128 and cfg.bk == 128 and cfg.warps_n == 4:
        return "fat_deep_k_a"
    if cfg.bm == 128 and cfg.bn == 128 and cfg.bk == 128:
        return "fat_deep_k_b"
    if cfg.bm == 128 and cfg.bn == 128 and cfg.bk == 64:
        return "fat_bk64"
    # Fallback: treat unknown as fat_bk64-class for force_path symmetry.
    return "fat_bk64"

def select_path(
    M: int,
    N: int,
    K: int,
    *,
    wgps: Optional[int] = None,
    force_path: Optional[TilePath] = None,
) -> TilePath:
    """Cheap static tile-path gate (no runtime autotune).

    Delegates size→tile to plain ``pick_tile_config`` (established breakpoints),
    then labels the result for gate-table tests / docs.
    """
    if force_path is not None:
        allowed = (
            "skinny_bk128",
            "skinny_bk64",
            "fat_deep_k_a",
            "fat_deep_k_b",
            "fat_bk64",
        )
        if force_path not in allowed:
            raise ValueError(f"force_path must be one of {allowed}, got {force_path!r}")
        return force_path
    if wgps is None:
        # Match plain default without importing private cache.
        try:
            wgps = int(torch.cuda.get_device_properties(0).multi_processor_count) or 32
        except Exception:  # noqa: BLE001
            wgps = 32
    cfg = pick_tile_config(int(M), int(N), int(K), wgps=int(wgps))
    return _tile_path_name(cfg, int(M), int(N), int(K), int(wgps))

def select_tile(
    M: int,
    N: int,
    K: int,
    *,
    wgps: Optional[int] = None,
    force_path: Optional[TilePath] = None,
) -> TileConfig:
    """Return the plain ``TileConfig`` for MNK (honors ``force_path`` labels)."""
    if force_path is None:
        return pick_tile_config(int(M), int(N), int(K), wgps=wgps)
    # Map force labels onto the same TileConfig constants the plain picker uses.
    from kernels.gemm import rdna4_scaled_mm_fp8 as _u

    table = {
        "skinny_bk128": _u._CFG_64_64_128,
        "skinny_bk64": _u._CFG_64_64_64,
        "fat_deep_k_a": _u._CFG_128_128_128_A,
        "fat_deep_k_b": _u._CFG_128_128_128_B,
        "fat_bk64": _u._CFG_128_128_64,
    }
    if force_path not in table:
        raise ValueError(f"unknown force_path {force_path!r}")
    return table[force_path]

def gate_rule_doc(*, wgps: int = 32) -> dict:
    """Structured description of the active scaled_mm tile gate."""
    return {
        "wgps_assumed_for_doc": wgps,
        "rule": (
            "force_path if set; else plain pick_tile_config(M,N,K,wgps) "
            "labeled as skinny_bk128|skinny_bk64|fat_deep_k_a|fat_deep_k_b|fat_bk64"
        ),
        "plain": "kernels.gemm.rdna4_scaled_mm_fp8",
        "host": "scaled_mm_fp8_auto",
        "smoke_anchor": dict(_SMOKE_ANCHORS),
        "disclaimer": (
            "EXPERIMENTAL gfx120x size/shape gate; not a general FlyDSL autotune API; "
            "Not a production default without reading gate rules"
        ),
    }

def _out_dtype_name(dtype: torch.dtype) -> str:
    return {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[dtype]

def scaled_mm_fp8_auto(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    e5m2: bool = False,
    wgps: Optional[int] = None,
    force_path: Optional[TilePath] = None,
) -> torch.Tensor:
    """Gated host: select tile/path then launch plain ``scaled_mm_fp8``.

    ``a`` / ``b_nk`` are packed FP8 (``float8_e4m3fn`` or ``float8_e5m2``);
    ``b_nk`` is [N, K]. Scales are per-tensor float32.
    """
    require_gfx120x(a.device, what='scaled_mm_fp8_auto (gfx120x size-dispatch)')
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.tensor_shim import _run_compiled

    if a.ndim != 2 or b_nk.ndim != 2:
        raise ValueError("a and b_nk must be 2D [M,K] and [N,K]")
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b_nk.shape[0])
    if int(b_nk.shape[1]) != k:
        raise ValueError(f"inner dim mismatch: a K={k}, b K={b_nk.shape[1]}")

    # FLYDSL_DISPATCH_MODE bypass (force_path still wins when set).
    if force_path is None and get_dispatch_mode() == "force_hip":
        from comfy_kitchen.scaled_mm_v2 import scaled_mm_v2

        return scaled_mm_v2(
            a,
            b_nk.T,
            scale_a,
            scale_b,
            out_dtype=out_dtype,
        )

    cfg = select_tile(m, n, k, wgps=wgps, force_path=force_path)
    skip_bounds = m % cfg.bm == 0 and n % cfg.bn == 0
    launch = build_scaled_mm_fp8_module(_out_dtype_name(out_dtype), cfg, skip_bounds, e5m2=e5m2)
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    scale_a = scale_a.to(device=a.device, dtype=torch.float32).reshape(1).contiguous()
    scale_b = scale_b.to(device=a.device, dtype=torch.float32).reshape(1).contiguous()

    def _ptr(t: torch.Tensor):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    _run_compiled(
        launch,
        _ptr(a.view(torch.uint8)),
        _ptr(b_nk.view(torch.uint8)),
        _ptr(out.view(torch.uint8)),
        _ptr(scale_a),
        _ptr(scale_b),
        m,
        n,
        k,
        torch.cuda.current_stream(device=a.device),
    )
    return out

__all__ = [
    "KERNEL_NAME",
    "TilePath",
    "select_path",
    "select_tile",
    "gate_rule_doc",
    "scaled_mm_fp8_auto",
]

# Thin alias: gate/_gated shorthand for size-dispatcher callers.
scaled_mm_fp8_gated = scaled_mm_fp8_auto
