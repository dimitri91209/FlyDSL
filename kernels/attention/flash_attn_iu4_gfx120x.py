# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""iu4 FA kitchen-pack: prefer native iu4 WMMA FA; fallback unpack→iu8."""

from __future__ import annotations

KERNEL_NAME = "flash_attn_func_iu4_gfx120x_kernel"
KERNEL_NAME_GFX1201 = "flash_attn_func_iu4_gfx1201_kernel"
_NATIVE_BUILD_ERROR = None


def build_flash_attn_func_iu4_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="iu4",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
    prefer_native=True,
):
    global _NATIVE_BUILD_ERROR
    if prefer_native:
        try:
            from kernels.attention._flash_attn_iu4_native_gfx120x import build_flash_attn_func_iu4_native_module

            return build_flash_attn_func_iu4_native_module(
                num_heads=num_heads,
                head_dim=head_dim,
                causal=causal,
                dtype_str=dtype_str,
                sm_scale=sm_scale,
                waves_per_eu=waves_per_eu,
                flat_work_group_size=flat_work_group_size,
                block_m=block_m,
                block_n=block_n,
                unsafe_fp_math=unsafe_fp_math,
                fast_fp_math=fast_fp_math,
                daz=daz,
                path_tag=path_tag,
            )
        except Exception as exc:
            _NATIVE_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
    from kernels.attention.flash_attn_int8_gfx120x import build_flash_attn_func_int8_module

    return build_flash_attn_func_int8_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="int8",
        sm_scale=sm_scale,
        waves_per_eu=waves_per_eu,
        flat_work_group_size=flat_work_group_size,
        block_m=block_m,
        block_n=block_n,
        unsafe_fp_math=unsafe_fp_math,
        fast_fp_math=fast_fp_math,
        daz=daz,
        path_tag=path_tag,
    )


build_flash_attn_func_iu4_module = build_flash_attn_func_iu4_module_primary
build_flash_attn_func_iu4_module_gfx1201 = build_flash_attn_func_iu4_module_primary
build_flash_attn_func_iu4_module_gfx120x = build_flash_attn_func_iu4_module_primary


def native_iu4_build_error():
    return _NATIVE_BUILD_ERROR
