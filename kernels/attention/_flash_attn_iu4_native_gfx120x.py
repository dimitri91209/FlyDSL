# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
def build_flash_attn_func_iu4_native_module(**kwargs):
    from kernels.attention._flash_attn_iu4_native_body_gfx120x import build_flash_attn_func_iu4_native_body

    return build_flash_attn_func_iu4_native_body(**kwargs)
