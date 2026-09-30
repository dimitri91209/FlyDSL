# SPDX-License-Identifier: Apache-2.0
"""Alias: prefer flash_attn_fp8_gfx120x."""

from kernels.attention.flash_attn_fp8_gfx120x import *  # noqa: F403
from kernels.attention.flash_attn_fp8_gfx120x import (  # noqa: F401
    KERNEL_NAME,
    build_flash_attn_func_fp8_module,
    build_flash_attn_func_fp8_module_primary,
)
