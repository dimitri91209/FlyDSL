# SPDX-License-Identifier: Apache-2.0
"""Compat alias for the gfx120x family Flash Attention module.

Prefer ``kernels.attention.flash_attn_gfx120x``. Hardware may still report
``gfx1201`` in ``gcnArchName``; the family name is gfx120x.
"""

from kernels.attention.flash_attn_gfx120x import *  # noqa: F403
from kernels.attention.flash_attn_gfx120x import (  # noqa: F401
    KERNEL_NAME,
    build_flash_attn_func_module,
    build_flash_attn_func_module_primary,
)
