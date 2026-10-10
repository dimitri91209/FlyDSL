# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x dense FlashAttention fp8 / int8 hosts.

Sibling of ``examples/08`` (bf16). Descales stay on device.

gfx120x only.
"""

import sys

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.flash_attn_gfx120x_host import (
    flydsl_flash_attn_fp8_func,
    flydsl_flash_attn_int8_func,
)

_arch = str(get_rocm_arch() or "")
if not _arch.startswith("gfx120"):
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)


B, S, H, D = 1, 64, 2, 64
qf = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)

# fp8 e4m3fn
qs = float(qf.abs().amax().clamp(min=1e-12) / 448.0)
q8 = (qf / qs).to(torch.float8_e4m3fn)
o8 = flydsl_flash_attn_fp8_func(q8, q8, q8, causal=False, q_descale=qs, k_descale=qs, v_descale=qs)

# int8
si = float(qf.abs().amax().clamp(min=1e-12) / 127.0)
qi = (qf / si).clamp(-128, 127).round().to(torch.int8)
oi = flydsl_flash_attn_int8_func(qi, qi, qi, causal=False, q_descale=si, k_descale=si, v_descale=si)

torch.cuda.synchronize()
ok = (
    o8.shape == (B, S, H, D)
    and oi.shape == (B, S, H, D)
    and bool(torch.isfinite(o8.float()).all())
    and bool(torch.isfinite(oi.float()).all())
)
print(f"arch={_arch} fa_fp8/int8", o8.shape, o8.dtype, "correct:", ok)
if not ok:
    sys.exit(1)
