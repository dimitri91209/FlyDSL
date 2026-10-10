# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x AdaLN (scale/shift after RMS or LayerNorm).

RoPE is ``examples/12``. This launches ``build_adaln_module`` with ``fx.Tensor``
args (not the raw-pointer RoPE ABI).

gfx120x only.
"""

import sys

import torch

import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from kernels.norm.adaln_gfx120x import build_adaln_module

_arch = str(get_rocm_arch() or "")
if not _arch.startswith("gfx120"):
    print(f"SKIP {__file__}: needs gfx120x, got {_arch or '<unknown>'}")
    sys.exit(0)

EPS = 1e-5
rows, width = 2, 256
x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
scale = torch.randn_like(x) * 0.1
shift = torch.randn_like(x) * 0.1
out = torch.empty_like(x)
args = (x, scale, shift, out, rows, 1, 1, EPS, torch.cuda.current_stream())
compiled = flyc.compile(build_adaln_module(width, "bfloat16", False, block_threads=256), *args)
compiled(*args)
torch.cuda.synchronize()
xf = x.float()
norm = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
ref = norm * (1.0 + scale.float()) + shift.float()
ok = torch.allclose(out.float(), ref, atol=5e-2, rtol=5e-2)
print(f"arch={_arch} adaln", tuple(out.shape), out.dtype, "correct:", ok)
if not ok:
    print("Max diff:", (out.float() - ref).abs().max().item())
    sys.exit(1)
