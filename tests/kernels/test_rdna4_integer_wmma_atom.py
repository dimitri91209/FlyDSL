#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness tests for gfx120x (RDNA4) integer WMMA atoms.

RDNA4 uses the v8 register ABI: each lane holds 8 i8 A/B elements
(vector<2xi32> packed), not the gfx11 duplicated 16-element fragments.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl  # noqa: E402,F401
import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl._mlir.dialects import fly  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 integer WMMA requires gfx120*, got {_ARCH}", allow_module_level=True)

WAVE_SIZE = 32
M = N = K = 16


def _compile_single_integer_wmma(*, sign_a, sign_b):
    """Build C[16,16] = A[16,16] @ B[16,16].T with one gfx120x iu8 atom."""

    @flyc.kernel
    def wmma_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        lane = fx.thread_idx.x
        lane16 = lane % 16
        lane_half = lane // 16

        c2d = fx.make_view(fx.get_iter(C), fx.make_layout((M, N), (N, 1)))
        a2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, K), (K, 1)))
        b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1)))

        # RDNA4 v8 ABI: K = (lane/16)*8 + val  → 8 elements per lane.
        a_vec = fx.Vector.from_elements(
            [a2d[lane16, lane_half * 8 + k].to(fx.Int8) for k in fx.range_constexpr(8)],
            fx.Int8,
        )
        b_vec = fx.Vector.from_elements(
            [b2d[lane16, lane_half * 8 + k].to(fx.Int8) for k in fx.range_constexpr(8)],
            fx.Int8,
        )
        acc = fx.Vector.filled(8, 0, fx.Int32)

        mma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(M, N, K, fx.Int8, fx.Int32, sign_a=sign_a, sign_b=sign_b)
        )
        result = fx.Vector(
            fly.mma_atom_call_ssa(
                [fx.Vector.make_type(8, fx.Int32)],
                mma_atom,
                a_vec.ir_value(),
                b_vec.ir_value(),
                acc.ir_value(),
            )
        )

        # C layout (gfx1250 helper): M = (lane/16)*8 + v, N = lane%16
        for value_idx in fx.range_constexpr(8):
            row = lane_half * 8 + value_idx
            c2d[row, lane16] = result[value_idx]

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        wmma_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize(
    "sign_a, sign_b",
    [(True, True), (False, False), (True, False)],
    ids=["signed", "unsigned", "mixed_sign"],
)
def test_single_iu8_wmma_atom(sign_a, sign_b):
    torch.manual_seed(0)
    # Keep values in a range that is exact for both signed and unsigned interpret.
    lo, hi = (-8, 8) if (sign_a or sign_b) else (0, 15)
    a = torch.randint(lo, hi, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(lo, hi, (N, K), device="cuda", dtype=torch.int8)
    c = torch.zeros(M, N, dtype=torch.int32, device="cuda")

    launch = _compile_single_integer_wmma(sign_a=sign_a, sign_b=sign_b)
    launch(a, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    a_ref = a.to(torch.int32)
    b_ref = b.to(torch.int32)
    if not sign_a:
        a_ref = a.to(torch.uint8).to(torch.int32)
    if not sign_b:
        b_ref = b.to(torch.uint8).to(torch.int32)
    # ROCm PyTorch lacks int32 addmm; compute reference in float32 then cast.
    ref = (a_ref.to(torch.float32) @ b_ref.to(torch.float32).T).to(torch.int32)
    torch.testing.assert_close(c, ref, atol=0, rtol=0)


if __name__ == "__main__":
    test_single_iu8_wmma_atom(True, True)
