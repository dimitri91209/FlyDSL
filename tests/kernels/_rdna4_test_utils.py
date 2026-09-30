# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Shared launch helpers for the raw-pointer gfx120x / RDNA4 kernel tests.

``ptr`` wraps a torch CUDA tensor's device pointer for FlyDSL raw-pointer
kernels. ``run`` compiles a launch once and reuses the cached callable on
later calls in the same process.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx


def ptr(tensor):
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


def run(launch, *args):
    compiled = getattr(launch, "_cf", None)
    if compiled is None:
        launch._cf = flyc.compile(launch, *args)
    else:
        compiled(*args)
