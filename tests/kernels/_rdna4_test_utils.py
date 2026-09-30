# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Shared launch helpers for the raw-pointer RDNA4 kernel tests."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch


def ptr(tensor):
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


def run(launch, *args):
    compiled = getattr(launch, "_cf", None)
    if compiled is None:
        launch._cf = flyc.compile(launch, *args)
    else:
        compiled(*args)
