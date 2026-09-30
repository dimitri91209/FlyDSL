# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""RDNA4 quantization and elementwise kernels (gfx1201).

Includes per-tensor FP8 quant/dequant (e4m3 **and** e5m2 via ``e5m2=``),
stochastic FP8 rounding (same format flag), and plain SiLU*mul / chunk-2
SwiGLU. Shared buffer helpers live in ``rdna4_common``.
"""
