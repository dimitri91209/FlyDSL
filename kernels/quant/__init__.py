# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""RDNA4 quantization and elementwise kernels (gfx1201).

Includes per-tensor FP8 quant/dequant (e4m3 **and** e5m2 via ``e5m2=``),
stochastic FP8 rounding (same format flag), and plain SiLU*mul / chunk-2
SwiGLU. Shared buffer helpers live in ``rdna4_common``.

Also ``rdna4_int8_convrot``: kitchen-compatible ``quantize_int8_convrot_weight`` /
``quantize_and_rotate_rowwise`` (Hadamard G∈{16,64,256} + rowwise INT8). Host
``int8_linear_convrot`` needs the PR-B iu8 GEMM (skipped in suite-only tests).
``convrot_w4a4`` remains a separate packed format (follow-up).
"""
