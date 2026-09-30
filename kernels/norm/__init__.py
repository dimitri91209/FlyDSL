# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx1201 norm, RoPE, and AdaLN kernels.

FlyDSL-native RMS/RoPE/AdaLN implementations for RDNA4. Prefer fused
``rms_rope_gfx1201`` when both RMSNorm and RoPE run in the same layer.
"""
