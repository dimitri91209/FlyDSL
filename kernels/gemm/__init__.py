# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FlyDSL gemm kernels.

RDNA4 (gfx1201) entries in this tree include:

* ``rdna4_scaled_mm_fp8`` — tensorwise FP8 e4m3 scaled_mm
* ``rdna4_scaled_mm_fp8_fused`` — fused act-quant ⊕ scaled_mm (+ optional LoRA)
"""
