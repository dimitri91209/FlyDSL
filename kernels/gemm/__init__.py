# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FlyDSL gemm kernels.

RDNA4 (gfx1201) entries in this tree include:

* ``rdna4_scaled_mm_fp8`` — tensorwise FP8 e4m3 / e5m2 scaled_mm
* ``rdna4_scaled_mm_fp8_fused`` — fused act-quant ⊕ scaled_mm; N-adapter
  LoRA via host bf16 residuals (in-kernel LoRA idle-loses vs HIP on gfx1201)

INT8 / iu8 GEMM lives on the stacked ``gfx1201-iu8-int8`` branch.
"""
