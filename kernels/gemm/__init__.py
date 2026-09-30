# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FlyDSL gemm kernels.

RDNA4 (gfx1201) entries in this tree include:

* ``rdna4_scaled_mm_fp8`` — tensorwise FP8 e4m3 / e5m2 scaled_mm
* ``rdna4_scaled_mm_fp8_gated`` — experimental size/shape tile gate → ungated
* ``rdna4_scaled_mm_fp8_fused`` — fused act-quant ⊕ scaled_mm; N-adapter
  LoRA via host bf16 residuals (in-kernel LoRA idle-loses vs HIP on gfx1201)
* ``rdna4_w8a16_path_a`` — W8A16 Path A (bf16 WMMA; no iu8 atom)
* ``rdna4_int8_ab_gate`` — host Path A/B MNK size gate (Path B needs tip B)
* ``rdna4_int8_ab_gated`` — opt-in ``*_gated`` surface wrapping AB M/M*K rules

INT8 / iu8 Path B GEMM lives on the stacked ``gfx1201-iu8-int8`` branch.

``*_gated`` modules are EXPERIMENTAL gfx1201 size/shape gates (lab breakpoints;
not a general FlyDSL autotune API; Comfy may own product env). Ungated siblings
remain the default tip surface.
"""
