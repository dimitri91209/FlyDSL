# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FlyDSL gemm kernels.

RDNA4 (gfx1201) entries in this tree include:

* ``rdna4_scaled_mm_fp8`` — tensorwise FP8 e4m3 / e5m2 scaled_mm
  (``e5m2=`` specialize flag; ``float8_e4m3fn`` or ``float8_e5m2`` operands)
* ``rdna4_scaled_mm_fp8_fused`` — fused act-quant ⊕ scaled_mm (+ optional LoRA);
  host infers format from weight dtype (bf8 path when e5m2);
  ``scaled_mm_fp8_fused_multi`` for N adapters in load order
"""
