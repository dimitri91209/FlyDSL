Pre-built kernels
=================

The FlyDSL repository includes a collection of pre-built GPU kernels in the
``kernels/`` directory, organized into subpackages (``gemm/``, ``norm/``,
``attention/``, ``moe/``, ``mma/``, ``common/``, ``comm/``, ``conv/``, ``quant/``).
These serve as both ready-to-use components and reference implementations for
kernel development.

The ``kernels`` tree is used from a source checkout; it is not installed by the
``flydsl`` wheel and is not covered by the stable Python API policy. Pin the
repository revision when integrating one of these entry points.

GEMM kernels
-------------

- ``kernels.gemm.preshuffle_gemm`` -- MFMA-based GEMM with LDS pipeline and pre-shuffled weights (FP8, INT8, FP16, BF16)
- ``kernels.gemm.mxfp4_preshuffle`` -- MXFP4 / FP4 (and f8f4) preshuffle GEMM
- ``kernels.gemm.fp4_gemm_4wave`` -- 4-wave FP4 GEMM (gfx950)
- ``kernels.gemm.rdna_f16_gemm`` -- RDNA f16/bf16 WMMA GEMM (``create_wmma_gemm_module``)
- ``kernels.gemm.rdna_fp8_preshuffle_gemm`` -- RDNA4 FP8 preshuffle GEMM (``compile_fp8_gemm``). e4m3, small M, no LDS
- ``kernels.gemm.rdna4_scaled_mm_fp8`` -- gfx120x FP8 scaled GEMM (``scaled_mm_fp8``). e4m3×e4m3 or e5m2×e5m2
- ``kernels.gemm.rdna4_scaled_mm_fp8_fused`` -- gfx120x fused activation quant + scaled FP8 GEMM (``scaled_mm_fp8_fused``, ``scaled_mm_fp8_fused_multi``)
- ``kernels.gemm.rdna4_w8a16_linear`` -- gfx120x bf16/fp16 activations × int8, e4m3, or e5m2 weights (``w8a16_linear``, ``w8a16_gemm``)
- ``kernels.gemm.rdna4_int8_linear`` -- gfx120x int8×int8 linear (``int8_linear``)
- ``kernels.gemm.rdna4_int8_linear_fused`` -- gfx120x fused activation quant + int8 linear (``int8_linear_fused``, ``int8_linear_fused_multi``)
- ``kernels.gemm.rdna4_iu4_gemm`` -- gfx120x packed int4 GEMM (``iu4_gemm``)
- ``kernels.gemm.rdna4_mxfp8_block_gemm`` -- gfx120x MXFP8 block-scaled GEMM (``mxfp8_block_gemm``)
- ``kernels.gemm.rdna4_mxfp4_block_gemm`` -- gfx120x MXFP4 block-scaled GEMM (``mxfp4_block_gemm``)
- ``kernels.gemm.rdna4_fused_mlp_nmajor`` -- gfx120x fp16/bf16 GEMM and SwiGLU (``gemm_bf16_nmajor``, ``gemm_bf16_nmajor_lds``, ``fused_gemm_tn``, ``fused_swiglu_mlp_nmajor``, ``fused_swiglu_mlp_lds``, ``fused_swiglu_mlp_inreg``)

MoE (Mixture-of-Experts) kernels
----------------------------------

- ``kernels.moe.moe_gemm_2stage`` -- fp8 MoE GEMM with 2-stage pipeline (stage1 gate-up +
  stage2 down-projection), gfx94*/gfx95*. Also provides the MoE reduction (sum over the topk
  dimension, ``Y[t, d] = sum(X[t, :, d])``), compiled via ``compile_moe_reduction()``.
- ``kernels.moe.mxfp_moe`` -- Fused a4w4 / a8w4 MoE 2-stage GEMM (device-side fp4 re-quant)

Paged attention
----------------

- ``kernels.attention.pa_decode_fp8`` -- Paged attention decode kernel with FP8 support

FlashAttention
--------------

- ``kernels.attention.flash_attn_gfx120x`` -- gfx120x bf16/fp16 FlashAttention kernel (dense, varlen, paged, including vectorized)
- ``kernels.attention.flash_attn_fp8_gfx120x`` -- gfx120x fp8 and int8 FlashAttention kernel. ``flash_attn_int8_gfx120x`` calls this kernel
- ``kernels.attention.flash_attn_gfx120x_host`` -- gfx120x launches: ``flydsl_flash_attn_func``, ``flydsl_flash_attn_varlen_func``, ``flydsl_flash_attn_paged_func``, ``flydsl_flash_attn_varlen_paged_func``, ``flydsl_flash_attn_fp8_func``, ``flydsl_flash_attn_int8_func``, ``flydsl_flash_attn_fp8_varlen_func``, ``flydsl_flash_attn_int8_varlen_func``, ``flydsl_flash_attn_quant_paged_func``. The shared ``flydsl_flash_attn_func`` enters here on gfx120x. Score mask, online softmax, and LSE live in this module
- ``kernels.attention.flash_attn_gfx120x_splitk`` -- gfx120x split-K combine (dense only)
- ``kernels.attention.gfx120x_alibi_bias`` -- ALiBi bias (``fill_alibi_bias``)
- ``kernels.attention.gfx120x_attn_mask`` -- bool mask to additive bias (``bool_mask_to_additive``)

Normalization
-------------

- ``kernels.norm.layernorm_kernel`` -- Layer normalization
- ``kernels.norm.rmsnorm_kernel`` -- RMS normalization
- ``kernels.norm.rope_gfx120x`` -- gfx120x RoPE (``build_rope_module``, ``build_rope_split_module``, ``build_rope_qk_fused_module``, ``build_rope_split_half_qk_fused_module``)
- ``kernels.norm.rms_rope_gfx120x`` -- gfx120x RMSNorm + RoPE (``build_rms_rope_module``, ``build_rms_rope_split_module``, ``build_rms_rope_qk_fused_module``, ``build_rms_rope_split_qk_fused_module``)
- ``kernels.norm.adaln_gfx120x`` -- gfx120x AdaLN (``build_adaln_module``)

Quantization
------------

- ``kernels.quant.rdna4_fp8_quant`` -- gfx120x per-tensor FP8 (``fp8_quant_direct``, ``dequantize_fp8``). e4m3 and e5m2
- ``kernels.quant.rdna4_stoch_fp8`` -- gfx120x stochastic FP8 cast (``stoch_fp8_direct``). One-way, no stored scale
- ``kernels.quant.rdna4_mxfp8_e8m0`` -- gfx120x MXFP8 (``quantize_mxfp8_device``, ``dequantize_mxfp8_device``)
- ``kernels.quant.rdna4_mxfp4_e2m1`` -- gfx120x MXFP4 (``quantize_mxfp4_device``, ``dequantize_mxfp4_device``)
- ``kernels.quant.rdna4_quantize_int8_rowwise`` -- gfx120x per-row int8 (``quantize_int8_rowwise``, ``dequantize_int8_rowwise``)
- ``kernels.quant.rdna4_quantize_int8_tensorwise`` -- gfx120x per-tensor int8 (``quantize_int8_tensorwise``, ``dequantize_int8_tensorwise``)
- ``kernels.quant.rdna4_int8_convrot`` -- gfx120x int8 ConvRot (``quantize_int8_convrot_weight``, ``dequantize_int8_convrot_weight``, ``int8_linear_convrot``, ``convrot_fwht``)
- ``kernels.quant.rdna4_convrot_w4a4`` -- gfx120x ConvRot W4A4 (``quantize_convrot_w4a4_weight``, ``dequantize_convrot_w4a4_weight``, ``convrot_w4a4_linear``, ``expand_signed_i4``). Default linear is packed int4
- ``kernels.quant.rdna4_asym_w4a8`` -- gfx120x asymmetric W4A8 (``quantize_w4a8_int8_weight``, ``dequant_int4_grouped_to_int8``, ``dequantize_w4a8_int8_weight``, ``w4a8_int8_linear``)
- ``kernels.quant.rdna4_awq_w4a16`` -- gfx120x AWQ W4A16 (``dequant_awq_w4a16_weight``, ``gemv_awq_w4a16``)
- ``kernels.quant.rdna4_svdquant_w4a4`` -- gfx120x SVDQuant W4A4 (``quantize_svdquant_w4a4``, ``dequant_svdquant_w4a4_weight``, ``scaled_mm_svdquant_w4a4``, ``svdquant_w4a4_linear``)
- ``kernels.quant.rdna4_int4_codec`` -- gfx120x int4 pack (``pack_int4_row_major``, ``dequant_int4_groupwise_signed``, ``dequant_uint4_groupwise_awq``)

Softmax
-------

- ``kernels.norm.softmax_kernel`` -- Numerically stable softmax

Utilities
---------

- ``kernels.common.kernels_common`` -- Shared constants and helper functions
- ``kernels.common.layout_utils`` -- Layout utility functions
- ``kernels.common.mma.mfma_preshuffle_pipeline`` -- B layout builder and XCD block remapping used by preshuffle GEMM and MoE kernels
- ``kernels.common.gfx120x_arch`` -- ``require_gfx120x`` (process arch starts with ``gfx120``)
- ``kernels.common.gfx120x_swiglu`` -- gfx120x SiLU and SwiGLU (``silu_mul``, ``swiglu_chunk``)
- ``kernels.common.gfx120x_pad`` -- gfx120x pad (``device_pad``)
- ``kernels.common.gfx120x_row_bias`` -- gfx120x row bias (``add_row_bias``, ``add_same``, ``mul_by_scale1``)

.. seealso:: :doc:`../prebuilt_kernels_guide` for usage, and :doc:`../rdna4_functions_guide` for every gfx120x function.
