# RDNA4 function guide

These calls live in the `kernels/` tree of a source checkout. They are not installed by the `flydsl` wheel. A host imports them by the dotted name below.

Every launch calls `kernels.common.gfx120x_arch.require_gfx120x`. That reads `get_rocm_arch()` and raises unless the name starts with `gfx120`. `gfx1250` does not match.

The list is every function another module can import. A name that starts with `_`, and a function defined inside another function, are left out.

When one job has several numeric formats, the wider format is listed first: bf16 and fp16, then FP8, then int8, then int4.

## Matrix multiply

A matrix multiply writes `C = A @ B.T`. Wider numbers come first. bf16 and fp16 share a kernel. FP8 comes next, then a wide activation with a narrow weight, then int8, then int4. A short K is filled with zeros inside the kernel.

### `kernels.gemm.rdna_f16_gemm`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna_f16_gemm.create_wmma_gemm_module` | Compile the f16/bf16 WMMA GEMM. The caller passes the tile and the stream. |

### `kernels.gemm.rdna_fp8_preshuffle_gemm`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna_fp8_preshuffle_gemm.preshuffle_b_fp8` | Preshuffle B[K,N] fp8 for WMMA B operand layout. |
| `kernels.gemm.rdna_fp8_preshuffle_gemm.fp8_quantize_per_token` | Quantize f32 tensor to fp8_e4m3fn with per-token (per-row) scale. |
| `kernels.gemm.rdna_fp8_preshuffle_gemm.fp8_quantize_per_channel` | Quantize f32 tensor to fp8_e4m3fn with per-channel (per-column) scale. |
| `kernels.gemm.rdna_fp8_preshuffle_gemm.compile_fp8_gemm` | Compile fp8 GEMM for RDNA4. |

### `kernels.gemm.rdna4_scaled_mm_fp8`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_scaled_mm_fp8.pick_tile_config` | Size-based tile pick mirroring HIP ``launch_gemm_wmma`` (gemm_wmma.h). |
| `kernels.gemm.rdna4_scaled_mm_fp8.build_scaled_mm_fp8_module` | Compile the shared gfx120x quant GEMM for one specialization. |
| `kernels.gemm.rdna4_scaled_mm_fp8.scaled_mm_fp8` | Product host: contiguify, pick or use ``tile``, then launch. |

### `kernels.gemm.rdna4_scaled_mm_fp8_fused`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_scaled_mm_fp8_fused.build_scaled_mm_fp8_fused_module` | Compile fused act-quant + FP8 WMMA GEMM, optionally with LoRA epilogue. |
| `kernels.gemm.rdna4_scaled_mm_fp8_fused.scaled_mm_fp8_fused` | Host wrapper: fused act-quant + FP8 scaled_mm with optional LoRA. |
| `kernels.gemm.rdna4_scaled_mm_fp8_fused.scaled_mm_fp8_fused_multi` | N-adapter fused path: one quantized base GEMM, then device LoRA residuals. |

### `kernels.gemm.rdna4_mxfp8_block_gemm`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_mxfp8_block_gemm.build_mxfp8_block_gemm_module` | Compile one (out_dtype) MXFP8 block GEMM. A/B are fp8 e4m3; scales are uint8 E8M0. |
| `kernels.gemm.rdna4_mxfp8_block_gemm.mxfp8_block_gemm` | MXFP8 block GEMM host. ``a``/``b`` are ``float8_e4m3fn`` [M,K]/[N,K]; scales are uint8 E8M0. |

### `kernels.gemm.rdna4_w8a16_linear`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_w8a16_linear.pick_tile_config` | Return the W8A16 tile for this M, N, and K. |
| `kernels.gemm.rdna4_w8a16_linear.build_w8a16_linear_module` | Compile W8A16 linear: bf16/fp16 A + 8-bit B cast-in-reg; w_scale in epilogue. |
| `kernels.gemm.rdna4_w8a16_linear.fp8_max_for` | Finite max for e4m3fn (448) or e5m2 (57344); mirrors scaled_mm_fp8 helpers. |
| `kernels.gemm.rdna4_w8a16_linear.w8a16_gemm` | W8A16 linear GEMM: A[M,K] (bf16/fp16) @ B[N,K].T (int8/FP8); w_scale in epi → out[M,N]. |
| `kernels.gemm.rdna4_w8a16_linear.w8a16_linear` | W8A16 linear: bf16/fp16 x @ int8/FP8 weight.T (cast in-reg; w_scale in epi). |

### `kernels.gemm.rdna4_int8_linear`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_int8_linear.pick_tile_config` | Size-based tile pick; prefer IU8 128×128×128 for deep K. |
| `kernels.gemm.rdna4_int8_linear.build_int8_linear_module` | Compile iu8 int8 linear through the shared quant GEMM builder. |
| `kernels.gemm.rdna4_int8_linear.create_wmma_int8_linear_module` | Create a gfx120x int8-linear launcher for one tile configuration. |
| `kernels.gemm.rdna4_int8_linear.int8_linear` | int8 activations × int8 weights through the iu8 WMMA. |

### `kernels.gemm.rdna4_int8_linear_fused`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_int8_linear_fused.build_int8_linear_fused_module` | Compile fused act-quant + int8 WMMA GEMM, optionally with LoRA epilogue. |
| `kernels.gemm.rdna4_int8_linear_fused.int8_linear_fused` | Host wrapper: fused act-quant + int8_linear with optional LoRA. |
| `kernels.gemm.rdna4_int8_linear_fused.int8_linear_fused_multi` | N-adapter fused path: one quantized base GEMM, then device LoRA residuals. |

### `kernels.gemm.rdna4_mxfp4_block_gemm`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_mxfp4_block_gemm.build_mxfp4_block_gemm_module` | Compile one (out_dtype) MXFP4 block GEMM. |
| `kernels.gemm.rdna4_mxfp4_block_gemm.mxfp4_block_gemm` | MXFP4 block GEMM. ``a``/``b`` are packed E2M1 ``[M, K//2]`` / ``[N, K//2]``; scales are uint8 E8M0. |

### `kernels.gemm.rdna4_iu4_gemm`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_iu4_gemm.pick_tile_config` | Size-based tile pick for native iu4 (logical M/N/K). |
| `kernels.gemm.rdna4_iu4_gemm.shapes_ok_for_native_iu4` | True when a raw launcher with ``k_tail=0`` can run (K%16==0, positive dims). |
| `kernels.gemm.rdna4_iu4_gemm.build_iu4_gemm_module` | Compile multi-wave LDS iu4 WMMA GEMM for one (out_dtype, tile, scale mode). |
| `kernels.gemm.rdna4_iu4_gemm.create_wmma_iu4_gemm_module` | Create a gfx120x iu4 GEMM launcher for one tile configuration. |
| `kernels.gemm.rdna4_iu4_gemm.iu4_gemm` | Native iu4 GEMM: packed int4 A/B → ``[M, N]`` (bf16/fp16/fp32 or i32). |

### `kernels.gemm.rdna4_tile`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_tile.TileConfig` | WMMA launch tile. ``threads`` is the block size. ``name`` is the tile label. |
| `kernels.gemm.rdna4_tile.TileConfig.threads` | Threads in the block: warps_m * warps_n * 32. |
| `kernels.gemm.rdna4_tile.TileConfig.name` | Tile label, for logs and the autotune table. |

## MLP

One SwiGLU layer is two matrix multiplies, SiLU, a multiply, and a third matrix multiply. The bf16 and fp16 path can keep the middle values in registers or in local memory. The plain bf16 matmul used by that path lives in the same module. `silu_mul` and `swiglu_chunk` are the activation by itself.

### `kernels.gemm.rdna4_fused_mlp_nmajor`

| Call | What it does |
| --- | --- |
| `kernels.gemm.rdna4_fused_mlp_nmajor.build_wmma_tile_module` | One wave, one 16×16×16 WMMA tile. ``swap_ab`` selects GPUOpen A/B swap. |
| `kernels.gemm.rdna4_fused_mlp_nmajor.build_fused_gemm_tn_module` | Zero-LDS fused ``C1 = (A0 @ B0.T) @ B1.T`` for 16×16×16 panels. |
| `kernels.gemm.rdna4_fused_mlp_nmajor.build_fused_swiglu_mlp_module` | Zero-LDS fused SwiGLU MLP for 16×16 panels (in-register SiLU×mul). |
| `kernels.gemm.rdna4_fused_mlp_nmajor.pick_nmajor_lds_tile` | Pick a multi-wave LDS block tile and padded ``(Mp, Np, Kp)``. |
| `kernels.gemm.rdna4_fused_mlp_nmajor.build_gemm_nmajor_lds_module` | Compile multi-wave LDS-pipelined ``C = A @ B.T`` for fixed padded MNK. |
| `kernels.gemm.rdna4_fused_mlp_nmajor.gemm_bf16_nmajor_lds` | C[M,N] = A[M,K] @ B[N,K].T via multi-wave LDS-pipelined WMMA (gfx120x). |
| `kernels.gemm.rdna4_fused_mlp_nmajor.fused_gemm_tn` | ``C1 = (A0 @ B0.T) @ B1.T`` via zero-LDS fused WMMA (16×16 panels only). |
| `kernels.gemm.rdna4_fused_mlp_nmajor.gemm_bf16_nmajor` | C[M,N] = A[M,K] @ B[N,K].T on the LDS device kernel. |
| `kernels.gemm.rdna4_fused_mlp_nmajor.fused_swiglu_mlp_inreg` | In-register SwiGLU MLP for the 16-cube (host-tiles M/N). |
| `kernels.gemm.rdna4_fused_mlp_nmajor.build_fused_swiglu_mlp_lds_module` | In-kernel SwiGLU: K and FFN loops, 16-wide mid kept in LDS. |
| `kernels.gemm.rdna4_fused_mlp_nmajor.fused_swiglu_mlp_lds` | SwiGLU MLP with the mid in LDS inside one kernel per 16×16 output tile. |
| `kernels.gemm.rdna4_fused_mlp_nmajor.fused_swiglu_mlp_nmajor` | SwiGLU MLP host. |

### `kernels.common.gfx120x_swiglu`

| Call | What it does |
| --- | --- |
| `kernels.common.gfx120x_swiglu.build_silu_mul_module` | Flat elementwise silu(gate)*up. |
| `kernels.common.gfx120x_swiglu.build_swiglu_chunk_module` | Contiguous last-dim chunk-2 SwiGLU (interleaved gate/up loads). |
| `kernels.common.gfx120x_swiglu.silu_mul` | Product silu(gate)*up. A length that is not a vector multiple stays in the kernel. |
| `kernels.common.gfx120x_swiglu.swiglu_chunk` | Product chunk-2 SwiGLU: ``x[..., 2*H]`` → ``silu(gate)*up`` with shape ``[..., H]``. |

## Attention

Flash attention on gfx120x. bf16 and fp16 are one kernel. fp8 and int8 are the other kernel, and the output of that kernel is bf16. The host picks dense, packed-varlen, or paged KV. Causal masking is bottom-right: key `j` is kept when `j <= i + (Sk - Sq)`. `kernels.attention.gfx120x_online_softmax` defines no functions of its own. It re-exports `kill_score_columns`, `apply_sliding_window`, `add_score_bias`, `add_alibi_scores`, `attention_lse`, `attention_inv_l`, `clear_o_if_empty`, `online_softmax_tile`, and `fold_attention_sink` from the host module.

### `kernels.attention.flash_attn_gfx120x_host`

| Call | What it does |
| --- | --- |
| `kernels.attention.flash_attn_gfx120x_host.kill_score_columns` | Replace a score with -inf when any ``(op, limit)`` pred matches. |
| `kernels.attention.flash_attn_gfx120x_host.apply_sliding_window` | Drop scores outside the window. Returns the masked scores and row keep-mass. |
| `kernels.attention.flash_attn_gfx120x_host.add_score_bias` | Add one bias row onto the 16 scores. ``scale`` is None for a bf16 logit. |
| `kernels.attention.flash_attn_gfx120x_host.add_alibi_scores` | Add ``-slope * \|q_row + sk - sq - key\|`` onto the 16 unscaled scores. |
| `kernels.attention.flash_attn_gfx120x_host.attention_lse` | ``m + log(l)``. An empty row with no sink stores -inf. |
| `kernels.attention.flash_attn_gfx120x_host.attention_inv_l` | ``1/l`` when the row has mass, else 0. Callers apply their own output scale. |
| `kernels.attention.flash_attn_gfx120x_host.clear_o_if_empty` | Zero O when the row has no mass so a later ``0 * NaN`` cannot leak. |
| `kernels.attention.flash_attn_gfx120x_host.online_softmax_tile` | One KV tile of the running max, sum, and output rescale. |
| `kernels.attention.flash_attn_gfx120x_host.fold_attention_sink` | Rescale O and l when a sink logit wins the final max. Returns O, l, m. |
| `kernels.attention.flash_attn_gfx120x_host.mask_is_noop` | True when mask can be ignored without reading values. |
| `kernels.attention.flash_attn_gfx120x_host.normalize_attn_mask` | Normalize SDPA-style mask to dense fp32 additive bias [Sq, Skv]. |
| `kernels.attention.flash_attn_gfx120x_host.fold_alibi_to_bias` | Fold ALiBi slopes into additive bias (bottom-right aligned) on device. |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_func` | Run FlyDSL Flash Attention on RDNA4 (gfx120x family). |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_varlen_func` | Packed-varlen FA: Q/K/V/O as [total, H, D] with cu_seqlens int32 [B+1]. |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_paged_func` | Paged FA. linear 4D, linear3d page-1 (same bytes), or vectorized 5D. |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_varlen_paged_func` | Packed Q with paged K/V. cu_seqlens_q selects Q rows; block_table/seqlen_k select KV. |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_fp8_func` | fp8 e4m3 or e5m2 flash attention. Output is bf16. |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_int8_func` | int8 flash attention. Same launch as fp8, iu8 WMMA. Output is bf16. |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_fp8_varlen_func` | Packed fp8 flash attention. Q, K, and V are `[total, H, D]`. Output is bf16. |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_quant_paged_func` | Quant paged KV. The cache and block table are passed through. No gather copy. |
| `kernels.attention.flash_attn_gfx120x_host.flydsl_flash_attn_int8_varlen_func` | Packed int8 flash attention. Same launch as the fp8 varlen path. Output is bf16. |

### `kernels.attention.flash_attn_gfx120x`

| Call | What it does |
| --- | --- |
| `kernels.attention.flash_attn_gfx120x.build_flash_attn_func_module_primary` | Build the gfx120x Flash Attention kernel. |

### `kernels.attention.flash_attn_fp8_gfx120x`

| Call | What it does |
| --- | --- |
| `kernels.attention.flash_attn_fp8_gfx120x.build_flash_attn_func_fp8_module_primary` | Build the gfx120x FP8 Flash Attention kernel (QKV=E4M3FN, O=bf16). |

### `kernels.attention.flash_attn_int8_gfx120x`

| Call | What it does |
| --- | --- |
| `kernels.attention.flash_attn_int8_gfx120x.build_flash_attn_func_int8_module_primary` | Build iu8 FlashAttention through the shared fp8/int8 quant kernel. |

### `kernels.attention.flash_attn_gfx120x_splitk`

| Call | What it does |
| --- | --- |
| `kernels.attention.flash_attn_gfx120x_splitk.build_splitk_combine_module` | Build the gfx120x split-K combine (block=64). It writes the merged output and LSE. |

### `kernels.attention.flash_attn_gfx120x_ext`

| Call | What it does |
| --- | --- |
| `kernels.attention.flash_attn_gfx120x_ext.reject_varlen_with_paged` | Packed Q + paged KV is in-kernel. Kept as a no-op for callers. |
| `kernels.attention.flash_attn_gfx120x_ext.reject_gqa` | Allow GQA/MQA when ``Hq % Hkv == 0`` (including MHA, ``Hq == Hkv``). |
| `kernels.attention.flash_attn_gfx120x_ext.reject_quant_extras` | fp8/int8 bias, ALiBi, sink, and LSE run in the quant kernel. This check is a no-op. |
| `kernels.attention.flash_attn_gfx120x_ext.reject_paged_layout` | Raise unless the paged KV layout is linear, linear3d, or vectorized. |
| `kernels.attention.flash_attn_gfx120x_ext.lengths_uniform` | True when every batch length is equal (optionally to ``max_len``). |
| `kernels.attention.flash_attn_gfx120x_ext.cu_seqlens_from_seqlens` | Build int32 ``cu_seqlens`` ``[B+1]`` from per-batch lengths (device cumsum). |
| `kernels.attention.flash_attn_gfx120x_ext.reject_sink_with_alibi` | Sink and ALiBi both apply inside one gfx120x launch (ALiBi via bias, sink in epilogue). |
| `kernels.attention.flash_attn_gfx120x_ext.reject_splitk_extras` | Split-K partials include bias/ALiBi; combine writes LSE. No longer rejected. |
| `kernels.attention.flash_attn_gfx120x_ext.commit_caller_out` | Return ``produced``, or copy it into caller-owned ``out`` and return that. |

### `kernels.attention.gfx120x_alibi_bias`

| Call | What it does |
| --- | --- |
| `kernels.attention.gfx120x_alibi_bias.build_alibi_bias_module` | Internal helper. |
| `kernels.attention.gfx120x_alibi_bias.fill_alibi_bias` | Fill ALiBi bias on device. slopes is contiguous fp32 [H] (or [1]). |

### `kernels.attention.gfx120x_attn_mask`

| Call | What it does |
| --- | --- |
| `kernels.attention.gfx120x_attn_mask.build_bool_mask_to_bias_module` | Internal helper. |
| `kernels.attention.gfx120x_attn_mask.bool_mask_to_additive` | Convert a bool mask to fp32 additive bias (True→0, False→-inf) on device. |

## Normalization and rotary position

RMSNorm, rotary position, and AdaLN. bf16, fp16, and fp32 are selected by the builder argument, not by a second module. Interleaved pairs and split-half pairs are separate builders. A fused Q and K launch is a separate builder again.

### `kernels.norm.adaln_gfx120x`

| Call | What it does |
| --- | --- |
| `kernels.norm.adaln_gfx120x.build_adaln_module` | Specialize fused AdaLN kernel on (N, dtype, subtract_mean). |

### `kernels.norm.rms_rope_gfx120x`

| Call | What it does |
| --- | --- |
| `kernels.norm.rms_rope_gfx120x.build_rms_rope_module` | Build the RMSNorm-then-RoPE kernel for one tensor. |
| `kernels.norm.rms_rope_gfx120x.build_rms_rope_split_module` | Build RMSNorm-then-RoPE with split-half rotary pairs. |
| `kernels.norm.rms_rope_gfx120x.build_rms_rope_qk_fused_module` | Build RMSNorm-then-RoPE for Q and K in one launch. |
| `kernels.norm.rms_rope_gfx120x.build_rms_rope_split_qk_fused_module` | Q then K in one block — reuse LDS norm+red, shared freqs. |

### `kernels.norm.rope_gfx120x`

| Call | What it does |
| --- | --- |
| `kernels.norm.rope_gfx120x.resolve_rope_block` | Use ``block`` when set. Otherwise 256 threads. |
| `kernels.norm.rope_gfx120x.build_rope_module` | Specialize the RoPE kernel. The default block is 256 threads. |
| `kernels.norm.rope_gfx120x.build_rope_split_module` | Specialize the RoPE kernel. The default block is 256 threads. |
| `kernels.norm.rope_gfx120x.build_rope_qk_fused_module` | Specialize the RoPE kernel. The default block is 256 threads. |
| `kernels.norm.rope_gfx120x.build_rope_split_half_qk_fused_module` | Specialize the RoPE kernel. The default block is 256 threads. |

## Quantization

Turn a float tensor into a smaller code, or turn a packed code back into a value you can multiply. FP8 is first, then MXFP8, then int8, then MXFP4 and packed int4. ConvRot is a Hadamard rotation around that quantize step. The int4 codec is ordinary Python. It does not launch a GPU kernel.

### `kernels.quant.rdna4_fp8_quant`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_fp8_quant.build_fp8_quant_module` | Build the per-tensor FP8 quant kernel. |
| `kernels.quant.rdna4_fp8_quant.build_fp8_dequant_module` | Build the per-tensor FP8 dequant kernel. |
| `kernels.quant.rdna4_fp8_quant.fp8_quant_direct` | Direct JIT entry. ``tuning_schema`` partitions the autotune cache. |
| `kernels.quant.rdna4_fp8_quant.fp8_dequant_direct` | Direct JIT entry. ``tuning_schema`` partitions the autotune cache. |
| `kernels.quant.rdna4_fp8_quant.dequantize_fp8` | Per-tensor FP8 dequant. ``q * scale`` using the gfx120x dequant kernel. |

### `kernels.quant.rdna4_stoch_fp8`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_stoch_fp8.build_stoch_fp8_module` | Build @flyc.jit launcher. path in {"bitcast","select"}. |
| `kernels.quant.rdna4_stoch_fp8.stoch_fp8_direct` | Direct JIT entry. ``tuning_schema`` partitions the autotune cache. |

### `kernels.quant.rdna4_mxfp8_e8m0`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_mxfp8_e8m0.build_mxfp8_quant_module` | Internal helper. |
| `kernels.quant.rdna4_mxfp8_e8m0.build_mxfp8_dequant_module` | One thread per 32-block. ``clamp_tiny`` matches ``quant_dequant_mxfp8``. |
| `kernels.quant.rdna4_mxfp8_e8m0.build_mxfp8_quant_tail_module` | Same quant as the aligned builder, for a K that is not a multiple of 32. |
| `kernels.quant.rdna4_mxfp8_e8m0.build_mxfp8_dequant_tail_module` | Decode a short last MX group without reading past the caller's K. |
| `kernels.quant.rdna4_mxfp8_e8m0.quantize_mxfp8_device` | Quantize ``[..., K]`` to FP8 plus one E8M0 scale per started group of 32. |
| `kernels.quant.rdna4_mxfp8_e8m0.dequantize_mxfp8_device` | Decode row-major MXFP8 and per-32 E8M0 scales to fp32. |
| `kernels.quant.rdna4_mxfp8_e8m0.quant_dequant_mxfp8_device` | Per-1x32 MXFP8 round trip. The dequant scale is clamped the way the torch helper clamps it. |

### `kernels.quant.rdna4_quantize_int8_tensorwise`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_quantize_int8_tensorwise.build_quantize_int8_tensorwise_module` | Three launches: block partial amax, tree-reduce amax, quantize. |
| `kernels.quant.rdna4_quantize_int8_tensorwise.quantize_int8_tensorwise_partial_direct` | Internal helper. |
| `kernels.quant.rdna4_quantize_int8_tensorwise.quantize_int8_tensorwise_reduce_direct` | Internal helper. |
| `kernels.quant.rdna4_quantize_int8_tensorwise.quantize_int8_tensorwise_quant_direct` | Internal helper. |
| `kernels.quant.rdna4_quantize_int8_tensorwise.build_quantize_int8_given_scale_module` | Quantize with a scalar scale already chosen by the caller. |
| `kernels.quant.rdna4_quantize_int8_tensorwise.quantize_int8_given_scale_direct` | Internal helper. |
| `kernels.quant.rdna4_quantize_int8_tensorwise.quantize_int8_tensorwise` | Tensorwise absmax int8 quant → ``(q, scale)``. |
| `kernels.quant.rdna4_quantize_int8_tensorwise.dequantize_int8_tensorwise` | Inverse of ``quantize_int8_tensorwise``: ``q * scale``. |

### `kernels.quant.rdna4_quantize_int8_rowwise`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_quantize_int8_rowwise.build_quantize_int8_rowwise_module` | Build the rowwise absmax int8 quant kernel. |
| `kernels.quant.rdna4_quantize_int8_rowwise.quantize_int8_rowwise_direct` | Direct JIT entry. ``tuning_schema`` partitions the autotune cache. |
| `kernels.quant.rdna4_quantize_int8_rowwise.quantize_int8_rowwise` | Rowwise absmax int8 quant → (q, scale). |
| `kernels.quant.rdna4_quantize_int8_rowwise.dequantize_int8_rowwise` | Inverse of ``quantize_int8_rowwise``: ``q * scale`` per row. |

### `kernels.quant.rdna4_int8_convrot`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_int8_convrot.build_int8_convrot_quant_module` | Compile one-row-per-block ConvRot rotate + rowwise INT8 quant kernel. |
| `kernels.quant.rdna4_int8_convrot.build_int8_convrot_dequant_module` | One row per block: int8 * row scale, then the same FWHT (H is its own inverse). |
| `kernels.quant.rdna4_int8_convrot.build_convrot_fwht_module` | In-place-shaped ``W @ H`` for one ConvRot group size. Device FWHT, not a matmul. |
| `kernels.quant.rdna4_int8_convrot.int8_convrot_quant_direct` | Internal helper. |
| `kernels.quant.rdna4_int8_convrot.int8_convrot_dequant_direct` | Internal helper. |
| `kernels.quant.rdna4_int8_convrot.convrot_fwht_direct` | Internal helper. |
| `kernels.quant.rdna4_int8_convrot.convrot_fwht` | Device ``W @ H`` per ConvRot group. Replaces a host Hadamard matmul. |
| `kernels.quant.rdna4_int8_convrot.quantize_int8_convrot_weight` | Offline ConvRot weight rotation + rowwise INT8 quantize (HIP-compatible API). |
| `kernels.quant.rdna4_int8_convrot.quantize_and_rotate_rowwise` | Online activation ConvRot rotate + rowwise INT8 (same kernel as weights). |
| `kernels.quant.rdna4_int8_convrot.dequantize_int8_convrot_weight` | Dequant INT8 ConvRot weights and un-rotate on the gfx120x FWHT kernel. |
| `kernels.quant.rdna4_int8_convrot.dequantize_int8_convrot_weight_dtype` | Kitchen entry: ``output_dtype_code`` matches ``DTYPE_CODE_TO_DTYPE``. |
| `kernels.quant.rdna4_int8_convrot.int8_linear_convrot` | ConvRot INT8 linear: online act rotate+quant + iu8 WMMA GEMM. |

### `kernels.quant.rdna4_mxfp4_e2m1`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_mxfp4_e2m1.build_mxfp4_quant_module` | Internal helper. |
| `kernels.quant.rdna4_mxfp4_e2m1.build_mxfp4_dequant_module` | Internal helper. |
| `kernels.quant.rdna4_mxfp4_e2m1.quantize_mxfp4_device` | Quantize ``[..., K]`` to packed MXFP4 plus per-32 E8M0 scales. |
| `kernels.quant.rdna4_mxfp4_e2m1.dequantize_mxfp4_device` | Decode packed MXFP4 and per-32 E8M0 scales to fp32. |

### `kernels.quant.rdna4_convrot_w4a4`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_convrot_w4a4.build_convrot_w4a4_quant_module` | Compile one-row-per-block ConvRot rotate + rowwise signed INT4 pack. |
| `kernels.quant.rdna4_convrot_w4a4.convrot_w4a4_quant_direct` | Direct JIT entry. ``tuning_schema`` partitions the autotune cache. |
| `kernels.quant.rdna4_convrot_w4a4.quantize_convrot_w4a4_weight` | Offline ConvRot + signed INT4 pack (HIP-compatible API). |
| `kernels.quant.rdna4_convrot_w4a4.build_expand_signed_i4_module` | Packed signed nibbles → int8 codes, one byte per thread. |
| `kernels.quant.rdna4_convrot_w4a4.expand_signed_i4_direct` | Internal helper. |
| `kernels.quant.rdna4_convrot_w4a4.expand_signed_i4` | Device signed-nibble unpack. ``packed[..., K//2]`` → ``[..., K]`` int8. |
| `kernels.quant.rdna4_convrot_w4a4.build_i4_row_scale_module` | Signed nibble × per-row scale → f32, one packed byte per thread. |
| `kernels.quant.rdna4_convrot_w4a4.i4_row_scale_direct` | Internal helper. |
| `kernels.quant.rdna4_convrot_w4a4.dequantize_convrot_w4a4_weight` | Dequant packed W4 and un-rotate. Nibble expand and FWHT are both device kernels. |
| `kernels.quant.rdna4_convrot_w4a4.convrot_w4a4_linear` | ConvRot W4A4 linear: bf16/fp16 ``x`` × packed W4 weights → ``[M, N]``. |

### `kernels.quant.rdna4_asym_w4a8`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_asym_w4a8.build_w4a8_dequant_int4_to_int8_module` | Decode packed unsigned INT4 (+ optional codebook) × s_rel → INT8 grid. |
| `kernels.quant.rdna4_asym_w4a8.rotate_convrot_weight` | ``W @ H`` per ConvRot group, on the gfx120x FWHT kernel. |
| `kernels.quant.rdna4_asym_w4a8.validate_w4a8_weight_shape` | Raise unless the packed W4A8 weight matches N, K, and group size. |
| `kernels.quant.rdna4_asym_w4a8.build_w4a8_pack_module` | Device W4A8 pack. |
| `kernels.quant.rdna4_asym_w4a8.quantize_w4a8_int8_weight` | Rotate on device, then pack W4A8 with the device quant kernel. |
| `kernels.quant.rdna4_asym_w4a8.dequant_int4_grouped_to_int8` | FlyDSL decode of packed W4A8 → INT8 GEMM grid. |
| `kernels.quant.rdna4_asym_w4a8.build_w4a8_reconstruct_module` | Packed int4 → original-basis weight. Direct JIT, autotuned BLOCK. |
| `kernels.quant.rdna4_asym_w4a8.w4a8_reconstruct_direct` | Specialize reconstruct through JIT Constexpr inputs. |
| `kernels.quant.rdna4_asym_w4a8.dequantize_w4a8_int8_weight` | Original-basis W4A8 weight. Direct JIT, autotuned BLOCK. |
| `kernels.quant.rdna4_asym_w4a8.w4a8_int8_linear` | ``x @ W.T + bias`` via INT4→INT8 decode + ConvRot INT8 linear. |

### `kernels.quant.rdna4_awq_w4a16`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_awq_w4a16.pick_awq_gemv_tiles` | Return ``(max_m, n_tile, block_threads)`` for fused AWQ GEMV (measured). |
| `kernels.quant.rdna4_awq_w4a16.unpack_uint4_row_major` | (..., K//2) int8 → (..., K) int8 in [0, 15]. |
| `kernels.quant.rdna4_awq_w4a16.build_awq_dequant_w4a16_module` | Unpack AWQ uint4 + apply group scales/zeros → bf16/fp16 weight row. |
| `kernels.quant.rdna4_awq_w4a16.awq_dequant_direct` | Internal helper. |
| `kernels.quant.rdna4_awq_w4a16.dequant_awq_w4a16_weight` | FlyDSL AWQ W4A16 weight dequant → bf16/fp16 ``(N, K)``. |
| `kernels.quant.rdna4_awq_w4a16.build_awq_gemv_module` | Fused AWQ W4A16 GEMV — i32×8 decode, optional LDS-X + N-tile. |
| `kernels.quant.rdna4_awq_w4a16.awq_gemv_direct` | Internal helper. |
| `kernels.quant.rdna4_awq_w4a16.gemv_awq_w4a16` | Host-API ``gemv_awq_w4a16``: fused FlyDSL GEMV (packed W, in-reg dequant). |

### `kernels.quant.rdna4_svdquant_w4a4`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_svdquant_w4a4.pick_svdquant_n_tile` | Measured gfx1201 N-tile for fused SVDQuant (2026-09-30). |
| `kernels.quant.rdna4_svdquant_w4a4.build_svdquant_dequant_w4_module` | Unpack signed INT4 + apply group scales → bf16/fp16 weight row. |
| `kernels.quant.rdna4_svdquant_w4a4.svdquant_dequant_direct` | Internal helper. |
| `kernels.quant.rdna4_svdquant_w4a4.dequant_svdquant_w4a4_weight` | FlyDSL SVDQuant W4 weight dequant → bf16/fp16 ``(N, K)``. |
| `kernels.quant.rdna4_svdquant_w4a4.build_svdquant_act_quant_module` | One row per block: smooth, group absmax, nibble pack, and LoRA-down dot. |
| `kernels.quant.rdna4_svdquant_w4a4.svdquant_act_quant_direct` | Block is the group (64). No thread-block choice to autotune. |
| `kernels.quant.rdna4_svdquant_w4a4.quantize_svdquant_w4a4` | Device INT4 act quant: smooth, group scale, nibble pack, LoRA-down dot. |
| `kernels.quant.rdna4_svdquant_w4a4.build_svdquant_scaled_mm_fused_module` | Fused SVDQuant W4A4 scaled mm: packed A×W, in-reg unpack+scale, N-tile. |
| `kernels.quant.rdna4_svdquant_w4a4.svdquant_fused_direct` | Internal helper. |
| `kernels.quant.rdna4_svdquant_w4a4.scaled_mm_svdquant_w4a4` | SVDQuant W4A4 scaled mm (host API). |
| `kernels.quant.rdna4_svdquant_w4a4.svdquant_w4a4_linear` | Layout linear: quantize + scaled_mm (unpad M). |

### `kernels.quant.rdna4_int4_codec`

| Call | What it does |
| --- | --- |
| `kernels.quant.rdna4_int4_codec.pack_int4_row_major` | Pack ``(..., K)`` int4 values into ``(..., K//2)`` int8 (low = even). |
| `kernels.quant.rdna4_int4_codec.unpack_int4_row_major` | Inverse of :func:`pack_int4_row_major` with **signed** nibble ``[-8, 7]``. |
| `kernels.quant.rdna4_int4_codec.unpack_uint4_row_major` | Inverse of :func:`pack_int4_row_major` with **unsigned** nibble ``[0, 15]``. |
| `kernels.quant.rdna4_int4_codec.dequant_int4_groupwise_signed` | Host dequant: signed packed INT4 × per-group scales → compute dtype. |
| `kernels.quant.rdna4_int4_codec.dequant_uint4_groupwise_awq` | Host AWQ-style dequant: ``(u4 - 8) * scale + zero`` (reference AWQ wire). |

## Elementwise

Pad, add a row bias, add two tensors, or multiply by one scale. The arithmetic is in fp32. The stored dtype can be bf16, fp16, or fp32.

### `kernels.common.gfx120x_pad`

| Call | What it does |
| --- | --- |
| `kernels.common.gfx120x_pad.build_pad_module` | Internal helper. |
| `kernels.common.gfx120x_pad.ensure_contiguous` | Return ``x`` contiguous; if a copy is needed, run it on ``stream``. |
| `kernels.common.gfx120x_pad.device_pad` | Pad ``x`` on device. ``mode`` matches ``F.pad`` (``zeros`` is constant). |
| `kernels.common.gfx120x_pad.ceil_to_multiple` | Smallest multiple of ``multiple`` that is >= ``n`` (``n`` must be > 0). |

### `kernels.common.gfx120x_row_bias`

| Call | What it does |
| --- | --- |
| `kernels.common.gfx120x_row_bias.build_row_bias_module` | Internal helper. |
| `kernels.common.gfx120x_row_bias.add_row_bias` | ``out[..., N] + bias[N]``. The add is a gfx120x kernel, accumulated in f32. |
| `kernels.common.gfx120x_row_bias.build_add_same_module` | Internal helper. |
| `kernels.common.gfx120x_row_bias.add_same` | Elementwise ``a + b`` on gfx120x. Both tensors must match shape and dtype. |
| `kernels.common.gfx120x_row_bias.build_mul_by_scale1_module` | ``out[i] = in[i] * scale[0]`` for bf16/fp16/fp32 (scale is 1xf32). |
| `kernels.common.gfx120x_row_bias.mul_by_scale1` | Elementwise ``x * scale[0]`` on gfx120x. ``scale`` is a 1-element fp32 CUDA tensor. |

## Checks and buffer helpers

The arch check and the buffer-copy helpers used inside the kernels above. These do not compute a layer by themselves.

### `kernels.common.gfx120x_arch`

| Call | What it does |
| --- | --- |
| `kernels.common.gfx120x_arch.require_gfx120x` | Raise unless the process arch starts with ``gfx120``. |

### `kernels.common.gfx120x_buf_helpers`

| Call | What it does |
| --- | --- |
| `kernels.common.gfx120x_buf_helpers.kernel_signature` | Render specialization parameters into a legal kernel-name suffix. |
| `kernels.common.gfx120x_buf_helpers.buf_base_i64` | Return the byte address of a pointer or tensor-like kernel argument. |
| `kernels.common.gfx120x_buf_helpers.ptr_buf_tensor` | Create a buffer-resource view over a raw device pointer or tensor-like arg. |
| `kernels.common.gfx120x_buf_helpers.buf_copy_load` | Load one scalar/vector unit through an AMD buffer-copy atom. |
| `kernels.common.gfx120x_buf_helpers.buf_copy_store` | Store one scalar/vector unit through an AMD buffer-copy atom. |

201 calls are listed.
