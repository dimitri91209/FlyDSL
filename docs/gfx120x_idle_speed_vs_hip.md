# gfx120x idle Speed vs HIP

**Hardware:** AMD Radeon AI PRO R9700 (GFX120X family; the device string often
reads `gfx1201`)  
**Date:** 2026-09-30 (America/Chicago)  
**Credit:** dimitri91209 + Grokbot

This document is the measured speed reference for the gfx120x kernels in this
tree. It answers three questions in plain English:

1. **Which call should I make?** (recommended defaults)
2. **How were the size cutoffs chosen?** (breakpoint math)
3. **What is actually fast or slow vs HIP?** (tables, one family at a time)

HIP baseline means the matching ROCm / Comfy HIP kernel timed in a separate
process. We do not mix families in one verdict paragraph.

---

## How timing works

Idle Speed vs HIP follows FlyDSL Device timing (`do_bench`) in
[`docs/autotune_guide.md`](autotune_guide.md):

1. One process measures one `(kernel, case, backend)` on one tensor set.
   HIP and FlyDSL never share a process. Ratios are computed after both
   results exist.
2. Warmup, then up to five batches. Each batch is preceded by a GPU backlog
   (`torch.cuda._sleep`), then a CUDA-event window of N launches. We keep the
   **median** of the batch averages (reported in microseconds).
3. GEMM suite defaults: **warm=10, rep=50**.
4. Speedup = `HIP_µs / Fly_µs`. Greater than 1 means FlyDSL is faster.
5. Verdicts: **WIN** ≥ 1.05 · **PARITY** 0.95–1.05 · **LOSE** < 0.95.
   PARITY and LOSE still ship unless a maintainer asks to drop them.

Shape labels such as `wanish` or `flux77` only explain why a geometry was
chosen. They are not product wiring.

---

## Recommended calls (read this first)

**EXPERIMENTAL:** the size rules below are the **default** way to pick a
pipeline on gfx120x. They come from idle medians on one R9700. They are **not**
a full FlyDSL autotune search across every GPU. You can force a specific kernel
when you already know what you want. A future per-device autotune cache can
replace the static numbers; until then these cutoffs are the shipped default.

| What you want | Call this | Why |
|---|---|---|
| Int8-weight linear (usual case) | `int8_linear_auto` / `int8_linear_dispatched` | Picks W8A16 or iu8 from the size rule so large shapes do not stay on the slow path |
| Always W8A16 (bf16 acts, 8-bit weights) | `w8a16_gemm` / `rdna4_w8a16_linear` | You already know the shape is small/mid |
| Always iu8 int8×int8 | `rdna4_int8_linear` | You already know the shape is large, or you need HIP-bit int8 acts |
| FP8 tensorwise scaled_mm | `scaled_mm_fp8_auto` | Picks a measured tile; plain `scaled_mm_fp8` on huge shapes can sit at parity |
| ConvRot W4A4 linear | `convrot_w4a4_linear(...)` | **Default is native int4** (device act ConvRot-i4 + iu4 WMMA). Pass `linear_dtype="int8"` only if you need the unpack→iu8 path |
| Bare packed int4 GEMM | `iu4_gemm(...)` | Native iu4 by default when `K % 16 == 0`; otherwise unpack→iu8 |
| AWQ W4A16 | `gemv_awq_w4a16` | Separate family; currently LOSE vs HIP — still ships |
| SVDQuant W4A4 | `scaled_mm_svdquant_w4a4` | Separate family; WIN mid/large, LOSE at M=1 |

**How “better kernels” turn on:** call the **auto / dispatched** entry, or for
ConvRot leave the default `linear_dtype="int4"`. You do not flip a global
environment flag. To pin an older path, pass an explicit force
(`force_kernel="iu8"|"w8a16"`, or `linear_dtype="int8"`, or
`prefer_native=False`).

---

## Breakpoint math (how cutoffs are calculated)

We do not invent a magic constant. For each competing pipeline we measure
idle speedup S = t_HIP / t_Fly on a fixed shape grid. The default rule is the
simplest host predicate that keeps every measured point on a WIN or PARITY
pipeline whenever one exists.

### Int8: W8A16 vs iu8

Two different pipelines:

| Name | Module | What runs |
|---|---|---|
| **W8A16 linear** | `rdna4_w8a16_linear` | Activations stay bf16/fp16. Int8 or FP8 (e4m3fn / e5m2) weights cast to bf16, then float WMMA. Does **not** need the iu8 atom. |
| **iu8 int8 linear** | `rdna4_int8_linear` | Activations are quantized to int8. Int8xint8 WMMA uses the gfx120x **iu8** atom, then a scale epilogue. |

Measured anchors (same methodology as the full table below):

| Shape | W8A16 alone S | iu8 alone S | What a good default must pick |
|---|---:|---:|---|
| tiny `[32, 128, 64]` | **1.822** WIN | ~1.12 WIN | W8A16 (faster) |
| mid `[256, 512, 512]` | **1.566** WIN | ~1.22 WIN | W8A16 (faster) |
| large `[1024, 4096, 4096]` | **0.574** LOSE | **1.503** WIN | iu8 (W8A16 loses) |

Define the W8A16 region as shapes where W8A16 is competitive, and the iu8
region as everything else. A two-threshold envelope fits the grid without a
per-shape table:

```
pick(M, K):
  if M <= M_star:           choose W8A16
  else if M * K <= MK_star: choose W8A16
  else:                     choose iu8
```

Shipped defaults: M_star = 128, MK_star = 500_000
(`DEFAULT_M_W8A16_MAX`, `DEFAULT_MK_W8A16_MAX` in
`kernels/gemm/rdna4_int8_linear_dispatch.py`).

**Why these numbers:** mid `[256,512,512]` has M=256 > 128 but
M*K = 131_072 <= 500_000, so it stays on W8A16 (WIN). Large
`[1024,4096,4096]` has M*K ~ 4.2e9, so it flips to iu8 (WIN). Tiny shapes
hit M <= 128 and stay on W8A16.

**Override:** `force_kernel="w8a16"` or `"iu8"`.
**Autotune later:** a per-device `do_bench` search (see
[`docs/autotune_guide.md`](autotune_guide.md)) can replace M_star / MK_star
with cached cutoffs; the predicate shape stays the same.

Modules: `rdna4_int8_linear_dispatch` (rule + helpers),
`rdna4_int8_linear_auto` (default entry that applies the rule). Python aliases
`gate` / `_gated` mean this same size dispatcher.

### FP8: tile picker

`scaled_mm_fp8_auto` does not change the math. It picks a **tile** from
measured breakpoints inside `pick_tile_config` (skinny BK on tiny shapes,
deeper K on fat shapes). Same shape, plain vs auto:

| Call | Shape | × vs HIP |
|---|---|---:|
| Plain `scaled_mm_fp8` | wanish `[1024, 5120, 5120]` | **1.003** PARITY |
| Auto tile pick | same | **3.970** WIN |

Default call for FP8 scaled_mm on gfx120x: **`scaled_mm_fp8_auto`**.

### Native iu4: when the WMMA path is legal

Native iu4 needs K % 16 == 0 and positive M,N,K
(`shapes_ok_for_native_iu4`). That is a hardware/ABI constraint (16-wide iu4
WMMA), not a speed crossover. If the shape fails, `iu4_gemm` / ConvRot fall
back to unpack→iu8 instead of crashing.

## Full idle suite (2026-09-30) — core ops

| Op | Case | Shape | HIP µs | Fly µs | × | Verdict |
|---|---|---|---:|---:|---:|---|
| int8_dispatch | large | `[1024, 4096, 4096]` | 310.882 | 207.041 | 1.502 | WIN |
| int8_dispatch | tiny | `[32, 128, 64]` | 11.468 | 6.356 | 1.804 | WIN |
| adaln | flux77_ln | `[77, 3072]` | 14.080 | 4.436 | 3.174 | WIN |
| convrot | simple64 | `[64, 64, 64, 64]` | 12.044 | 10.624 | 1.134 | WIN |
| int8_linear_auto | large | `[1024, 4096, 4096]` | 336.278 | 213.665 | 1.574 | WIN |
| int8_linear_auto | mid | `[256, 512, 512]` | 29.392 | 18.624 | 1.578 | WIN |
| int8_linear_auto | tiny | `[32, 128, 64]` | 11.736 | 6.388 | 1.837 | WIN |
| int8_linear (iu8) | large | `[1024, 4096, 4096]` | 343.266 | 228.405 | 1.503 | WIN |
| int8_linear (iu8) | mid | `[128, 256, 512]` | 21.908 | 17.992 | 1.218 | WIN |
| int8_linear (iu8) | tiny | `[64, 64, 64]` | 11.548 | 10.356 | 1.115 | WIN |
| int8_linear (iu8) | wanish | `[1024, 5120, 5120]` | 409.823 | 422.924 | 0.969 | PARITY |
| int8_rowwise | large | `[1024, 4096]` | 34.236 | 11.784 | 2.905 | WIN |
| int8_rowwise | tiny | `[64, 256]` | 3.840 | 3.424 | 1.121 | WIN |
| rms_qk | flux77 | `[1, 77, 24, 128]` | 10.352 | 6.636 | 1.560 | WIN |
| rms_rope1 | small | `[1, 8, 4, 64]` | 4.516 | 3.744 | 1.206 | WIN |
| rope_sh_qk | flux77 | `[1, 77, 24, 128]` | 8.608 | 5.332 | 1.614 | WIN |
| scaled_mm_fp8_auto | mid | `[128, 512, 512]` | 16.708 | 6.836 | 2.444 | WIN |
| scaled_mm_fp8_auto | tiny | `[32, 128, 64]` | 9.840 | 4.924 | 1.998 | WIN |
| scaled_mm_fp8_auto | wanish | `[1024, 5120, 5120]` | 1137.267 | 286.498 | 3.970 | WIN |
| scaled_mm_fp8 | mid_fat | `[128, 256, 512]` | 15.316 | 11.636 | 1.316 | WIN |
| scaled_mm_fp8 | mid_suite | `[128, 512, 512]` | 16.908 | 12.680 | 1.333 | WIN |
| scaled_mm_fp8 | tiny_k128 | `[32, 128, 128]` | 10.104 | 9.332 | 1.083 | WIN |
| scaled_mm_fp8 | tiny_k64 | `[32, 128, 64]` | 10.024 | 9.232 | 1.086 | WIN |
| scaled_mm_fp8 | wanish | `[1024, 5120, 5120]` | 1143.343 | 1139.479 | 1.003 | PARITY |
| stoch | tiny | `[64, 128]` | 3.716 | 3.436 | 1.081 | WIN |
| w8a16_linear | large | `[1024, 4096, 4096]` | 316.538 | 551.759 | 0.574 | LOSE |
| w8a16_linear | mid | `[256, 512, 512]` | 29.092 | 18.580 | 1.566 | WIN |
| w8a16_linear | tiny | `[32, 64, 64]` | 11.492 | 6.308 | 1.822 | WIN |

**Summary:** n=28 · WIN=25 · PARITY=2 · LOSE=1.

The single LOSE is **forced** W8A16 on a large shape. The default dispatcher
sends that shape to iu8, where the same class of work wins. Do not read
W8A16-large LOSE as a failure of W8A16 on the sizes it is meant for.

---

## Native iu4 GEMM and ConvRot int4

**Verdict first:** native iu4 is **faster than or even with HIP** on the shapes
we timed (WIN / PARITY). This is **not** the AWQ path.

Compiler: GFX120X `MmaAtom` accepts Int4 / iu4 and lowers to
`rocdl.wmma.i32.16x16x16.iu4` with **scalar i32** A/B (gfx12 ABI). Device smoke:
`tests/kernels/test_rdna4_integer_wmma_atom.py`.

Reusable GEMM: `kernels/gemm/rdna4_iu4_gemm.py` (packed `[M,K//2]` / `[N,K//2]`
int8 nibbles → scaled bf16/fp16/fp32 or raw i32).

**How to call it:**

* Bare GEMM: `iu4_gemm(a_packed, b_packed, ...)` — native when
  `shapes_ok_for_native_iu4` passes; pass `prefer_native=False` to force
  unpack→iu8.
* ConvRot linear: `convrot_w4a4_linear(x, qweight, wscales, ...)` —
  **default `linear_dtype="int4"`** runs device act ConvRot-i4 + native iu4
  GEMM (same idea as HIP’s int4 ConvRot linear). Pass `linear_dtype="int8"`
  only for the unpack→iu8 path. If the native shape gate fails, ConvRot falls
  back to unpack→iu8 automatically.

**Why the default is int4:** idle shows native int4 ConvRot at PARITY–WIN vs
HIP, and several times faster than FlyDSL’s own unpack→iu8 ConvRot on the same
tensors. Matching HIP’s int4 act quant also keeps the hot path aligned with
the reference stack. Activations are quantized to int4 on this path (different
numerics than int8 act quant); that is intentional.

AWQ and SVDQuant stay on their own modules. They are **not** “int4” in this
document’s naming.

Baseline: matching HIP ConvRot-W4A4 GEMM / linear. Method: warm=10 / rep=50 /
`do_bench` backlog median; one shape and backend per process.

| Op | Case | Shape | HIP µs | Fly µs | × | Verdict |
|---|---|---|---:|---:|---:|---|
| iu4_gemm (bare) | tiny | `[64, 64, 64]` | 10.524 | 10.016 | **1.051** | **WIN** |
| iu4_gemm (bare) | mid | `[128, 128, 128]` | 16.708 | 13.736 | **1.216** | **WIN** |
| iu4_gemm (bare) | 256 | `[256, 256, 256]` | 16.932 | 15.108 | **1.121** | **WIN** |
| convrot_w4a4_linear int4 | tiny | `[64, 64, 256]` | 17.236 | 17.112 | 1.007 | PARITY |
| convrot_w4a4_linear int4 | mid | `[128, 128, 512]` | 26.676 | 22.640 | **1.178** | **WIN** |

Same tensors, FlyDSL unpack→iu8 ConvRot vs native int4 (not HIP): tiny ~4.2×,
mid ~3.2× faster on the int4 path.

---

## AWQ W4A16 fused GEMV (separate family)

**Verdict first:** currently **LOSE** vs HIP (~×0.18–0.39). Still ships. This
is **not** native iu4.

Module: `kernels/quant/rdna4_awq_w4a16.py` (packed uint4 in-register dequant +
tiled GEMV). Tiling cut absolute time versus an earlier one-block path but did
not beat HIP. Correctness maxabs 0 vs the HIP reference. **No DROP.**

| Op | Case | Shape `[M,N,K]` G=64 | HIP µs | Fly µs | × | Verdict |
|---|---|---|---:|---:|---:|---|
| gemv_awq_w4a16 | tiny_m1 | `[1, 32, 128]` | 4.784 | 27.332 | 0.175 | LOSE |
| gemv_awq_w4a16 | small_m4 | `[4, 64, 256]` | 4.788 | 26.164 | 0.183 | LOSE |
| gemv_awq_w4a16 | mid_m1 | `[1, 256, 1024]` | 8.248 | 27.420 | 0.301 | LOSE |

---

## SVDQuant fused scaled_mm (separate family)

**Verdict first:** **WIN** on mid/large shapes; **LOSE** at M=1 (launch-bound).
Still ships. **No DROP.**

Module: `kernels/quant/rdna4_svdquant_w4a4.py` (N-tile=8, unpad-aware M, host
bf16 LoRA residual). Pad rules: `K % 64 == 0`; packed K multiple of 8.

| Op | Case | Shape `[M,N,K]` | HIP µs | Fly µs | × | Verdict |
|---|---|---|---:|---:|---:|---|
| scaled_mm_svdquant_w4a4 | m1n64k256 | `[1, 64, 256]` | 19.164 | 28.196 | 0.680 | LOSE |
| scaled_mm_svdquant_w4a4 | m4n64k256 | `[4, 64, 256]` | 49.596 | 38.548 | **1.287** | **WIN** |
| scaled_mm_svdquant_w4a4 | m8n128k512 | `[8, 128, 512]` | 102.968 | 37.856 | **2.720** | **WIN** |
| scaled_mm_svdquant_w4a4 | m16n256k1024 | `[16, 256, 1024]` | 128.360 | 40.376 | **3.179** | **WIN** |
| scaled_mm_svdquant_w4a4 | m32n256k1024 | `[32, 256, 1024]` | 229.209 | 51.948 | **4.412** | **WIN** |

**Summary:** n=5 · WIN=4 · LOSE=1 (M=1).

---

## Other notes

* **Multi-LoRA:** quantized base GEMM runs once; adapters stay small-rank host
  bf16 residuals. Fused FP8 and fused int8 keep that residual path in-tree.
* **`int8_linear_fused`:** fused act-quant + iu8 GEMM with **host** LoRA
  residual (same policy as `scaled_mm_fp8_fused`). Idle vs HIP int8 linear is
  currently **LOSE** (~×0.49–0.81). Ships and is flagged; no DROP without
  permission.
* **Ship policy:** no DROP without an explicit maintainer ask. Near-misses
  still ship.

## Suggested PR title (when opening is approved)

`[Kernel][Feature][Perf][Doc][Test] RDNA4 expanded ops: FlashAttention / attention (RoPE/RMS-RoPE/AdaLN), native iu4/int4 + iu8/int8 WMMA, int4/int8 quant, size-dispatch defaults`

(Do not open the PR until a maintainer explicitly says to open it.)


## FlashAttention / RDNA4 attention (gfx120x)

Status: dense bf16/fp16/fp8/int8/iu4; in-kernel bottom-right causal×cross; per-head ALiBi; sink/varlen/paged/split-K via ``flash_attn_gfx120x_ext``; FP8 e4m3fn+e5m2; iu4 kitchen-pack (native iu4 WMMA attempt + unpack→iu8 fallback). See ``tests/kernels/test_flash_attn_gfx120x.py``.

FlyDSL-native FA lives under `kernels/attention/flash_attn_*_gfx120x.py` with
host `flash_attn_gfx120x_host.py`. Lab idle medians (HIP coverage matrix,
2026-09-25-class shapes) previously showed:

| Shape | Notes |
|---|---|
| self_wan1024 B1 S1024 H16 D128 | WIN vs HIP (~1.53× in matrix) |
| cross_q128_k1024 H16 D64 | WIN (~1.32×) |
| cross Sq77 Sk1024 | was LOSE on older Q-pad path; current host skips Q pad + adaptive BLOCK_M |
| self_flux77 | was FALLBACK seq_pad; now HIT via `seq_len_kv_valid` |

Re-measure on this branch before quoting new idle numbers. D in `[64,256]`,
FP8 e4m3fn/e5m2 (+ descales), additive attn mask path included.

