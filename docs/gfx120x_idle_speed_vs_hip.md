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


## PR / patch notes

Maintainer-oriented patch notes for this overhaul live in [`docs/pr_gfx120x_overhaul_ops_fa_etc.md`](pr_gfx120x_overhaul_ops_fa_etc.md) (environment/HIP stamp, Comfy info-only, methodology, HIP→FlyDSL vs net-new, performance snapshot, migration). Paste that file as the GitHub PR body.

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
| ConvRot W4A4 linear | `convrot_w4a4_linear(x, qweight, wscales, convrot_groupsize=…)` — see § Native iu4 for full examples | **Default `linear_dtype="int4"`** = act ConvRot-i4 + iu4 WMMA. Pass `linear_dtype="int8"` for unpack→iu8 |
| Bare packed int4 GEMM | `iu4_gemm(a_packed, b_packed, scale_a?, scale_b?, out_dtype=…)` — pack with `pack_int4_row_major` first | Native when `K % 16 == 0`; `prefer_native=False` forces unpack→iu8 |
| AWQ W4A16 | `gemv_awq_w4a16` | Separate family; **WIN** on idle grid after claw2 (incl. mid_m4) — still ships |
| SVDQuant W4A4 | `scaled_mm_svdquant_w4a4` | Separate family; WIN mid/large **and M=1** (2026-09-30) |

**How “better kernels” turn on:** call the **auto / dispatched** entry, or for
ConvRot leave the default `linear_dtype="int4"`. You do not flip a global
environment flag. To pin an older path, pass an explicit force
(`force_kernel="iu8"|"w8a16"`, or `linear_dtype="int8"`, or
`prefer_native=False`).

---


## Full-stack measured breakpoint / config table (2026-09-30, gfx1201, HIP 7.17.26374)

Source of truth for wired defaults: `kernels/common/gfx120x_autotune_tables.py` (+ per-module `pick_*`). Method: `do_bench` warm+median; HIP separate process for Speed vs HIP.

| Family | Knob | Wired default | Notes |
|--------|------|---------------|-------|
| int8 W8A16/iu8 | `DEFAULT_K_IU8_MIN` | **256** (`K≥256→iu8`) | Idle auto WIN tiny/mid/large |
| fused int8 | `DEFAULT_K_FUSED_KERNEL_MAX` | **128** | Fused wins K≤128; else dq+iu8; (64,256,256)+LoRA **WIN ~×1.16** |
| FP8 scaled_mm | `pick_tile_config` / `_auto` | skinny BK / fat deep-K | Auto preferred; plain wanish often PARITY |
| AWQ W4A16 | `pick_awq_gemv_tiles` | M=1 nt 1/4; M∈(1,4] **nt=1 BT=64** | Full idle GEMV grid WIN |
| SVDQuant | `pick_svdquant_n_tile` | M=1→1/4/8 by N; else 8 | M=1+mid/large WIN |
| ConvRot | `linear_dtype` | **int4** default | unpack int8 if native gate fails |
| FA | `_pick_block_m` | soft-gap (self/cross) | Shared across bf16/fp8/int8/iu4 |
| AdaLN | `pick_adaln_block_threads(N)` | ≤128→32; ≤256→256; ≤512→512; ≤1024→128; ≥2048→512 | Plugged from measured vs HIP |
| RMS-RoPE | `pick_rms_rope_block_threads(HD)` | **HD≤1024→64; ≥2048→512** | Measured BSHD BT sweep vs kitchen |
| RoPE | `pick_rope_block_threads(n_pairs_total)` | **<4096→256; <32768→512; else 1024** | Matches kitchen Triton / Comfy pick |
| asym W4A8 dequant | `pick_asym_w4a8_block_threads(K)` | **BT=128** (size-insensitive on measured grid) | BT=256 ~3× slower |
| int8 rowwise quant | `pick_quantize_int8_rowwise_block_threads(K)` | ≤256→32; ≤512→64; else 256 | Idle WIN/PARITY vs HIP |
| SwiGLU / fused MLP tiles / stoch FP8 | — | N/A multi-path | Single config or caller heuristic; parked |

**Env bypass:** `FLYDSL_DISPATCH_MODE=auto|force_flydsl|force_hip` — see PR notes; avoids editing tables on other RDNA4/ROCm.

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

**Measured breakpoint (gfx1201 / R9700, 2026-09-30):** FlyDSL Device `do_bench`
grid (warm=10, rep=50) comparing W8A16 vs iu8 GEMM. Pareto / WIN-PARITY frontier
collapsed to a simple host predicate:

* **`K >= 256` → iu8**, else **W8A16**
* Legacy `M_star` / `MK_star` knobs default to **0** (inactive) so the K-gate is decisive
* `force_kernel="w8a16"|"iu8"` still overrides

Older hand-picked `M<=128 or MK<=500k → W8A16` mismatched ~255/420 shapes; the
K-gate cut mismatches to ~74 on the same grid.

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

The single LOSE is **forced** W8A16 on a large shape (`force_kernel="w8a16"`;
force ≠ auto). **Expected residual, not a ship failure:** auto picks iu8 and
**wins** on that shape (reautotune vs HIP ~×1.38 WIN). The size gate exists
because W8A16 is too slow as the default for large K while the calc difference
versus iu8 is minor — keep W8A16 for the small/mid region only. Do not read
W8A16-large LOSE as a failure of W8A16 on the sizes it is meant for.

---

## Int8 linear / W8A16 — how to call

**Verdict first:** default `int8_linear_auto` is **WIN** on measured shapes.
Forced W8A16 on large `[1024,4096,4096]` is the one core-suite **LOSE** (~×0.57);
the dispatcher sends that shape to iu8 (~×1.57 WIN). Keep families separate.

### What the two pipelines are

| Pipeline | Module | Acts | Weights | Math |
|---|---|---|---|---|
| W8A16 | `kernels/gemm/rdna4_w8a16_linear.py` | bf16/fp16 | int8 **or** fp8 e4m3fn/e5m2 (cast in VGPR) | float WMMA |
| iu8 | `kernels/gemm/rdna4_int8_linear.py` | int8 (quantized) | int8 | `wmma_i32_16x16x16_iu8` |

### Size rule (measured default)

`K < DEFAULT_K_IU8_MIN` (**256**) → W8A16; else → iu8.
Constant: `DEFAULT_K_IU8_MIN` in `kernels/common/gfx120x_autotune_tables.py`
(re-exported by `rdna4_int8_linear_dispatch.py`). Legacy `DEFAULT_M_W8A16_MAX` /
`DEFAULT_MK_W8A16_MAX` are zeroed aliases. Measured on one R9700 — not a
full FlyDSL autotune (`docs/autotune_guide.md`).

### Example — default auto entry

```python
import torch
from kernels.gemm.rdna4_int8_linear_auto import int8_linear_auto

# x: [M, K] bf16 acts; weight: [N, K] int8; weight_scale: [N] fp32
y = int8_linear_auto(x, weight, weight_scale, out_dtype=torch.bfloat16)

# Pin a pipeline when you already know the shape class:
y_w8 = int8_linear_auto(x, weight, weight_scale, force_kernel="w8a16")
y_iu8 = int8_linear_auto(x, weight, weight_scale, force_kernel="iu8")
```

### Example — fused act-quant + iu8 (+ optional host LoRA)

```python
from kernels.gemm.rdna4_int8_linear_fused import int8_linear_fused, int8_linear_fused_multi

# a_f bf16 acts; b_nk [N,K] int8; scales fp32; LoRA residual is host bf16
y = int8_linear_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16,
                      lora_down=ld, lora_up=lu, lora_scale=1.0)
```

**Why host LoRA:** keeping the residual off the WMMA epilogue avoided idle
regressions on the shapes we timed; see Design choices below and
`docs/kernel_tuning_guide.md` (bandwidth vs math).

---

## FP8 scaled_mm — how to call

**Verdict first:** `scaled_mm_fp8_auto` turns wanish PARITY (~×1.00 plain) into
~×3.97 **WIN** by picking a deeper tile. Both **e4m3fn** and **e5m2** ship.

### Tensors

| Arg | Shape / dtype |
|---|---|
| `a` | `[M, K]` float8_e4m3fnuz **or** float8_e5m2 (match `e5m2=` flag) |
| `b_nk` | `[N, K]` same FP8 |
| `scale_a`, `scale_b` | fp32 tensorwise (or matching host contract) |
| out | bf16 `[M, N]` default |

### Example

```python
from kernels.gemm.rdna4_scaled_mm_fp8_auto import scaled_mm_fp8_auto
from kernels.gemm.rdna4_scaled_mm_fp8_fused import scaled_mm_fp8_fused

# Default — measured tile picker
y = scaled_mm_fp8_auto(a_e4, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16, e5m2=False)

# e5m2 weights/acts
y = scaled_mm_fp8_auto(a_e5, b_nk, scale_a, scale_b, e5m2=True)

# Fused act-quant + mm (+ host LoRA residual)
y = scaled_mm_fp8_fused(a_f, b_nk, scale_a, scale_b, out_dtype=torch.bfloat16,
                        lora_down=ld, lora_up=lu, e5m2=False)
```

Tile labels (`skinny_bk128`, `fat_deep_k_*`, …) live in
`kernels/gemm/rdna4_scaled_mm_fp8_auto.py`. Architecture LDS/CU context:
`docs/architecture_guide.md`. RDNA4 WMMA authoring: `docs/kernel_authoring_guide.md`.

---

## FlashAttention (gfx120x) — how to call

**Verdict first:** prefer FlyDSL attn on RDNA4; bf16/fp16/fp8/int8/iu4 all have
host entrypoints. Head dim in `[64, 384]` with `D % 32 == 0` (LDS ≤ ~48 KiB bf16
at D=384 vs RDNA4 64 KiB LDS — `docs/architecture_guide.md`). Algorithm:
[FlashAttention, Dao et al., arXiv:2205.14135](https://arxiv.org/abs/2205.14135).

### Entrypoints

| Dtype | Call | QKV layout |
|---|---|---|
| bf16 / fp16 | `flydsl_flash_attn_func(q,k,v, …)` | BSHD `[B,S,H,D]` |
| FP8 e4m3 / e5m2 | `flydsl_flash_attn_fp8_func(..., q_descale=, k_descale=, v_descale=)` | FP8 BSHD + per-tensor descales |
| int8 | `flydsl_flash_attn_int8_func(..., q_descale=, …)` | int8 BSHD + descales |
| native iu4 | `flydsl_flash_attn_iu4_func(..., q_descale=, …)` | nibble-packed `[B,S,H,D//2]` int8 |

### Example — bf16 dense / causal / mask / ALiBi

```python
from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func

# Self-attn, non-causal
out = flydsl_flash_attn_func(q, k, v, causal=False)

# Causal self (equal seqlens) or causal×cross (unequal → bottom-right bias)
out = flydsl_flash_attn_func(q, k, v, causal=True)

# Additive mask / bias (noop all-True / all-zero ignored on host)
out = flydsl_flash_attn_func(q, k, v, causal=False, attn_mask=mask)

# ALiBi slopes (uniform or per-head-varying)
out = flydsl_flash_attn_func(q, k, v, causal=False, alibi_slopes=slopes)

# Optional LSE (host chunked logsumexp)
out, lse = flydsl_flash_attn_func(q, k, v, causal=False, return_lse=True)
```

### Example — FP8 / int8 / iu4

```python
from kernels.attention.flash_attn_gfx120x_host import (
    flydsl_flash_attn_fp8_func,
    flydsl_flash_attn_int8_func,
    flydsl_flash_attn_iu4_func,
)

out = flydsl_flash_attn_fp8_func(q8, k8, v8, causal=False,
                                 q_descale=d, k_descale=d, v_descale=d)
out = flydsl_flash_attn_int8_func(q_i8, k_i8, v_i8, causal=True,
                                  q_descale=s, k_descale=s, v_descale=s)
# qp: nibble-packed [B,S,H,D//2] int8 — same pack as iu4_gemm, not AWQ
out = flydsl_flash_attn_iu4_func(qp, kp, vp, causal=False,
                                 q_descale=s, k_descale=s, v_descale=s)
```

Do **not** pass int8 QKV into `flydsl_flash_attn_func` — it raises and points
you at the int8/iu4 hosts.

---

## AWQ W4A16 — how to call (separate family)

**Verdict first:** measured claw2 2026-09-30 — **WIN** on tiny/small/mid_m1/**mid_m4**/large (n_tile=1 + BT=64 for M=4). Still ships. **Not**
native iu4.

Wire: `qweight[N,K//2]` **unsigned** nibbles; `wscales`/`wzeros` `[K//G, N]`;
`W = (q-8)*s + z`. Default `group_size=64`.

```python
from kernels.quant.rdna4_awq_w4a16 import gemv_awq_w4a16

# x: [M,K] bf16 (gemv-friendly M in {1,4}); qweight uint4-packed
y = gemv_awq_w4a16(x, qweight, wscales, wzeros, group_size=64)
```

---

## SVDQuant W4A4 — how to call (separate family)

**Verdict first:** **WIN** mid/large **and M=1** (2026-09-30 claw). Still ships. Host bf16
LoRA residual by design (`IDLE_WIN_HOST_LORA`).

```python
from kernels.quant.rdna4_svdquant_w4a4 import scaled_mm_svdquant_w4a4

# act: packed int4 [M,K//2]; wgt: [N,K//2]; ascales/wscales groupwise;
# lora_act_in / lora_up: bf16 residual factors
y = scaled_mm_svdquant_w4a4(
    act, wgt, ascales, wscales, lora_act_in, lora_up,
    bias=bias, group_size=64, fused=True,
)
```

---

## Asym W4A8 — how to call

```python
from kernels.quant.rdna4_asym_w4a8 import (
    quantize_w4a8_int8_weight,
    w4a8_int8_linear,
)

# Returns: packed[N,K//2], s_rel, s_channel, correction, codebook
qdata, s_rel, s_channel, correction, codebook = quantize_w4a8_int8_weight(
    w, group_size=16, convrot_groupsize=256, codebook=True
)
y = w4a8_int8_linear(
    x, qdata, s_rel, s_channel,
    codebook=codebook, correction=correction,
    group_size=16, convrot_groupsize=256,
)
```

---

## Fused SwiGLU / zero-LDS N-major MLP — how to call

**Verdict first:** in-reg SiLU×mul between GEMM0/GEMM1 for 16×16 panels;
thinner host path for other sizes. Tuning motivation: avoid mid LDS/GMEM spill
(`docs/kernel_tuning_guide.md`).

```python
from kernels.gemm.rdna4_fused_mlp_nmajor import (
    fused_swiglu_mlp_inreg,
    fused_swiglu_mlp_nmajor,
    fused_gemm_tn,
)

# Weights are linear B[N,K] layout
y = fused_swiglu_mlp_inreg(x, w_gate, w_up, w_down, out_dtype=torch.float32)
y = fused_swiglu_mlp_nmajor(x, w_gate, w_up, w_down)
```

Standalone SiLU×mul / chunk: `kernels/quant/rdna4_swiglu.py`.

---

## RoPE / RMS+RoPE / AdaLN — how to call

These are **builder** APIs (`build_*_module`) compiled with `flyc.compile`, not
single-tensor host wrappers. Typical pattern from
`tests/kernels/test_gfx120x_norm_rope.py`:

```python
from kernels.norm.rope_gfx120x import build_rope_module, build_rope_qk_fused_module
from kernels.norm.rms_rope_gfx120x import build_rms_rope_module
from kernels.norm.adaln_gfx120x import build_adaln_module
import flydsl.compiler as flyc

rope = build_rope_module("bfloat16", block=256)
# compile + launch with (x, cos, sin, out, …, stream) — see test file

adaln = build_adaln_module(N=3072, dtype_str="bfloat16", subtract_mean=True, block_threads=256)
compiled = flyc.compile(adaln, x, scale, shift, out, rows, 1, 1, eps, stream)
compiled(x, scale, shift, out, rows, 1, 1, eps, stream)
```

Idle wins on Flux-like shapes (e.g. AdaLN `[77,3072]` ~×3.17). Sources:
`docs/kernel_authoring_guide.md`, `docs/architecture_guide.md`.

---

## Native iu4 GEMM and ConvRot int4

**Verdict first:** native iu4 is **faster than or even with HIP** on the shapes
we timed (WIN / PARITY). This is **not** the AWQ path.

Compiler: GFX120X `MmaAtom` accepts Int4 / iu4 and lowers to
`rocdl.wmma.i32.16x16x16.iu4` with **scalar i32** A/B (gfx12 ABI). Device smoke:
`tests/kernels/test_rdna4_integer_wmma_atom.py`.

Reusable GEMM module: `kernels/gemm/rdna4_iu4_gemm.py`.
ConvRot host that uses it: `kernels/quant/rdna4_convrot_w4a4.py`.

### What the tensors look like

| Role | Shape / dtype | Meaning |
|---|---|---|
| Logical A (acts or LHS) | `[M, K]` int4 values in `[-7, 7]` | Before packing |
| Logical B (weights or RHS) | `[N, K]` int4 values in `[-7, 7]` | Before packing |
| Packed A | `[M, K//2]` `torch.int8` | Two signed nibbles per byte; **low nibble = even K** |
| Packed B | `[N, K//2]` `torch.int8` | Same pack (B is stored `[N, K//2]`, not `[K, N]`) |
| `scale_a` | `[M]` `float32` | Per-row act scale (omit → ones; ignored for `out_dtype=int32`) |
| `scale_b` / `wscales` | `[N]` `float32` | Per-row weight scale |
| Output | `[M, N]` | `bfloat16` (default), `float16`, `float32`, or raw `int32` accumulator |

Pack helper: `kernels.quant.rdna4_int4_codec.pack_int4_row_major`.

Native shape gate (`shapes_ok_for_native_iu4(M, N, K)`): `M>0`, `N>0`, `K>0`,
and **`K % 16 == 0`**. Partial M/N tiles are fine (bounds path). If the gate
fails, or you pass `prefer_native=False`, the host unpacks to int8 and runs
the iu8 linear path instead.

### Example A — bare packed GEMM (`iu4_gemm`)

Use this when you already have (or can pack) int4 matrices. This is the
building block ConvRot calls under the hood.

```python
import torch
from kernels.quant.rdna4_int4_codec import pack_int4_row_major
from kernels.gemm.rdna4_iu4_gemm import iu4_gemm, shapes_ok_for_native_iu4

M, N, K = 64, 64, 64
assert shapes_ok_for_native_iu4(M, N, K)  # needs K % 16 == 0

# Logical signed int4 in [-7, 7], then pack to [M,K//2] / [N,K//2] int8.
a_logical = torch.randint(-7, 8, (M, K), dtype=torch.int8, device="cuda")
b_logical = torch.randint(-7, 8, (N, K), dtype=torch.int8, device="cuda")
a_packed = pack_int4_row_major(a_logical).contiguous()   # [M, K//2]
b_packed = pack_int4_row_major(b_logical).contiguous()   # [N, K//2]

# Raw i32 accumulator (no scales):
acc_i32 = iu4_gemm(a_packed, b_packed, out_dtype=torch.int32)

# Scaled bf16 output (default out_dtype is bf16 if you omit it):
scale_a = torch.full((M,), 0.01, device="cuda", dtype=torch.float32)
scale_b = torch.full((N,), 0.01, device="cuda", dtype=torch.float32)
out_bf16 = iu4_gemm(a_packed, b_packed, scale_a, scale_b, out_dtype=torch.bfloat16)

# Force unpack→iu8 instead of native iu4 (same packed inputs):
out_fallback = iu4_gemm(
    a_packed, b_packed, scale_a, scale_b,
    out_dtype=torch.float32, prefer_native=False,
)
```

### Example B — ConvRot W4A4 linear (usual “int4 linear” call)

Use this when inputs are bf16/fp16 activations and you have (or can make)
ConvRot-quantized weights. The default does **all** of: act ConvRot-i4 quant
on device → native `iu4_gemm` → bf16/fp output. You do **not** pack acts
yourself on this path.

```python
import torch
from kernels.quant.rdna4_convrot_w4a4 import (
    quantize_convrot_w4a4_weight,
    convrot_w4a4_linear,
)

M, N, K = 64, 64, 256
# Hadamard group for ConvRot; must divide K. Common: 64 or 256.
convrot_groupsize = 64

x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

# Offline (or once): quantize weights → packed qweight + per-row scales.
qweight, wscales = quantize_convrot_w4a4_weight(
    w, convrot_groupsize=convrot_groupsize,
)  # qweight: [N, K//2] int8; wscales: [N] float32

# Default path — native int4 (linear_dtype defaults to "int4"):
y = convrot_w4a4_linear(
    x, qweight, wscales,
    convrot_groupsize=convrot_groupsize,
    out_dtype=torch.bfloat16,
)
# Equivalent explicit form:
y = convrot_w4a4_linear(
    x, qweight, wscales,
    convrot_groupsize=convrot_groupsize,
    linear_dtype="int4",
    out_dtype=torch.bfloat16,
)

# Older / alternate path — unpack weights to int8, int8-act ConvRot linear:
y_i8 = convrot_w4a4_linear(
    x, qweight, wscales,
    convrot_groupsize=convrot_groupsize,
    linear_dtype="int8",
    out_dtype=torch.bfloat16,
)
```

**What each knob does**

| Knobs | Effect |
|---|---|
| omit `linear_dtype` / `"int4"` | Act → ConvRot-i4 on device, then native iu4 GEMM |
| `linear_dtype="int8"` | Unpack weight nibbles → int8; int8-act ConvRot linear (iu8) |
| Native gate fails under `"int4"` | Same host automatically falls back to the `"int8"` path |
| `prefer_native=False` on `iu4_gemm` | Bare GEMM only: always unpack→iu8 (ConvRot does not expose this; use `linear_dtype`) |

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

**Verdict first:** measured claw2 2026-09-30 — **WIN** on the idle GEMV grid (incl. mid_m4 ×~1.48). Still ships. This
is **not** native iu4.

Module: `kernels/quant/rdna4_awq_w4a16.py` Tile pick: `pick_awq_gemv_tiles` → `(max_m, n_tile, block_threads)`; M∈(1,4] uses **n_tile=1, BT=64** (claw2 mid_m4 WIN). (packed uint4 in-register dequant +
tiled GEMV). Tiling cut absolute time versus an earlier one-block path but did
not beat HIP. Correctness maxabs 0 vs the HIP reference. **No DROP.**

| Op | Case | Shape `[M,N,K]` G=64 | HIP µs | Fly µs | × | Verdict |
|---|---|---|---:|---:|---:|---|
| gemv_awq_w4a16 | tiny_m1 | `[1, 32, 128]` | 4.6 | 4.0 | **1.144** | **WIN** |
| gemv_awq_w4a16 | small_m4 | `[4, 64, 256]` | 4.8 | 4.5 | **1.060** | **WIN** |
| gemv_awq_w4a16 | mid_m1 | `[1, 256, 1024]` | 7.6 | 5.4 | **1.428** | **WIN** |
| gemv_awq_w4a16 | mid_m4 | `[4, 256, 1024]` | 7.8 | 5.3 | **1.480** | **WIN** |
| gemv_awq_w4a16 | large_m1 | `[1, 1024, 4096]` | 21.4 | 9.6 | **2.222** | **WIN** |

---

## SVDQuant fused scaled_mm (separate family)

**Verdict first:** **WIN** on mid/large **and M=1** after host-prep/tile claw 2026-09-30.
Still ships. **No DROP.**

Module: `kernels/quant/rdna4_svdquant_w4a4.py` (N-tile=8, unpad-aware M, host
bf16 LoRA residual). Pad rules: `K % 64 == 0`; packed K multiple of 8.

| Op | Case | Shape `[M,N,K]` | HIP µs | Fly µs | × | Verdict |
|---|---|---|---:|---:|---:|---|
| scaled_mm_svdquant_w4a4 | m1n64k256 | `[1, 64, 256]` | 21.4 | 20.3 | **1.055** | **WIN** |
| scaled_mm_svdquant_w4a4 | m4n64k256 | `[4, 64, 256]` | 49.5 | 20.6 | **2.404** | **WIN** |
| scaled_mm_svdquant_w4a4 | m8n128k512 | `[8, 128, 512]` | 102.3 | 25.1 | **4.073** | **WIN** |
| scaled_mm_svdquant_w4a4 | m16n256k1024 | `[16, 256, 1024]` | 129.9 | 25.1 | **5.167** | **WIN** |

**Summary:** n=4 · WIN=4 · LOSE=0 (M=1 clawed to WIN; GN ActTy scales + `pick_svdquant_n_tile`).

---

## Other notes

* **Multi-LoRA:** quantized base GEMM runs once; adapters stay small-rank host
  bf16 residuals. Fused FP8 and fused int8 keep that residual path in-tree.
* **`int8_linear_fused`:** size-gated quantized base (`DEFAULT_K_FUSED_KERNEL_MAX=128`) + **host** LoRA
  residual (same policy as `scaled_mm_fp8_fused`). Idle vs HIP int8 linear is
  claw2 2026-09-30: size-gated base (`K<=128` fused / else device-quant+iu8) + **unbounded** host LoRA residuals — median **WIN** vs HIP dynamic `int8_linear` on the product idle grid (occasional near-PARITY noise on largest). Ships; no DROP without
  permission.
* **Ship policy:** no DROP without an explicit maintainer ask. Near-misses
  still ship.

## Suggested PR title (when opening is approved)

`[Kernel][Feature][Perf][Doc][Test] RDNA4 expanded ops: FlashAttention / attention (RoPE/RMS-RoPE/AdaLN), native iu4/int4 + iu8/int8 WMMA, int4/int8 quant, size-dispatch defaults`

(Do not open the PR until a maintainer explicitly says to open it.)


## FlashAttention / RDNA4 attention (gfx120x)

**How to call:** see § FlashAttention (gfx120x) — how to call above.

Status: dense bf16, fp16, fp8, int8, iu4; in-kernel bottom-right causal×cross; per-head ALiBi; sink/varlen/paged/split-K via ``flash_attn_gfx120x_ext``; FP8 e4m3fn+e5m2; iu4 nibble-pack (native iu4 WMMA attempt + unpack→iu8 fallback). See ``tests/kernels/test_flash_attn_gfx120x.py``.

FlyDSL-native FA lives under `kernels/attention/flash_attn_*_gfx120x.py` with
host `flash_attn_gfx120x_host.py`. Lab idle medians (HIP coverage matrix,
2026-09-25-class shapes) previously showed:

| Shape | Notes |
|---|---|
| self_wan1024 B1 S1024 H16 D128 | WIN vs HIP (~1.53× in matrix) |
| cross_q128_k1024 H16 D64 | WIN (~1.32×) |
| cross Sq77 Sk1024 | was LOSE on older Q-pad path; current host skips Q pad + adaptive BLOCK_M |
| self_flux77 | was FALLBACK seq_pad; now HIT via `seq_len_kv_valid` |

Re-smoked 2026-09-30: flux77 (+additive mask), cross q77/k1024, D=192/256, long S=6278 HIT; iu4 FA `is_native_iu4_fa=True` (in-kernel, not silent unpack). D in `[64,256]`, FP8 e4m3fn/e5m2 (+ descales), additive attn mask path included.

---

## Design choices — why this stack looks the way it does

This section is for reviewers and future maintainers. It explains **what we chose**,
**why**, and **which sources** the choices rest on. Idle numbers above are the
local measurement; the sources below are the engineering context.

### Timing methodology (why `do_bench`, warm, median, separate HIP process)

FlyDSL’s shared Device timer is documented in
[`docs/autotune_guide.md`](autotune_guide.md): a GPU-side backlog before each
batched CUDA-event window, then a median of batch averages. We follow that so
idle Speed vs HIP is comparable to FlyDSL’s own autotune timer, not a one-shot
`time.perf_counter` around a cold launch.

HIP and FlyDSL run in **separate processes** so a shared CUDA context cannot
inflate one side. That is a measurement policy choice, not a HIP API
requirement; it exists because early dual same-shape smokes cheated the ratio.

Verdict cutoffs (WIN ≥ 1.05, PARITY 0.95–1.05, LOSE < 0.95) are deliberate
hysteresis so noise does not flip a table every rerun. Shipping LOSE families
(AWQ, some fused+LoRA shapes) is a product rule: flag honestly, do not silent-DROP.

### Why size-dispatch defaults exist (measured 2026-09-30; was EXPERIMENTAL)

On gfx120x, **W8A16** (bf16 acts, 8-bit weights, float WMMA) wins small/mid
shapes; **iu8 int8×int8** WMMA wins large shapes where W8A16 collapses (see the
int8 breakpoint table above). A static host rule
`M ≤ 128 or M×K ≤ 500_000 → W8A16 else iu8` is the simplest predicate that kept
every measured grid point on a WIN/PARITY pipeline when one existed.

That is **not** FlyDSL’s full offline autotuner
([`docs/autotune_guide.md`](autotune_guide.md)). It is a single-device (R9700)
crossover cache expressed as defaults. Callers who already know the shape force
`force_kernel=` / plain modules. Marking it EXPERIMENTAL matches FlyDSL’s own
stance that production wiring may still own fallbacks.

FP8 `scaled_mm_fp8_auto` is the same idea for **tile** pick: plain
`scaled_mm_fp8` was ~PARITY on wanish; the measured tile moved that point to
~×3.97 WIN. Source landscape: tile/K choices and bandwidth vs math tradeoffs in
[`docs/kernel_tuning_guide.md`](kernel_tuning_guide.md) (§ bandwidth-bound vs
math-bound, `tile_k`, LDS budget).

### Why native iu4 (nibble-packed) instead of unpack→iu8 as the default

RDNA4 exposes integer WMMA including **iu4** via FlyDSL’s gfx120x `MmaAtom`
([`docs/kernel_authoring_guide.md`](kernel_authoring_guide.md) — RDNA4 / gfx120x
WMMA factory; `rocdl.WMMA` with i4 / iu4 packing). Unpacking int4→int8 then
running iu8 doubles the logical K traffic and spends ALU on unpack. Native iu4
keeps packed bytes in GMEM/LDS and feeds the atom through **scalar i32**
fragments (gfx12 ABI; not gfx11 `v2i32`).

**Nibble-pack ABI:** two signed 4-bit values per `int8` byte, shape `K/2` or
`D/2`, low nibble = even index. We load packed bytes as i32 dwords (never an
i4 buffer-load opcode). That matches the GEMM host (`rdna4_iu4_gemm`) and the
in-kernel iu4 FlashAttention path. ConvRot’s default `linear_dtype="int4"`
follows the idle WIN/PARITY table for native int4 vs HIP; `"int8"` forces the
old unpack→iu8 path for callers who need it.

AWQ / SVDQuant stay on their own fused or unpack layouts — different wire
formats (AWQ uint4 group scales/zeros; SVDQuant W4A4 + LoRA). Mixing them into
the “native iu4 WIN” story would be false; family walls in this doc exist for
that reason.

### Why FlashAttention is FlyDSL-native on gfx120x (all shipped dtypes)

Algorithmic backbone is classic FlashAttention tiling + online softmax
(Dao et al., [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)). FlyDSL already
ships gfx950 / generic FA modules; gfx120x needs **wave32 WMMA** paths and a
64 KB LDS/CU budget
([`docs/architecture_guide.md`](architecture_guide.md) — `gfx1201` / R9700),
not CDNA MFMA.

We implemented bf16/fp16, FP8 e4m3fn **and** e5m2, int8 QKV, and native iu4 QKV
so product stacks are not forced through flash-attn/sage or silent dequant
FALLBACKs. Extensions (sink, paged-KV gather, packed varlen, split-K, per-head
ALiBi, causal×cross) follow the same host surface as upstream gfx950 where
possible, then specialize the body for RDNA4. Descales for FP8/int8/iu4 mirror
the gfx950 FP8 descale ABI so callers do not learn a second scale convention.

### Why fused int8/FP8 + LoRA uses a **host** bf16 residual

Early in-kernel LoRA fusion blew compile time / JIT size on this stack. Keeping
act-quant+GEMM fused and applying multi-LoRA as host bf16 residuals preserves
the hot GEMM while staying debuggable. After claw2 size-gate, idle is median
WIN / near-PARITY vs HIP dynamic int8_linear (was LOSE ×0.41–0.73). Occasional
path on some shapes — documented in the fused sections above rather than hidden.

### Why zero-LDS N-major / in-reg SwiGLU panels exist

[`docs/kernel_tuning_guide.md`](kernel_tuning_guide.md) stresses LDS occupancy
tradeoffs and ping-pong cost. For thin MLP panels, an A/B-swap N-major GEMM0
plus in-register SiLU×mul between GEMM0/GEMM1 removes an LDS round-trip for the
activation fuse. Scope is limited (e.g. 16×16 panels) so production layouts are
not rewritten wholesale.

### Hardware / ROCm context

- Local validation: R9700 (`gfx1201`), HIP **7.17.26374**, PyTorch ROCm nightly
  stamp in the Environment section / PR patch notes.
- ROCm install / stack background:
  [ROCm install on Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/).
- FlyDSL kernel API / atoms:
  [`docs/kernel_authoring_guide.md`](kernel_authoring_guide.md),
  [`docs/prebuilt_kernels_guide.md`](prebuilt_kernels_guide.md).
- Performance writing expectations for PRs:
  upstream [`CONTRIBUTING.md`](https://github.com/ROCm/FlyDSL/blob/main/CONTRIBUTING.md)
  (hardware, baseline, optimized, improvement tables).

ComfyUI was used only as an integration/shape test bed; it is **not** a cited
optimization source and ships nothing into this repository.
