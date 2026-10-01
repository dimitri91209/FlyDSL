# PR / patch notes source: gfx120x-overhaul-ops-fa-etc

Copy this file into the GitHub pull-request description when opening. Keep in sync with Desktop kit 13 `PR_BODY.md`.

<!-- Paste this entire file as the GitHub pull-request description.
     Branch: gfx120x-overhaul-ops-fa-etc
     Title: see Desktop kit 13 PR_TITLE.txt / commit subject
     This is intentionally denser than the CONTRIBUTING template — treat it as
     patch notes for a major gfx120x overhaul so maintainers do not have to
     reconstruct the story from the diff. -->

# Patch notes — gfx120x / RDNA4 overhaul

**Branch:** `gfx120x-overhaul-ops-fa-etc`  
**Form:** one squashed commit (full multi-commit history kept in local backup refs)  
**Credit:** dimitri91209 + Grokbot  
**Suggested version impact:** this change is large enough to warrant a **minor (or greater) FlyDSL version bump** when maintainers cut a release — new matrix-multiply atoms, new FlashAttention query/key/value dtypes, new size-dispatch defaults, ahead-of-time export coverage, and new documentation.

**Validated on:** AMD Radeon AI PRO R9700 (`gfx1201` / family **gfx120x**), HIP **7.17.26374**, PyTorch **2.15.0a0+rocm10.2.0a20260924**.

Compare (fork branch vs upstream `main`, three dots):  
`https://github.com/ROCm/FlyDSL/compare/main...dimitri91209:FlyDSL:gfx120x-overhaul-ops-fa-etc`

---

## Summary

This branch adds an RDNA4 / **gfx120x** package for ROCm FlyDSL: native **iu4 / int4** and **iu8 / int8** wave-matrix multiply-accumulate atoms, size-dispatch defaults taken from measured idle Speed-versus-HIP breakpoints on one R9700, and FlashAttention for the full shipped query/key/value dtype stack:

| Query/key/value dtype | Host entry | Kernel module |
|-----------------------|------------|---------------|
| bfloat16 / float16 | `flydsl_flash_attn_func` | `flash_attn_gfx120x.py` |
| **float8_e4m3fn** and **float8_e5m2** (FP8 query/key/value) | `flydsl_flash_attn_fp8_func` | `flash_attn_fp8_gfx120x.py` |
| int8 query/key/value | `flydsl_flash_attn_int8_func` | `flash_attn_int8_gfx120x.py` |
| packed native iu4 query/key/value | `flydsl_flash_attn_iu4_func` | `flash_attn_iu4_gfx120x.py` + `_flash_attn_iu4_native_*` |

FlashAttention protocol coverage on gfx120x includes sink tokens, paged key/value gather, packed variable-length sequences, split-K, per-head ALiBi, and causal×cross attention for FP8 and int8 **without** a dequant-to-bfloat16 fallback path.

Also included: expanded general matrix multiply, quantization, normalization, rotary position embedding (RoPE), RMS-RoPE, AdaLN, SwiGLU, AWQ, SVDQuant, and ConvRot paths; shared measured dispatch tables; ahead-of-time export examples; idle Speed-versus-HIP tables; and prebuilt-kernel guide updates.

**Python hygiene:** every `from __future__ import annotations` line that appeared on this branch versus `origin/main` was removed (19 real imports scrubbed to zero) so mid-file future-imports cannot raise `SyntaxError` and `@fx.struct` / array sizes keep live types.


## Opening (approved sense)

This pull request ships **FlyDSL FlashAttention** for the gfx120x family (not flash-attn / SageAttention): full query/key/value dtypes (bfloat16 / float16, float8_e4m3fn + float8_e5m2, int8, native iu4), expanded ops, ahead-of-time export coverage, and `FLYDSL_DISPATCH_MODE` as a **gfx120x-only** size-gate bypass (never a cross-arch unlock). Public hosts go through `require_gfx120x`. Size-dispatch defaults come from a **gfx1201-only** full-stack measured autotune (`kernels/common/gfx120x_autotune_tables.py`). HIP sanity was checked with local PyTorch **2.15.0a0+rocm10.2.0a20260924** and HIP **7.17.26374** in a ComfyUI real-use case.

---

## Size-dispatch and measured autotune tables (gfx120x only)

Defaults for `*_auto` / `_dispatch` / block-threads and tile pickers live in `kernels/common/gfx120x_autotune_tables.py`. They were **filled from measured idle Speed-versus-HIP and tile/block sweeps on one AMD Radeon AI PRO R9700 (`gfx1201`, family gfx120x)** with HIP **7.17.26374** — not hand-tuned guesses.

**Architecture gate:** every public host that uses these tables goes through `kernels.common.gfx120x_arch.require_gfx120x` / `is_gfx120x` (FlashAttention host re-exports the shared helper; `gemm_bf16_nmajor`, `quantize_int8_tensorwise`, and `quantize_w4a8_int8_weight` are gated the same way). **Other GPU architectures never enter these paths** — they raise `ValueError`. The arch detector is the primary fence; other arches should not see or depend on these defaults.

**Not a portable product autotune API:** the numbers are one-SKU review defaults documented in `docs/gfx120x_idle_speed_vs_hip.md`. They are not a general FlyDSL multi-device autotuner contract. On another RDNA4 board or ROCm stack, remeasure or use the local bypass below.

### FLYDSL_DISPATCH_MODE (RDNA4-local bypass only)

Environment override for **gfx120x hosts that already passed the arch gate**. It does **not** unlock these paths on other architectures.

| Value | Effect |
|-------|--------|
| `auto` (default) | Use the measured size / tile / block-threads gates |
| `force_flydsl` | Prefer FlyDSL native / wave-matrix paths (for example int8 → iu8, ConvRot → native int4; fused int8 ignores the K-gate for the in-kernel path) |
| `force_hip` | Route `*_auto` / fused / FP8 auto helpers to Comfy-Kitchen HIP when available; ConvRot → unpack then int8 |

Full re-autotune sanity on this ship (gates + non-gate single-config smokes): **PASS=83, FAIL=0, SKIP=1**. Remaining documented LOSE: forced W8A16 at large MNK only (auto picks iu8 and wins).

---

## Dependency chain (review order)

Read and review in this order so later layers make sense:

1. **Dialect / atoms** — `include/flydsl/Dialect/FlyROCDL/IR/MmaAtom.td` (GFX120X **iu8** and **iu4** `MmaAtom` / wave-matrix types).
2. **FlashAttention kernels** — `kernels/attention/flash_attn_*_gfx120x.py` plus `_flash_attn_iu4_native_*` bodies; host `flash_attn_gfx120x_host.py` (`_pick_block_m`, dtype entrypoints); interface `flash_attn_interface.py` (`attn_mask` alias merges into bias).
3. **Ops** — general matrix multiply (`rdna4_int8_linear`, `rdna4_w8a16_linear`, `rdna4_iu4_gemm`, `rdna4_scaled_mm_fp8*`), quantization (`rdna4_awq_*`, `rdna4_svdquant_*`, `rdna4_convrot_*`, int8 rowwise/tensorwise, asymmetric W4A8), norm/RoPE/AdaLN (`kernels/norm/*_gfx120x.py`), SwiGLU / fused multilayer perceptron.
4. **Dispatch** — size / tile / block-threads gates: `rdna4_int8_linear_dispatch` / `_auto`, `rdna4_scaled_mm_fp8_auto`, fused int8 `DEFAULT_K_FUSED_KERNEL_MAX`, `pick_awq_gemv_tiles`, `pick_svdquant_n_tile`, ConvRot `linear_dtype`, FlashAttention `_pick_block_m`, norm `_block_threads`, asymmetric W4A8 `pick_asym_w4a8_block_threads`. **Shared measured constants:** `kernels/common/gfx120x_autotune_tables.py`. **Arch gate:** `kernels/common/gfx120x_arch.py`. **Environment bypass (gfx120x only):** `FLYDSL_DISPATCH_MODE` in `kernels/common/dispatch_mode.py`.
5. **Tests / ahead-of-time** — `tests/kernels/test_flash_attn_gfx120x.py` (45/45 on this pass), rdna4 suite, `tests/python/examples/aot_gfx120x_example.py`.

---

## Ahead-of-time export coverage

`tests/python/examples/aot_gfx120x_example.py` exports gfx120x **RoPE**, **W8A16 linear**, and **bfloat16 FlashAttention** by default; `--all` also covers float8 / int8 / native iu4 FlashAttention, iu4 general matrix multiply, AWQ, SVDQuant, and ConvRot. FlashAttention host launches expose `.jit_function` for the export path. See `test_aot_gfx120x_rope_and_w8a16_export` / `test_aot_gfx120x_flash_attn_export`.

---

## What landed (by surface)

### A) Speed-versus-HIP counterparts (idle-measured)

| Area | Entrypoints | Notes |
|------|-------------|-------|
| Int8 / W8A16 linear | `rdna4_int8_linear`, `rdna4_w8a16_linear`, `int8_linear_auto` | Size-dispatch picks W8A16 versus iu8; `DEFAULT_K_IU8_MIN = 256` |
| FP8 `scaled_mm` | `scaled_mm_fp8`, `scaled_mm_fp8_auto` | **Both** `float8_e4m3fn` and `float8_e5m2` |
| ConvRot W4A4 | `convrot_w4a4_linear` | Default **native int4**; `"int8"` forces unpack then iu8 |
| Quant / dequant | int8 tensorwise / rowwise helpers | Hot-path helpers for linear and FlashAttention |
| Norm / RoPE / RMS-RoPE / AdaLN | gfx120x elementwise | Versus HIP / PyTorch references |
| SwiGLU / fused multilayer perceptron | fused FlyDSL versus HIP sequential | Multi-wave panels; fixed compile-time tiles (see N/A below) |
| FlashAttention bfloat16 / float16 | `flydsl_flash_attn_func` | FlyDSL-native versus prior HIP / external attention |

### B) New surface — no matching HIP counterpart (FlyDSL-first)

| Area | What landed |
|------|-------------|
| Compiler / atoms | GFX120X **iu8** + **iu4** `MmaAtom` / wave-matrix multiply-accumulate |
| Native iu4 general matrix multiply | `rdna4_iu4_gemm` / `iu4_gemm` (not unpack→iu8) |
| Native iu4 query/key/value FlashAttention | `flydsl_flash_attn_iu4_func` — **in-kernel** iu4 (not unpack→iu8) |
| Int8 query/key/value FlashAttention | `flydsl_flash_attn_int8_func` — int8 query/key/value, FP8-style loads + per-tensor descales |
| **FP8 query/key/value FlashAttention** | `flydsl_flash_attn_fp8_func` — query/key/value `float8_e4m3fn` **and** `float8_e5m2` + per-tensor descales (`flash_attn_fp8_gfx120x.py`) |
| FlashAttention protocol extensions | sink, paged key/value gather, packed variable-length, split-K, per-head ALiBi, causal×cross for FP8/int8 without dequant fallback |
| Size-dispatch defaults | `*_auto` / `_dispatch` + **R9700-measured** autotune tables; gfx120x-only via `require_gfx120x` |
| AWQ W4A16 fused general matrix-vector | FlyDSL-first; idle often **LOSE** historically — ships honestly; current measured grid **WIN** after tile claw |
| SVDQuant W4A4 | FlyDSL-first; WIN mid/large; M=1 clawed to WIN where noted |
| Fused int8 + host LoRA keep | One quantized base general matrix multiply + host bfloat16/float16 residuals for **unbounded N** adapters; `DEFAULT_K_FUSED_KERNEL_MAX = 128` |

### FlashAttention query/key/value entrypoints (gfx120x)

| Query/key/value dtype | Host entry | Kernel module |
|-----------------------|------------|---------------|
| bfloat16 / float16 | `flydsl_flash_attn_func` | `flash_attn_gfx120x.py` |
| **float8_e4m3fn / float8_e5m2** | `flydsl_flash_attn_fp8_func` | `flash_attn_fp8_gfx120x.py` |
| int8 | `flydsl_flash_attn_int8_func` | `flash_attn_int8_gfx120x.py` |
| native iu4 (packed) | `flydsl_flash_attn_iu4_func` | `flash_attn_iu4_gfx120x.py` + `_flash_attn_iu4_native_*` |

All four are first-class ship surface (not int8-only). Descale ABI for FP8 / int8 / iu4. Do **not** pass int8 query/key/value into `flydsl_flash_attn_func` — it raises and points at `flydsl_flash_attn_int8_func`.

### C) Docs / tests / packaging

- `docs/gfx120x_idle_speed_vs_hip.md` — methodology + full Speed-versus-HIP tables (family walls; verdict-first)
- `docs/prebuilt_kernels_guide.md` — how to call each gfx120x path
- `docs/pr_gfx120x_overhaul_ops_fa_etc.md` — this file (paste as pull-request body)
- `tests/kernels/` — pytest including `test_flash_attn_gfx120x.py` + rdna4 suite
- `tests/python/examples/aot_gfx120x_example.py` — ahead-of-time exports

---

## Measured dispatch defaults (R9700 / gfx120x autotune tables)

Shared table: `kernels/common/gfx120x_autotune_tables.py`.

| Constant / picker | Measured value | Meaning |
|-------------------|----------------|---------|
| `DEFAULT_K_IU8_MIN` | **256** | Int8 auto: prefer W8A16 when `K < 256`; else iu8 (keeps large shapes WIN versus HIP) |
| `DEFAULT_K_FUSED_KERNEL_MAX` | **128** | Fused int8+LoRA: in-kernel fused quant+mm when `K ≤ 128`; else device rowwise quantize + iu8 |
| FlashAttention `_pick_block_m` | self: `q≤128→64` else `128`; cross: `q≤96→16`, `q≤128→32`, else `128` | Soft-gap block-M policy reconfirmed |
| `pick_adaln_block_threads` | N→block_threads map from idle versus HIP | Wired into AdaLN host |
| `pick_rms_rope_block_threads` | **HD ≤ 1024 → 64**; **HD ≥ 2048 → 512** | Re-measured with correct HIP `q_scale` / `k_scale` signature |
| `pick_asym_w4a8_block_threads` | **128** | Dequant block-threads won across K in {128…4096} |
| AWQ `pick_awq_gemv_tiles` | M=1: `n_tile` 1/4; M in (1,4]: `n_tile=1`, block_threads=64 | After mid-M=4 claw |
| ConvRot default | `linear_dtype="int4"` | Native iu4; never AWQ for this default |
| FP8 tile picker | `rdna4_scaled_mm_fp8.pick_tile_config` | Wanish `[1024,5120,5120]`: no alternate `force_path` beat `auto_default` on this pass (PARITY stance versus prior idle HIP baseline compare kept) |

**Families with no multi-path host gate (honest N/A):** SwiGLU uses compile-time `BLOCK=256`; fused multilayer perceptron uses fixed 16×16 wave-matrix tiles; stochastic FP8 defaults `BLOCK=256` without a multi-block-threads host gate.

**RoPE (native FlyDSL pick):** `pick_rope_block_threads(n_pairs_total)` lives in `kernels/common/gfx120x_autotune_tables.py` and `kernels/norm/rope_gfx120x.py`. Thresholds: `n_pairs_total < 4096 → 256`, `< 32768 → 512`, else `1024`. Call `build_rope_module(..., n_pairs_total=n)` (or pass an explicit `block=`). Comfy kit 14 `_pick_block` is a thin import of the FlyDSL picker.

---

## Correctness and idle Speed-versus-HIP (this leftover-fix pass)

**FlashAttention HIT matrix (reconfirmed on branch HEAD after leftover fixes):**

- pytest `tests/kernels/test_flash_attn_gfx120x.py`: **45 passed / 45 collected**
- Comfy-style harness: **HIT=27, FAIL=0** covering bfloat16, float16, float8_e4m3fn, float8_e5m2, int8, native iu4; self / cross / causal; `attn_mask` via host and interface
- Interface fix: `attn_mask` alias on `flydsl_flash_attn_func` (merges into bias) — previously raised unexpected keyword on the interface path

**Focused pytest:** FlashAttention + native iu4 + fused multilayer perceptron: 69 passed. ConvRot / asymmetric W4A8 / int8 / float8 scaled_mm / dispatch: 91 passed, 13 skipped.

**Idle / smoke highlights (HIP in a separate process where applicable):**

| Family | Result |
|--------|--------|
| `int8_linear_auto` tiny / mid / large | **WIN** (large ≈ ×1.41 iu8) |
| Forced W8A16 at `[1024, 4096, 4096]` | **LOSE ≈ ×0.58** — **expected**; auto must not pick W8A16 here |
| Fused int8 + host LoRA grid | **WIN** (including `(64,256,256)` ≈ ×1.12–1.16) |
| `scaled_mm_fp8_auto` tiny / mid | **WIN**; wanish prior idle **PARITY ×1.00** (tile claw found no better picker path) |
| AWQ W4A16 grid | **WIN** (≈ ×1.17–2.54) |
| ConvRot native int4 versus unpack→int8 | int4 **wins** all smoke shapes (≈ ×1.9–2.1 on leftover smoke; earlier full-stack saw ≈ ×3–4) |
| RMS-RoPE block-threads versus HIP | HIP probe fixed (`q_scale`); best block-threads WIN/PARITY ≈ ×1.10–3.23 |

**Remaining LOSE to flag (still shipping):** forced W8A16 at large MNK only. No silent DROP of ops.

---

## Design choices (why), with sources

| Choice | Why | Sources |
|--------|-----|---------|
| Size-dispatch / `*_auto` defaults (gfx120x-only, R9700-measured) | Measured W8A16↔iu8 and FP8 tile crossovers on one R9700; arch-gated; not a portable product autotune API | idle doc; `gfx120x_autotune_tables.py`; `gfx120x_arch.py`; `docs/autotune_guide.md` |
| Native iu4 default (nibble-packed int8, i32 wave-matrix fragments) | gfx120x iu4 atom exists; unpack→iu8 wastes bandwidth/ALU | `docs/kernel_authoring_guide.md` (RDNA4 wave-matrix); `rdna4_iu4_gemm` |
| FlyDSL FlashAttention for bfloat16, float16, float8_e4m3fn, float8_e5m2, int8, iu4 | Product must not depend on flash-attn/sage; wave32 wave-matrix + 64 KB LDS | FlashAttention [arXiv:2205.14135](https://arxiv.org/abs/2205.14135); `docs/architecture_guide.md` (gfx1201); existing gfx950 FlashAttention modules |
| Host bfloat16 LoRA residual on fused int8/FP8 | Avoid in-kernel LoRA JIT blowup; keep act-quant+mm fused; **unbounded N** adapters | fused module notes; idle tables |
| No `from __future__ import annotations` on ship files | Mid-file future-imports raise SyntaxError; FlyDSL `@fx.struct` / array sizes need live types | scrub on this branch (19 → 0) |
| Suggest FlyDSL minor+ version bump | New atoms, FlashAttention dtypes, defaults, docs surface | release judgment for maintainers |

---

## Checklist for maintainers

- [x] Unit tests under `tests/kernels/` (pytest; FlashAttention + rdna4)
- [x] FlashAttention HIT matrix reconfirmed (45/45 pytest; Comfy-style 27/27)
- [x] Ahead-of-time export example + export smoke
- [x] Idle Speed-versus-HIP tables updated with family walls
- [x] Prebuilt guide call sections for gfx120x paths
- [x] `from __future__ import annotations` scrubbed on ship diff
- [x] DCO Signed-off-by: dimitri91209 \<dimitri.spencer912@gmail.com\>
- [ ] Black / Ruff / `scripts/run_tests.sh` full suite before merge (maintainer CI)
- [ ] Version bump decision

---

## Changelog bullets (recent)

* **Arch gate:** `kernels/common/gfx120x_arch.py` (`require_gfx120x` / `is_gfx120x`) on public gfx120x hosts; other arches never enter size-dispatch / fused / quant launch paths.
* **Full re-autotune sanity:** PASS=83 FAIL=0 SKIP=1 (forced W8A16 large LOSE documented only).
* **RoPE block pick native:** `pick_rope_block_threads(n_pairs_total)` in FlyDSL; Comfy kit 14 `_pick_block` is a thin wrapper.
* **Future-annotations scrub:** removed all 19 ship-diff `from __future__ import annotations` lines (including mid-file ConvRot); W8A16 NOTE kept, rephrased so grep stays empty.
* **FlashAttention:** `attn_mask` interface alias; HIT matrix **45/45** pytest and Comfy-style **27/27** across bfloat16 / float16 / float8_e4m3fn / float8_e5m2 / int8 / native iu4.
* **RMS-RoPE:** HIP compare uses `q_scale` / `k_scale`; `pick_rms_rope_block_threads` → HD≤1024→64, HD≥2048→512.
* **Asymmetric W4A8:** `pick_asym_w4a8_block_threads` → 128 wired.
* **FP8 wanish:** tile alternatives did not beat `auto_default`; picker unchanged (PARITY stance retained).
* **Fused int8 + host LoRA:** `DEFAULT_K_FUSED_KERNEL_MAX=128`; unbounded N LoRA; `(64,256,256)` stable WIN.
* **FLYDSL_DISPATCH_MODE** on int8 auto/select, FP8 auto, fused int8, ConvRot.
* **Ahead-of-time exports:** `aot_gfx120x_example.py --all` exports bfloat16/float8/int8/iu4 FlashAttention, iu4 general matrix multiply, AWQ, SVDQuant, ConvRot.

---

*End of patch notes. Open the GitHub pull request only when explicitly approved; until then this file is the commit-message body and Desktop kit 13 `PR_BODY.md` source.*
