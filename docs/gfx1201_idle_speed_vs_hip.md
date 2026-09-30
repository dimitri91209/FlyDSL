# gfx1201 idle Speed vs HIP

**Hardware:** AMD Radeon AI PRO R9700 (gfx1201)  
**Date:** 2026-09-30 (America/Chicago)  
**Credit:** dimitri91209 + Grokbot

## Measurement

Idle Speed vs HIP uses FlyDSL Device timing (`do_bench`) as documented in
[`docs/autotune_guide.md`](autotune_guide.md) (Device timing contract):

- One process measures one `(kernel, case, backend)` with one tensor set.
  HIP and Fly run in separate processes; compare ratios after both results exist.
- Timer: `python/flydsl/autotune.py` `do_bench` — warmup, then ≤5 batches each
  preceded by a GPU backlog (`torch.cuda._sleep`), CUDA-event window of N
  launches, **median** of batch averages (ms→µs).
- GEMM suite defaults: **warm=10, rep=50**.
- Speedup = `HIP_µs / Fly_µs` (>1 ⇒ Fly faster).
- Verdicts: **WIN** ≥ 1.05 · **PARITY** 0.95–1.05 (near tie) · **LOSE** < 0.95
  (slower). PARITY and LOSE are distinct; both still ship.

## Results (2026-09-30)

| Op | Case | Shape | HIP µs | Fly µs | × | Verdict |
|---|---|---|---:|---:|---:|---|
| ab_gate | large | `[1024, 4096, 4096]` | 310.882 | 207.041 | 1.502 | WIN |
| ab_gate | tiny | `[32, 128, 64]` | 11.468 | 6.356 | 1.804 | WIN |
| adaln | flux77_ln | `[77, 3072]` | 14.080 | 4.436 | 3.174 | WIN |
| convrot | simple64 | `[64, 64, 64, 64]` | 12.044 | 10.624 | 1.134 | WIN |
| int8_ab_gated | large | `[1024, 4096, 4096]` | 336.278 | 213.665 | 1.574 | WIN |
| int8_ab_gated | mid | `[256, 512, 512]` | 29.392 | 18.624 | 1.578 | WIN |
| int8_ab_gated | tiny | `[32, 128, 64]` | 11.736 | 6.388 | 1.837 | WIN |
| int8_full | large | `[1024, 4096, 4096]` | 343.266 | 228.405 | 1.503 | WIN |
| int8_full | mid | `[128, 256, 512]` | 21.908 | 17.992 | 1.218 | WIN |
| int8_full | tiny | `[64, 64, 64]` | 11.548 | 10.356 | 1.115 | WIN |
| int8_full | wanish | `[1024, 5120, 5120]` | 409.823 | 422.924 | 0.969 | PARITY |
| int8_rowwise | large | `[1024, 4096]` | 34.236 | 11.784 | 2.905 | WIN |
| int8_rowwise | tiny | `[64, 256]` | 3.840 | 3.424 | 1.121 | WIN |
| rms_qk | flux77 | `[1, 77, 24, 128]` | 10.352 | 6.636 | 1.560 | WIN |
| rms_rope1 | small | `[1, 8, 4, 64]` | 4.516 | 3.744 | 1.206 | WIN |
| rope_sh_qk | flux77 | `[1, 77, 24, 128]` | 8.608 | 5.332 | 1.614 | WIN |
| scaled_mm_fp8_gated | mid | `[128, 512, 512]` | 16.708 | 6.836 | 2.444 | WIN |
| scaled_mm_fp8_gated | tiny | `[32, 128, 64]` | 9.840 | 4.924 | 1.998 | WIN |
| scaled_mm_fp8_gated | wanish | `[1024, 5120, 5120]` | 1137.267 | 286.498 | 3.970 | WIN |
| scaled_mm | mid_fat | `[128, 256, 512]` | 15.316 | 11.636 | 1.316 | WIN |
| scaled_mm | mid_suite | `[128, 512, 512]` | 16.908 | 12.680 | 1.333 | WIN |
| scaled_mm | tiny_k128 | `[32, 128, 128]` | 10.104 | 9.332 | 1.083 | WIN |
| scaled_mm | tiny_k64 | `[32, 128, 64]` | 10.024 | 9.232 | 1.086 | WIN |
| scaled_mm | wanish | `[1024, 5120, 5120]` | 1143.343 | 1139.479 | 1.003 | PARITY |
| stoch | tiny | `[64, 128]` | 3.716 | 3.436 | 1.081 | WIN |
| w8a16_path_a | large | `[1024, 4096, 4096]` | 316.538 | 551.759 | 0.574 | LOSE |
| w8a16_path_a | mid | `[256, 512, 512]` | 29.092 | 18.580 | 1.566 | WIN |
| w8a16_path_a | tiny | `[32, 64, 64]` | 11.492 | 6.308 | 1.822 | WIN |

## Notes

- **Summary:** n=28 · WIN=25 · PARITY=2 · LOSE=1.
- **`w8a16_path_a` large:** outside Path A’s intended gate. Large shapes route to
  Path B via `int8_ab_gated` (large WIN ×1.574 on this suite). Do not treat the
  Path A large LOSE as a Path A failure.
- **Ship policy:** no DROP without an explicit maintainer ask. Inform near-misses
  (PARITY / LOSE); they still ship.
- Multi-LoRA: quantized base GEMM once; adapters stay small-rank host residuals
  (`sum_i scale_i * (x @ A_i) @ B_i`). No separate bf16/fp16 multi-LoRA GEMM.
- `int8_linear_fused` is not in this PR (idle-lose historically).
