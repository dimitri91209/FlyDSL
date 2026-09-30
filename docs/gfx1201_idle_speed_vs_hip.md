# gfx1201 idle Speed vs HIP policy

**Date:** 2026-09-30 (America/Chicago)  
**GPU:** AMD Radeon AI PRO R9700 (gfx1201)  
**Evidence:** `04_lab/results/FULL_SUITE_IDLE_VS_HIP_2026-09-30.json`,
`PARITY_CONFIRM_2026-09-30.json`, and `CONVROT_LEAN_2026-09-30.json`  
**Credit:** dimitri91209 + Grokbot

## Idle smoke methodology (one at a time)

Lab and tip refresh benches must use **one single smoke per process**:

1. Pick exactly one `(kernel, shape)`.
2. Confirm GPU idle (Comfy stopped; busy ≈ 0).
3. **Warm** ≥10 untimed launches of that path; `torch.cuda.synchronize()`.
4. Time **HIP only** (CUDA/HIP events, median ≥25–40 reps); sync.
5. Time **FlyDSL only** the same way; sync.
6. Write the row (HIP µs, Fly µs, ×, verdict); **exit**.
7. Next shape = **new process** — never loop shapes inside one invocation.

Do not batch shapes or tests per shape in one smoke run (I/O lag / cross-shape
noise corrupts idle comparisons). Prefer event timers over wall clock. Log only
after both backends for that shape are measured. Dual-launch sub-100 µs paths
may also report backlog-event (see FlyDSL `docs/autotune_guide.md`).

## Multi-LoRA methodology

There is no separate bf16/fp16 multi-LoRA GEMM kernel. The quantized base GEMM
runs once; adapters remain small-rank host residuals in Comfy load order:
`sum_i scale_i * (x @ A_i) @ B_i`. This preserves N=0 plain dispatch and
unlimited N≥1 adapter ordering without dequant-merge into the base. A fused
multi-residual device epilogue is an optimization follow-up only if measured
host residuals lose idle; it is not required for the current shipped path.

## Speed vs HIP: LOSER list (inform only)

These idle parity/near-miss rows are **LOSER (inform only; still ships)**. No
kernel or operation is removed because of them; optimize further. The ratios
below are copied from existing lab JSON and are not claims of wins:

| Operation / case | HIP µs | Fly µs | × | Status |
|---|---:|---:|---:|---|
| `stochastic_rounding_fp8` tiny (full suite) | 7.68 | 7.72 | 0.995× | LOSER (inform only; still ships) |
| `stochastic_rounding_fp8` tiny (confirm) | 10.96 | 10.80 | 1.015× | LOSER (inform only; still ships) |
| `scaled_mm_fp8` tiny | 26.081 | 25.32 | 1.030× | LOSER (inform only; still ships) |
| `scaled_mm_fp8` wanish (full suite) | 1134.168 | 1132.087 | 1.002× | LOSER (inform only; still ships) |
| `scaled_mm_fp8` wanish (confirm) | 1144.568 | 1134.529 | 1.009× | LOSER (inform only; still ships) |
| `int8_linear` wanish | 343.002 | 333.762 | 1.028× | LOSER (inform only; still ships) |
| `int8_linear_convrot` 64x64x64_G64, simple event | 22.92 | 23.48 | 0.976× | LOSER (inform only; still ships) |
| `rms_rope1` small | — | — | 1.032× | LOSER (inform only; still ships) |
| `adaln` flux77_ln | 19.6 | 18.72 | 1.047× | LOSER (inform only; still ships) |

For the ConvRot row, the same lean report records backlog-event **1.471× WIN**
on 64x64x64_G64; mid and large ConvRot shapes are WIN. Hot shapes for the
other listed operations are also WIN.

The one actual earlier removal is `int8_linear_fused` (+LoRA/multi), which is
not re-added by this policy. That historical removal is distinct from the
inform-only loser list above.
