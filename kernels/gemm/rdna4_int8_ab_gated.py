# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
################################################################################
# WARNING — EXPERIMENTAL SIZE/SHAPE GATE FOR GFX1201
#
# THIS MODULE IS AN EXPERIMENTAL SIZE/SHAPE GATE MEASURED LOCALLY ON GFX1201
# (RDNA4). BREAKPOINTS WERE MEASURED IN LAB IDLE SMOKES; THIS IS NOT A GENERAL
# FLYDSL AUTOTUNE API. COMFY / PRODUCT ENV MAY STILL OWN PRODUCTION WIRING AND
# FALLBACKS. CALLERS MUST NOT TREAT THIS AS A PRODUCTION DEFAULT WITHOUT READING
# THE GATE RULES BELOW AND THE MATCHING LAB RESULTS.
#
# UNGATED SIBLINGS: kernels/gemm/rdna4_w8a16_path_a.py (Path A) AND
# kernels/gemm/rdna4_int8_linear.py (Path B, tip B) REMAIN AVAILABLE. THE RULE
# TABLE ALSO LIVES IN kernels/gemm/rdna4_int8_ab_gate.py — THIS *_gated MODULE
# IS THE OPT-IN GATED SURFACE NAME (APPEND _gated) AND IS ADDITIVE ONLY.
################################################################################
"""Gated host for gfx1201 int8 AB / W8A16 Path A vs iu8 Path B.

Wraps the established **M-floor then M×K** breakpoints (WORKER_REFERENCE.md /
playbook §11 / IMPLEMENTED_AB_GATE 2026-09-25):

  tiny  32×64×128       M≤128              → A  (w8a16_path_a)
  mid   256×512×512     M>128, M*K≤500000  → A
  large 1024×4096×4096  M*K>500000         → B  (int8_linear / iu8)
  wanish 4096×3072×3072 M*K>500000         → B

Host symbols: ``int8_ab_gated`` (auto select+dispatch), ``select_path``,
``int8_linear_gated``. Delegates to ``rdna4_int8_ab_gate`` (no rule drift).

Credit: dimitri91209+Grokbot.
"""

from __future__ import annotations

from typing import Optional

import torch

from kernels.gemm.rdna4_int8_ab_gate import (
    DEFAULT_M_A_MAX,
    DEFAULT_MK_A_MAX,
    PathName,
    gate_rule_doc as _ab_gate_rule_doc,
    int8_linear_auto,
    int8_linear_gated,
    select_path,
    select_path_for_tensors,
)

KERNEL_NAME = "rdna4_int8_ab_gated"


def gate_rule_doc(
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
) -> dict:
    """Structured gate description (+ explicit experimental disclaimer)."""
    doc = _ab_gate_rule_doc(m_a_max=m_a_max, mk_a_max=mk_a_max)
    doc["gated_module"] = "kernels.gemm.rdna4_int8_ab_gated"
    doc["rule_source"] = "kernels.gemm.rdna4_int8_ab_gate"
    doc["disclaimer"] = (
        "EXPERIMENTAL gfx1201 size/shape gate; not a general FlyDSL autotune API; "
        "Comfy may own product env; not a production default without reading gate rules"
    )
    return doc


def int8_ab_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    x_scale: Optional[torch.Tensor] = None,
    a_int8: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    force_path: Optional[PathName] = None,
) -> torch.Tensor:
    """Opt-in gated surface: M/M*K select then Path A (W8A16) or B (iu8).

    Same contract as ``rdna4_int8_ab_gate.int8_linear_auto``. Path B needs tip B
    ``rdna4_int8_linear`` (+ pre-quant ``a_int8`` / ``x_scale``).
    """
    return int8_linear_auto(
        x,
        weight,
        weight_scale,
        x_scale=x_scale,
        a_int8=a_int8,
        bias=bias,
        out_dtype=out_dtype,
        m_a_max=m_a_max,
        mk_a_max=mk_a_max,
        force_path=force_path,
    )


__all__ = [
    "KERNEL_NAME",
    "DEFAULT_M_A_MAX",
    "DEFAULT_MK_A_MAX",
    "PathName",
    "select_path",
    "select_path_for_tensors",
    "gate_rule_doc",
    "int8_linear_gated",
    "int8_ab_gated",
]
