#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Host tests for gfx1201 Path A/B MNK size gate (CPU-safe select_path)."""

import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.gemm.rdna4_int8_ab_gate import (  # noqa: E402
    DEFAULT_M_A_MAX,
    DEFAULT_MK_A_MAX,
    gate_rule_doc,
    select_path,
)


@pytest.mark.parametrize(
    "m,n,k,expect",
    [
        (32, 128, 64, "A"),  # M <= 128
        (128, 256, 256, "A"),  # M == m_a_max
        (256, 512, 512, "A"),  # M*K = 131072 <= 500000
        (1024, 4096, 4096, "B"),  # M*K = 4194304
        (4096, 3072, 3072, "B"),  # wanish
    ],
)
def test_select_path_smoke_anchors(m, n, k, expect):
    assert select_path(m, n, k) == expect


def test_select_path_force():
    assert select_path(4096, 3072, 3072, force_path="A") == "A"
    assert select_path(32, 64, 64, force_path="B") == "B"
    with pytest.raises(ValueError):
        select_path(32, 64, 64, force_path="HIP")  # tip surface is A|B only


def test_gate_rule_doc_defaults():
    doc = gate_rule_doc()
    assert doc["m_a_max"] == DEFAULT_M_A_MAX == 128
    assert doc["mk_a_max"] == DEFAULT_MK_A_MAX == 500_000
    assert "tiny_expect_A" in doc["smoke_anchor"]
