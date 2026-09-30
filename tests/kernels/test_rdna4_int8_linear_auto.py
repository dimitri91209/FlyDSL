#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""CPU-safe tests for the gfx120x int8_linear_auto size rule."""

import os
import sys

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.gemm.rdna4_int8_linear_auto import (  # noqa: E402
    DEFAULT_M_A_MAX,
    DEFAULT_MK_A_MAX,
    gate_rule_doc,
    select_int8_kernel,
)


@pytest.mark.parametrize(
    "m,n,k,expect",
    [
        (32, 128, 64, "w8a16"),
        (128, 256, 256, "w8a16"),
        (256, 512, 512, "w8a16"),
        (1024, 4096, 4096, "iu8"),
        (4096, 3072, 3072, "iu8"),
    ],
)
def test_select_int8_kernel_anchors(m, n, k, expect):
    assert select_int8_kernel(m, n, k) == expect


def test_select_int8_kernel_force():
    assert select_int8_kernel(4096, 3072, 3072, force_kernel="w8a16") == "w8a16"
    assert select_int8_kernel(32, 64, 64, force_kernel="iu8") == "iu8"
    with pytest.raises(ValueError):
        select_int8_kernel(32, 64, 64, force_kernel="HIP")  # type: ignore[arg-type]


def test_auto_rule_doc():
    doc = gate_rule_doc()
    assert doc.get("m_a_max", doc.get("m_w8a16_max")) == DEFAULT_M_A_MAX == 128
    assert doc.get("mk_a_max", doc.get("mk_w8a16_max")) == DEFAULT_MK_A_MAX == 500_000
    assert (
        doc.get("auto_module") == "kernels.gemm.rdna4_int8_linear_auto"
        or doc.get("gated_module") == "kernels.gemm.rdna4_int8_linear_auto"
    )
    assert (
        "Experimental" in doc.get("disclaimer", "")
        or "experimental" in doc.get("disclaimer", "").lower()
        or "EXPERIMENTAL" in doc.get("disclaimer", "")
    )
