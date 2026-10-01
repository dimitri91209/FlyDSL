# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x / RDNA4 arch detection for host gates.

Public gfx120x-only entrypoints (size-dispatch ``*_auto``/``_dispatch``,
autotune-table picks used at launch, fused K-gates, ConvRot int4, FA hosts,
iu4/iu8 WMMA paths, etc.) must refuse on non-gfx120x. Other arches must not
enter these paths. On RDNA4 only, ``FLYDSL_DISPATCH_MODE`` may still force
flydsl/hip bypass of *size* gates — it is not a cross-arch license.

Credit: dimitri91209 + Grokbot.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["get_gcn_arch", "is_gfx120x", "require_gfx120x"]


def get_gcn_arch(device=None) -> str:
    """Return normalized GCN arch name (e.g. ``gfx1201``) or ``""`` if unknown."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return ""
    if device is None:
        if not torch.cuda.is_available():
            return ""
        device = torch.device("cuda")
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName or ""
    except Exception:  # noqa: BLE001
        return ""
    return str(arch).lower().split(":")[0]


def is_gfx120x(device=None) -> bool:
    """True when the device GCN arch is in the gfx120x family."""
    return get_gcn_arch(device).startswith("gfx120")


def require_gfx120x(device=None, *, what: str = "this gfx120x kernel") -> None:
    """Raise ``ValueError`` unless ``device`` is gfx120x (RDNA4)."""
    if is_gfx120x(device):
        return
    arch = get_gcn_arch(device) or "<unknown>"
    raise ValueError(f"{what} requires gfx120x (RDNA4), got arch={arch!r}")
