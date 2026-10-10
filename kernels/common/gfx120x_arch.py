# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""One gfx120x check for launch functions.

Uses ``get_rocm_arch``, the same process arch the gfx950 GEMMs use.
``gfx1250`` does not match.
"""

from flydsl.runtime.device import get_rocm_arch


def require_gfx120x(what: str = "this gfx120x kernel") -> None:
    """Raise unless the process arch starts with ``gfx120``."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx120"):
        raise ValueError(f"{what} requires gfx120x, got arch={arch!r}")


__all__ = ["require_gfx120x"]
