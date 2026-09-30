# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Host-side size dispatcher for gfx120x int8 linear (DEFAULT entry rule).

EXPERIMENTAL: this size rule is the default way to choose between the two
int8 pipelines on gfx120x. Cutoffs M_star=128 and MK_star=500000 come from
idle Speed vs HIP crossovers on an R9700 — not a full autotune search. See
docs/gfx120x_idle_speed_vs_hip.md for the math and tables.

W8A16 linear (rdna4_w8a16_linear)
  Activations stay bf16/fp16. Int8 / FP8 e4m3fn / e5m2 weights cast to bf16
  in VGPRs, then float WMMA. Faster on small and mid shapes; loses on large.

iu8 int8 linear (rdna4_int8_linear)
  Activations quantized to int8. Int8xint8 WMMA via the gfx120x iu8 atom.
  Wins on large shapes where W8A16 loses.

Rule: if force_kernel set -> that kernel; else if M <= 128 -> W8A16;
else if M*K <= 500000 -> W8A16; else -> iu8.

Call int8_linear_auto / int8_linear_dispatched for the default path.
Credit: dimitri91209 + Grokbot.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

KernelChoice = Literal["w8a16", "iu8"]

DEFAULT_M_W8A16_MAX = 128
DEFAULT_MK_W8A16_MAX = 500_000
DEFAULT_M_A_MAX = DEFAULT_M_W8A16_MAX  # alias during rename
DEFAULT_MK_A_MAX = DEFAULT_MK_W8A16_MAX


def select_int8_kernel(
    M: int,
    N: int,
    K: int,
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    force_kernel: Optional[KernelChoice] = None,
) -> KernelChoice:
    """Cheap static rule: ``"w8a16" | "iu8"``. ``N`` unused by size selection."""
    del N  # API symmetry with MNK docs
    if force_kernel is not None:
        if force_kernel not in ("w8a16", "iu8"):
            raise ValueError(f"force_kernel must be 'w8a16' or 'iu8', got {force_kernel!r}")
        return force_kernel
    m = int(M)
    k = int(K)
    if m <= int(m_a_max):
        return "w8a16"
    if m * k <= int(mk_a_max):
        return "w8a16"
    return "iu8"


def select_int8_kernel_for_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    force_kernel: Optional[KernelChoice] = None,
) -> KernelChoice:
    """M from flattened x rows; N/K from weight [N,K]."""
    x2d = x.reshape(-1, x.shape[-1])
    return select_int8_kernel(
        int(x2d.shape[0]),
        int(weight.shape[0]),
        int(x2d.shape[1]),
        m_a_max=m_a_max,
        mk_a_max=mk_a_max,
        force_kernel=force_kernel,
    )


def _load_iu8_int8_linear():
    """Resolve ``rdna4_int8_linear`` for iu8 int8 linear on device.

    The W8A16-only tree lacks this module; the iu8 tree ships it. A plain
    ``from kernels.gemm.rdna4_int8_linear import ...`` fails when the W8A16 tree's
    ``kernels`` package won ``sys.path`` first (both trees on path). Load the
    local sibling of *this* file, else search ``sys.path`` for the file.
    """
    import importlib.util
    import sys
    from pathlib import Path as _Path

    # Fast path: already importable from the active kernels package (iu8 tree).
    try:
        from kernels.gemm import rdna4_int8_linear as mod  # type: ignore

        if hasattr(mod, "create_wmma_int8_linear_module") and hasattr(mod, "pick_tile_config"):
            return mod
    except ImportError:
        pass

    candidates = []
    here = _Path(__file__).resolve().parent / "rdna4_int8_linear.py"
    candidates.append(here)
    for entry in sys.path:
        if not entry:
            continue
        p = _Path(entry) / "kernels" / "gemm" / "rdna4_int8_linear.py"
        if p not in candidates:
            candidates.append(p)

    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ImportError(
            "iu8 int8 linear requires kernels.gemm.rdna4_int8_linear "
            "(gfx120x iu8 WMMA atom); not found on "
            "sys.path"
        )

    name = "kernels.gemm._rdna4_int8_linear_iu8"
    existing = sys.modules.get(name)
    if existing is not None and _Path(getattr(existing, "__file__", "")).resolve() == path.resolve():
        return existing

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load iu8 int8 linear module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_iu8_int8_linear(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """iu8 int8 linear via ``rdna4_int8_linear`` raw-pointer launcher."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.tensor_shim import _run_compiled

    _pb = _load_iu8_int8_linear()
    create_wmma_int8_linear_module = _pb.create_wmma_int8_linear_module
    pick_tile_config = _pb.pick_tile_config

    if a_int8.dtype != torch.int8 or b_int8.dtype != torch.int8:
        raise ValueError("iu8 int8 linear requires int8 A/B")
    a = a_int8.contiguous()
    b = b_int8.contiguous()
    if a.data_ptr() % 16:
        a = a.clone()
    if b.data_ptr() % 16:
        b = b.clone()
    m, k = a.shape
    n = b.shape[0]
    if b.shape[1] != k:
        raise ValueError(f"inner dim mismatch: a K={k}, b K={b.shape[1]}")
    if k % 16 != 0:
        raise ValueError(f"K must be divisible by 16, got {k}")

    x_scale = x_scale.to(device=a.device, dtype=torch.float32).reshape(-1).contiguous()
    w_scale = w_scale.to(device=a.device, dtype=torch.float32).reshape(-1).contiguous()
    if x_scale.numel() != m:
        raise ValueError(f"x_scale must be [M]={m}, got {x_scale.numel()}")
    w_per_n = w_scale.numel() != 1
    if w_per_n and w_scale.numel() != n:
        raise ValueError(f"w_scale must be scalar or [N]={n}, got {w_scale.numel()}")

    store_dtype = torch.float32 if bias is not None else out_dtype
    store_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[store_dtype]
    out = torch.empty((m, n), device=a.device, dtype=store_dtype)
    cfg = pick_tile_config(m, n, k)
    launch = create_wmma_int8_linear_module(
        store_name,
        cfg,
        skip_bounds=(m % cfg.bm == 0 and n % cfg.bn == 0),
        w_scale_per_n=w_per_n,
    )

    def _ptr(t: torch.Tensor):
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    _run_compiled(
        launch,
        _ptr(a),
        _ptr(b),
        _ptr(out),
        _ptr(x_scale),
        _ptr(w_scale),
        int(m),
        int(n),
        int(k),
        torch.cuda.current_stream(device=a.device),
    )
    if bias is not None:
        bias = bias.to(device=out.device, dtype=torch.float32).reshape(-1)
        if bias.numel() != n:
            raise ValueError(f"bias must be [N]={n}")
        out = (out + bias).to(out_dtype)
    elif store_dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def dispatch_rule_doc(
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    m_w8a16_max: int | None = None,
    mk_w8a16_max: int | None = None,
) -> dict:
    """Structured description of the active size dispatcher."""
    if m_w8a16_max is not None:
        m_a_max = m_w8a16_max
    if mk_w8a16_max is not None:
        mk_a_max = mk_w8a16_max
    return {
        "m_a_max": m_a_max,
        "mk_a_max": mk_a_max,
        "m_w8a16_max": m_a_max,
        "mk_w8a16_max": mk_a_max,
        "default_m_a_max": DEFAULT_M_A_MAX,
        "default_mk_a_max": DEFAULT_MK_A_MAX,
        "rule": (f"force_kernel if set; else w8a16 if M <= {m_a_max}; " f"else w8a16 if M*K <= {mk_a_max}; else iu8"),
        "w8a16": "kernels.gemm.rdna4_w8a16_linear — int8 weights→bf16, bf16 acts→bf16 out",
        "iu8": "kernels.gemm.rdna4_int8_linear — int8 acts × int8 weights → bf16 out",
        "smoke_anchor": {
            "tiny": {"M": 32, "N": 64, "K": 128, "expect": "w8a16"},
            "mid": {"M": 256, "N": 512, "K": 512, "expect": "w8a16"},
            "large": {"M": 1024, "N": 4096, "K": 4096, "expect": "iu8"},
            "wanish": {"M": 4096, "N": 3072, "K": 3072, "expect": "iu8"},
        },
        "disclaimer": ("Experimental size rule measured on gfx120x (R9700). " "Not a general FlyDSL autotune API."),
    }


gate_rule_doc = dispatch_rule_doc


def int8_linear_dispatched(
    path: KernelChoice,
    *,
    a_int8: Optional[torch.Tensor] = None,
    b_int8: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    a_act: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dispatch to W8A16 linear or B by ``path``. Caller supplies matching tensors.

    Path ``"iu8"``: ``a_int8``, ``b_int8``, ``x_scale``, ``w_scale`` (needs iu8 WMMA).
    Path ``"w8a16"``: ``a_act`` (bf16/fp16), ``b_int8``, ``w_scale``.
    """
    if path == "iu8":
        if a_int8 is None or b_int8 is None or x_scale is None or w_scale is None:
            raise ValueError("iu8 int8 linear requires a_int8, b_int8, x_scale, w_scale")
        try:
            return _run_iu8_int8_linear(a_int8, b_int8, x_scale, w_scale, bias, out_dtype)
        except ImportError as e:
            raise ImportError(
                "iu8 int8 linear requires kernels.gemm.rdna4_int8_linear "
                "(gfx120x-iu8-int8 / iu8 WMMA atom). " + str(e)
            ) from e
    if path == "w8a16":
        if a_act is None or b_int8 is None or w_scale is None:
            raise ValueError("W8A16 linear requires a_act, b_int8, w_scale")
        from kernels.gemm.rdna4_w8a16_linear import w8a16_gemm

        return w8a16_gemm(a_act, b_int8, w_scale, bias=bias, out_dtype=out_dtype)
    raise ValueError(f"unknown path {path!r}")


def int8_linear_auto(
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
    force_kernel: Optional[KernelChoice] = None,
) -> torch.Tensor:
    """Select path from MNK then dispatch.

    W8A16 linear uses ``x`` (bf16/fp16) directly. iu8 int8 linear needs pre-quantized ``a_int8``
    + ``x_scale`` (from ``quantize_int8_rowwise``); if omitted, raises with guidance.
    """
    if out_dtype is None:
        out_dtype = x.dtype
    path = select_int8_kernel_for_tensors(x, weight, m_a_max=m_a_max, mk_a_max=mk_a_max, force_kernel=force_kernel)
    if path == "w8a16":
        return int8_linear_dispatched(
            "w8a16",
            a_act=x if x.ndim == 2 else x.reshape(-1, x.shape[-1]),
            b_int8=weight,
            w_scale=weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )
    if a_int8 is None or x_scale is None:
        raise ValueError(
            "iu8 int8 linear selected but a_int8/x_scale missing — run "
            "kernels.quant.rdna4_quantize_int8_rowwise.quantize_int8_rowwise(x) "
            "first, or force_kernel='A'"
        )
    return int8_linear_dispatched(
        "iu8",
        a_int8=a_int8,
        b_int8=weight,
        x_scale=x_scale,
        w_scale=weight_scale,
        bias=bias,
        out_dtype=out_dtype,
    )


__all__ = [
    "DEFAULT_M_A_MAX",
    "DEFAULT_MK_A_MAX",
    "KernelChoice",
    "select_int8_kernel",
    "select_int8_kernel_for_tensors",
    "gate_rule_doc",
    "int8_linear_dispatched",
    "int8_linear_auto",
]
