# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""FlyDSL-native helpers for the gfx1201 norm / RoPE / AdaLN kernels.

Provides buffer-pointer conversion, kernel-name specialization suffixes, and
small copy atoms shared by RoPE, RMS-RoPE, and AdaLN. Upstream aiter sources
use a private pointer/buffer shim; this module binds only the helpers these
kernels need to FlyDSL's public ``kernels.common`` surface so the PR has no
aiter import dependency.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.compiler.protocol import extract_to_ir_values
import flydsl.expr as fx
from flydsl.expr import ptrtoint
from flydsl.expr.typing import T, Vector as Vec


# Nominal extent for a raw pointer view; the descriptor bound is the real OOB guard.
BUF_VIEW_MAX_ELEMS = 0xFFFFFFFF


def kernel_signature(**params: object) -> str:
    """Render specialization parameters into a legal kernel-name suffix."""
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}"
        for name, value in params.items()
    ).replace("-", "_")


def buf_base_i64(base):
    """Return the byte address of a pointer or tensor-like kernel argument."""
    raw = extract_to_ir_values(base)[0]
    if str(raw.type).startswith(("!fly.ptr", "!llvm.ptr")):
        return fx.Int64(ptrtoint(base))
    if isinstance(raw.type, (ir.IntegerType, ir.IndexType)):
        return fx.Int64(base)
    aligned = fly.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr<1>"), raw)
    return fx.Int64(llvm.PtrToIntOp(T.i64, aligned).result)


def ptr_buf_tensor(
    ptr,
    elem=fx.Int32,
    n=BUF_VIEW_MAX_ELEMS,
    unit_elems=1,
    num_records_bytes=None,
    unit_stride=None,
):
    """Create a buffer-resource view over an ``fx.Pointer`` argument."""
    unit_stride = unit_elems if unit_stride is None else unit_stride
    layout = (
        fx.make_layout((n,), (1,))
        if unit_elems == 1
        else fx.make_layout((n, unit_elems), (unit_stride, 1))
    )
    ptr_type = fx.PointerType.get(
        elem.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=unit_stride * (elem.width // 8),
    )
    view = fx.make_view(fx.inttoptr(ptr_type, buf_base_i64(ptr)), layout)
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


_BUF_COPY_ATOM = {
    16: fx.rocdl.BufferCopy128b,
    8: fx.rocdl.BufferCopy64b,
    4: fx.rocdl.BufferCopy32b,
    2: fx.rocdl.BufferCopy16b,
    1: fx.rocdl.BufferCopy8b,
}


def _buf_copy_slice(buffer, index, unit_elems):
    if unit_elems == 1:
        grouped = fx.logical_divide(buffer, fx.make_layout(1, 1))
        return fx.slice(grouped, (None, index))
    return fx.slice(buffer, (index, None))


def buf_copy_load(buffer, index, elem=fx.Int32, unit_elems=1, cache_modifier=0):
    """Load one scalar/vector unit through FlyDSL's copy API."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    atom = fx.make_copy_atom(
        _BUF_COPY_ATOM[unit_elems * (elem.width // 8)](cache_modifier), elem
    )
    fx.copy(atom, _buf_copy_slice(buffer, index, unit_elems), fragment)
    value = Vec(fragment.load())
    return value[0] if unit_elems == 1 else value


def buf_copy_store(buffer, index, value, elem=fx.Int32, unit_elems=1, cache_modifier=0):
    """Store one scalar/vector unit through FlyDSL's copy API."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fragment.store(Vec.from_elements([value], elem) if unit_elems == 1 else Vec(value))
    atom = fx.make_copy_atom(
        _BUF_COPY_ATOM[unit_elems * (elem.width // 8)](cache_modifier), elem
    )
    fx.copy(atom, fragment, _buf_copy_slice(buffer, index, unit_elems))
