# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Small RDNA4 buffer-copy helpers for raw-pointer kernel arguments.

Wrap a raw device pointer as a FlyDSL buffer tensor so kernels can use the
shared ``buf_copy_load`` / ``buf_copy_store`` atoms. Used by the gfx120x quant
and elementwise suite.
"""

import flydsl.expr as fx
from flydsl.expr import ptrtoint
from flydsl.expr.typing import Vector as Vec

BUF_VIEW_MAX_ELEMS = 0xFFFFFFFF


def ptr_buf_tensor(
    ptr,
    elem=fx.Int32,
    n=BUF_VIEW_MAX_ELEMS,
    unit_elems=1,
    num_records_bytes=None,
    unit_stride=None,
):
    """Make an upstream FlyDSL buffer tensor over a raw device pointer."""
    unit_stride = unit_elems if unit_stride is None else unit_stride
    layout = fx.make_layout((n,), (1,)) if unit_elems == 1 else fx.make_layout((n, unit_elems), (unit_stride, 1))
    ptr_type = fx.PointerType.get(
        elem.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=unit_stride * (elem.width // 8),
    )
    view = fx.make_view(fx.inttoptr(ptr_type, fx.Int64(ptrtoint(ptr))), layout)
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


_COPY_ATOM = {
    16: fx.rocdl.BufferCopy128b,
    8: fx.rocdl.BufferCopy64b,
    4: fx.rocdl.BufferCopy32b,
    2: fx.rocdl.BufferCopy16b,
    1: fx.rocdl.BufferCopy8b,
}


def _slice(buffer, index, unit_elems):
    if unit_elems == 1:
        grouped = fx.logical_divide(buffer, fx.make_layout(1, 1))
        return fx.slice(grouped, (None, index))
    return fx.slice(buffer, (index, None))


def _atom(unit_elems, elem):
    return fx.make_copy_atom(_COPY_ATOM[unit_elems * (elem.width // 8)](0), elem)


def buf_copy_load(buffer, index, elem=fx.Int32, unit_elems=1):
    """Load one scalar/vector unit through an AMD buffer-copy atom."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fx.copy(_atom(unit_elems, elem), _slice(buffer, index, unit_elems), fragment)
    value = Vec(fragment.load())
    return value[0] if unit_elems == 1 else value


def buf_copy_store(buffer, index, value, elem=fx.Int32, unit_elems=1):
    """Store one scalar/vector unit through an AMD buffer-copy atom."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fragment.store(Vec.from_elements([value], elem) if unit_elems == 1 else Vec(value))
    fx.copy(_atom(unit_elems, elem), fragment, _slice(buffer, index, unit_elems))
