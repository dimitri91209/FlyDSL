# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""gfx120x FA extensions: sink, paged-KV gather, packed varlen, split-K combine."""

import math

import torch

def apply_attention_sink(out, q, k, sink, *, causal, attn_mask=None, sm_scale=None):
    if sink is None:
        return out
    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    s = sink.detach().float()
    if s.dim() == 1:
        s = s.view(1, H).expand(B, H)
    elif s.dim() != 2 or tuple(s.shape) != (B, H):
        raise ValueError(f"sink must be [H] or [B,H], got {tuple(sink.shape)}")
    qf, kf = q.float().transpose(1, 2), k.float().transpose(1, 2)
    m = torch.full((B, H, Sq), float("-inf"), device=q.device, dtype=torch.float32)
    lse = torch.zeros((B, H, Sq), device=q.device, dtype=torch.float32)
    for ks in range(0, Skv, 256):
        ke = min(Skv, ks + 256)
        scores = torch.matmul(qf, kf[:, :, ks:ke, :].transpose(-1, -2)) * sm_scale
        if attn_mask is not None:
            am = attn_mask[:, ks:ke] if attn_mask.dim() == 2 else attn_mask[:, :, ks:ke]
            scores = scores + am.to(scores.dtype)
        if causal:
            q_idx = torch.arange(Sq, device=q.device)[:, None]
            k_idx = torch.arange(ks, ke, device=q.device)[None, :]
            scores = scores.masked_fill(k_idx > q_idx + (Skv - Sq), float("-inf"))
        row_max = scores.amax(dim=-1)
        m_new = torch.maximum(m, row_max)
        corr = torch.exp(m - m_new)
        p = torch.exp(scores - m_new.unsqueeze(-1))
        lse = lse * corr + p.sum(dim=-1)
        m = m_new
    m_new = torch.maximum(m, s.unsqueeze(-1))
    corr = torch.exp(m - m_new)
    l_new = lse * corr + torch.exp(s.unsqueeze(-1) - m_new)
    o = out.float().transpose(1, 2)
    o = o * (lse * corr / l_new.clamp_min(1e-20)).unsqueeze(-1)
    return o.to(out.dtype).transpose(1, 2).contiguous()

def gather_paged_kv(k_cache, v_cache, block_table, seqlen_k, *, page_size, kv_cache_layout="linear"):
    if kv_cache_layout not in ("linear", "vectorized"):
        raise NotImplementedError(kv_cache_layout)
    bt = block_table
    B = bt.shape[0]
    max_sk = int(seqlen_k.max().item()) if seqlen_k.numel() else 0
    if max_sk == 0:
        Hkv, D = k_cache.shape[-2], k_cache.shape[-1]
        e = torch.empty(B, 0, Hkv, D, device=k_cache.device, dtype=k_cache.dtype)
        return e, e, 0
    if kv_cache_layout == "vectorized":
        k_cache = k_cache.permute(0, 2, 1, 3).contiguous()
        v_cache = v_cache.permute(0, 2, 1, 3).contiguous()
    _n, PageSize, Hkv, D = k_cache.shape
    if page_size != PageSize:
        raise ValueError("page_size mismatch")
    k_out = torch.zeros(B, max_sk, Hkv, D, device=k_cache.device, dtype=k_cache.dtype)
    v_out = torch.zeros(B, max_sk, Hkv, D, device=v_cache.device, dtype=v_cache.dtype)
    for b in range(B):
        sk = int(seqlen_k[b].item())
        for p in range((sk + page_size - 1) // page_size):
            pid = int(bt[b, p].item())
            t0, t1 = p * page_size, min(sk, (p + 1) * page_size)
            n = t1 - t0
            k_out[b, t0:t1] = k_cache[pid, :n]
            v_out[b, t0:t1] = v_cache[pid, :n]
    return k_out, v_out, max_sk

def packed_varlen_to_dense(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    cq, ck = cu_seqlens_q.to(torch.int64), cu_seqlens_kv.to(torch.int64)
    B = int(cq.numel() - 1)
    H, D = int(q.shape[1]), int(q.shape[2])
    q_d = torch.zeros(B, max_seqlen_q, H, D, device=q.device, dtype=q.dtype)
    k_d = torch.zeros(B, max_seqlen_kv, H, D, device=k.device, dtype=k.dtype)
    v_d = torch.zeros(B, max_seqlen_kv, H, D, device=v.device, dtype=v.dtype)
    sqv = torch.zeros(B, device=q.device, dtype=torch.int32)
    skv = torch.zeros(B, device=q.device, dtype=torch.int32)
    for b in range(B):
        qs, qe = int(cq[b]), int(cq[b + 1])
        ks, ke = int(ck[b]), int(ck[b + 1])
        sqv[b], skv[b] = qe - qs, ke - ks
        if qe > qs:
            q_d[b, : qe - qs] = q[qs:qe]
        if ke > ks:
            k_d[b, : ke - ks] = k[ks:ke]
            v_d[b, : ke - ks] = v[ks:ke]
    return q_d, k_d, v_d, sqv, skv

def dense_to_packed_varlen(o_dense, cu_seqlens_q):
    cq = cu_seqlens_q.to(torch.int64)
    B, total = int(cq.numel() - 1), int(cq[-1])
    out = torch.empty(total, o_dense.shape[2], o_dense.shape[3], device=o_dense.device, dtype=o_dense.dtype)
    for b in range(B):
        qs, qe = int(cq[b]), int(cq[b + 1])
        if qe > qs:
            out[qs:qe] = o_dense[b, : qe - qs]
    return out

def splitk_online_softmax_attn(q, k, v, *, causal=False, num_kv_splits=2, attn_mask=None, sm_scale=None):
    if num_kv_splits <= 1:
        raise ValueError("num_kv_splits > 1 required")
    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    m = torch.full((B, H, Sq), float("-inf"), device=q.device, dtype=torch.float32)
    lse = torch.zeros((B, H, Sq), device=q.device, dtype=torch.float32)
    o = torch.zeros((B, H, Sq, D), device=q.device, dtype=torch.float32)
    split = (Skv + num_kv_splits - 1) // num_kv_splits
    for si in range(num_kv_splits):
        ks, ke = si * split, min(Skv, (si + 1) * split)
        if ks >= ke:
            continue
        scores = torch.matmul(qf, kf[:, :, ks:ke].transpose(-1, -2)) * sm_scale
        if attn_mask is not None:
            am = attn_mask[:, ks:ke] if attn_mask.dim() == 2 else attn_mask
            scores = scores + am.to(scores.dtype)
        if causal:
            q_idx = torch.arange(Sq, device=q.device)[:, None]
            k_idx = torch.arange(ks, ke, device=q.device)[None, :]
            scores = scores.masked_fill(~(k_idx <= q_idx + (Skv - Sq)), float("-inf"))
        row_max = scores.amax(dim=-1)
        m_new = torch.maximum(m, row_max)
        alpha = torch.exp(m - m_new)
        p = torch.exp(scores - m_new.unsqueeze(-1))
        lse = lse * alpha + p.sum(dim=-1)
        o = o * alpha.unsqueeze(-1) + torch.matmul(p, vf[:, :, ks:ke])
        m = m_new
    return (o / lse.clamp_min(1e-20).unsqueeze(-1)).to(q.dtype).transpose(1, 2).contiguous()
