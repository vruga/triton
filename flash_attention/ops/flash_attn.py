"""
PyTorch-facing wrapper for the Flash Attention forward kernel.

Handles:
  - input validation & contiguous layout
  - grid / block configuration from the 4090 config table
  - launching the Triton kernel
"""

import torch
from flash_attention.kernel.flash_attn_fwd import _flash_attn_fwd_kernel
from flash_attention.configs.rtx4090 import get_block_config
import math


def flash_attn_forward(
    q: torch.Tensor,   # (B, H, Sq, D)
    k: torch.Tensor,   # (B, H, Skv, D)
    v: torch.Tensor,   # (B, H, Skv, D)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    O : (B, H, Sq, D)   fp16 output
    L : (B, H, Sq)      log-sum-exp, needed for the backward pass
    """
    assert q.dtype == k.dtype == v.dtype == torch.float16, \
        "All tensors must be fp16"
    assert q.is_cuda and k.is_cuda and v.is_cuda, \
        "All tensors must be on CUDA"

    # ensure contiguous memory so our stride math is correct
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    B, H, Sq,  D  = q.shape
    _B, _H, Skv, _D = k.shape

    assert D == _D,      "Q and K/V must have the same head_dim"
    assert B == _B and H == _H

    # ── look up block sizes for this head dim ─────────────────────────────
    BLOCK_M, BLOCK_N, num_warps, num_stages = get_block_config(D)

    # ── allocate outputs ──────────────────────────────────────────────────
    O = torch.empty_like(q)
    L = torch.empty((B, H, Sq), device=q.device, dtype=torch.float32)

    # ── flatten (batch, head) into a single grid dim ──────────────────────
    # grid = (number of Q-row tiles,  B*H)
    grid = (math.ceil(Sq / BLOCK_M), B * H)

    scale = 1.0 / math.sqrt(D)

    # ── Q strides: (B, H, Sq, D)  row-major ─────────────────────────────
    stride_qb, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kb, stride_kh, stride_kn, stride_kk = k.stride()
    stride_vb, stride_vh, stride_vn, stride_vk = v.stride()
    stride_ob, stride_oh, stride_om, stride_ok = O.stride()

    _flash_attn_fwd_kernel[grid](
        q, k, v, O, L,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        seq_len_q  = Sq,
        seq_len_kv = Skv,
        HEAD_DIM   = D,
        scale      = scale,
        BLOCK_M    = BLOCK_M,
        BLOCK_N    = BLOCK_N,
        num_warps  = num_warps,
        num_stages = num_stages,
    )
    return O, L
