"""
Flash Attention v2 — Forward Pass Triton Kernel
================================================
Algorithm (per query block):

  for each block of i rows in Q  (BLOCK_M rows):
    load Q_i  from HBM → SRAM                          [stays in SRAM whole loop]
    init: m_i = -inf (running max),  l_i = 0 (running sum),  O_i = 0

    for each block of j cols in K,V  (BLOCK_N cols):
      load K_j, V_j from HBM → SRAM
      S_ij = Q_i @ K_j^T  * scale                      [SRAM matmul]

      m_ij  = rowmax(S_ij)
      new_m = max(m_i, m_ij)                            [online softmax update]

      P_ij  = exp(S_ij - new_m)                         [re-centred scores]
      l_i   = exp(m_i - new_m) * l_i  +  rowsum(P_ij)  [rescale old sum]
      O_i   = exp(m_i - new_m) * O_i  +  P_ij @ V_j    [rescale old acc]

      m_i   = new_m

    O_i /= l_i                                          [final normalisation]
    write O_i back to HBM

Memory layout expected (all row-major / contiguous):
  Q  : (batch, heads, seq_len_q,  head_dim)
  K  : (batch, heads, seq_len_kv, head_dim)
  V  : (batch, heads, seq_len_kv, head_dim)
  O  : (batch, heads, seq_len_q,  head_dim)   output
  L  : (batch, heads, seq_len_q)              logsumexp, needed for backward
"""

import triton
import triton.language as tl
import torch


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _flash_attn_fwd_kernel(
    # ── tensor pointers ──────────────────────────────────────────────────
    Q_ptr, K_ptr, V_ptr,        # inputs
    O_ptr,                      # output
    L_ptr,                      # logsumexp  (batch, heads, seq_q)
    # ── strides  [row-major: stride = elements to skip per unit step] ────
    stride_qb, stride_qh, stride_qm, stride_qk,   # Q  strides
    stride_kb, stride_kh, stride_kn, stride_kk,   # K  strides
    stride_vb, stride_vh, stride_vn, stride_vk,   # V  strides
    stride_ob, stride_oh, stride_om, stride_ok,   # O  strides
    # ── problem shape ────────────────────────────────────────────────────
    seq_len_q: tl.constexpr,
    seq_len_kv: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,                      # 1 / sqrt(head_dim)
    # ── tile sizes (compile-time constants for SRAM allocation) ──────────
    BLOCK_M: tl.constexpr,      # rows of Q per CTA
    BLOCK_N: tl.constexpr,      # cols of K/V per inner-loop step
):
    """
    Grid: (ceil(seq_len_q / BLOCK_M),  batch * heads)

    Each program handles:
      - one tile of BLOCK_M query rows
      - one (batch, head) pair
    """
    # ── which tile of Q this program owns ────────────────────────────────
    tile_m  = tl.program_id(0)           # which Q-row block
    bh_idx  = tl.program_id(1)           # flattened (batch, head) index
    batch   = bh_idx // tl.num_programs(1)   # not needed for pointer math below
    # we encode both batch & head into the base pointer offset directly

    # ── base pointer for this (batch, head) ──────────────────────────────
    # tl.program_id(1) == batch_idx * num_heads + head_idx
    # use stride_qb and stride_qh split properly if needed.
    # Here we treat bh_idx as a single "batch*head" index with combined stride.
    # Assumes contiguous layout so stride_qh steps one full head.
    # (In the wrapper we'll pass stride_qb = num_heads*seq*d, stride_qh = seq*d)

    q_bh_offset = bh_idx * stride_qh          # jump to (batch,head) block in Q
    k_bh_offset = bh_idx * stride_kh
    v_bh_offset = bh_idx * stride_vh
    o_bh_offset = bh_idx * stride_oh

    # row indices this program is responsible for  [BLOCK_M values]
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_k = tl.arange(0, HEAD_DIM)                       # [HEAD_DIM]

    # ── load Q tile (stays resident in SRAM for the whole K/V loop) ──────
    # mask out-of-bounds rows
    q_mask  = offs_m[:, None] < seq_len_q                # [BLOCK_M, 1]  bool
    Q_tile  = tl.load(
        Q_ptr + q_bh_offset
               + offs_m[:, None] * stride_qm             # [BLOCK_M, HEAD_DIM]
               + offs_k[None, :] * stride_qk,
        mask=q_mask,
        other=0.0,
    )  # shape: [BLOCK_M, HEAD_DIM]

    # ── running accumulators (in registers) ──────────────────────────────
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)  # rowmax
    l_i = tl.zeros([BLOCK_M],                    dtype=tl.float32)  # rowsum
    O_i = tl.zeros([BLOCK_M, HEAD_DIM],          dtype=tl.float32)  # output acc

    # ── inner loop over K / V blocks ─────────────────────────────────────
    num_blocks_n = tl.cdiv(seq_len_kv, BLOCK_N)

    for j in range(num_blocks_n):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)    # [BLOCK_N]
        kv_mask = offs_n[None, :] < seq_len_kv           # [1, BLOCK_N]

        # load K_j  [HEAD_DIM, BLOCK_N]  (transposed for the matmul below)
        K_tile = tl.load(
            K_ptr + k_bh_offset
                   + offs_n[None, :] * stride_kn         # [1, BLOCK_N]
                   + offs_k[:, None] * stride_kk,        # [HEAD_DIM, 1]
            mask=kv_mask,
            other=0.0,
        )  # shape: [HEAD_DIM, BLOCK_N]  — pre-transposed

        # load V_j  [BLOCK_N, HEAD_DIM]
        V_tile = tl.load(
            V_ptr + v_bh_offset
                   + offs_n[:, None] * stride_vn
                   + offs_k[None, :] * stride_vk,
            mask=kv_mask[:, None] if kv_mask.shape == (1, BLOCK_N) else tl.broadcast_to(kv_mask.T, [BLOCK_N, HEAD_DIM]),
            other=0.0,
        )  # shape: [BLOCK_N, HEAD_DIM]

        # ── S = Q_i @ K_j^T  * scale  ────────────────────────────────────
        # Q_tile : [BLOCK_M, HEAD_DIM]
        # K_tile : [HEAD_DIM, BLOCK_N]  (already transposed at load time)
        S = tl.dot(Q_tile, K_tile) * scale              # [BLOCK_M, BLOCK_N]

        # mask padding columns so they don't affect softmax
        S = tl.where(kv_mask, S, float("-inf"))

        # ── online softmax update ─────────────────────────────────────────
        m_ij   = tl.max(S, axis=1)                      # [BLOCK_M]  block rowmax
        new_m  = tl.maximum(m_i, m_ij)                  # [BLOCK_M]

        # rescale factors
        alpha  = tl.exp(m_i  - new_m)                   # [BLOCK_M]  old scale
        beta   = tl.exp(m_ij - new_m)                   # [BLOCK_M]  new scale

        P      = tl.exp(S - new_m[:, None])             # [BLOCK_M, BLOCK_N]

        l_i    = alpha * l_i + tl.sum(P, axis=1)        # [BLOCK_M]
        O_i    = alpha[:, None] * O_i + tl.dot(P.to(tl.float16), V_tile)

        m_i    = new_m

    # ── final normalisation ───────────────────────────────────────────────
    O_i = O_i / l_i[:, None]

    # ── write output ──────────────────────────────────────────────────────
    tl.store(
        O_ptr + o_bh_offset
               + offs_m[:, None] * stride_om
               + offs_k[None, :] * stride_ok,
        O_i.to(tl.float16),
        mask=q_mask,
    )

    # store log-sum-exp  L = m + log(l)  for backward pass
    L_row_ptr = L_ptr + bh_idx * seq_len_q + offs_m
    tl.store(L_row_ptr, m_i + tl.log(l_i), mask=offs_m < seq_len_q)
