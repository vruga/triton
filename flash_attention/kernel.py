"""
Flash Attention forward — single-file Triton kernel.
Everything you need to run it is in kernel.py + run.py.
"""

import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd(
    # ── pointers ─────────────────────────────────────────────────────────
    Q_ptr, K_ptr, V_ptr,
    O_ptr,
    L_ptr,                   # logsumexp, saved for backward
    # ── strides (elements to advance per unit step in each dimension) ────
    #    for a (B, H, S, D) contiguous tensor:
    #      stride_b = H*S*D,  stride_h = S*D,  stride_s = D,  stride_d = 1
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    # ── shapes ───────────────────────────────────────────────────────────
    SEQ_Q:  tl.constexpr,
    SEQ_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,                   # 1/sqrt(head_dim), passed as a runtime scalar
    # ── tile sizes (must be powers of 2, known at compile time) ─────────
    BLOCK_M: tl.constexpr,   # rows of Q per program  (e.g. 128)
    BLOCK_N: tl.constexpr,   # rows of K/V per inner step  (e.g. 64)
):
    """
    Grid shape used when launching:  (ceil(SEQ_Q/BLOCK_M),  B*H)

    Program (tile_m, bh):
      tile_m  — which chunk of Q rows this program owns
      bh      — which (batch, head) pair
    """
    tile_m = tl.program_id(0)   # Q-row block index
    bh     = tl.program_id(1)   # flattened batch*head index

    # ── base offsets into this (batch, head) ────────────────────────────
    # Because we pass stride_qh = S*D and treat bh as a single index,
    # this jumps past all previous (batch, head) blocks in memory.
    q_base = bh * stride_qh
    k_base = bh * stride_kh
    v_base = bh * stride_vh
    o_base = bh * stride_oh

    # ── row / column index vectors ───────────────────────────────────────
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]  query rows
    offs_k = tl.arange(0, HEAD_DIM)                       # [HEAD_DIM] head-dim cols

    # ── load Q tile — stays in SRAM for the entire K/V loop ─────────────
    q_mask = offs_m[:, None] < SEQ_Q                      # [BLOCK_M, 1]
    Q = tl.load(
        Q_ptr + q_base
              + offs_m[:, None] * stride_qm               # row stride
              + offs_k[None, :] * stride_qk,              # col stride
        mask=q_mask,
        other=0.0,
    )  # [BLOCK_M, HEAD_DIM]  fp16

    # ── running accumulators (live in registers) ─────────────────────────
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # row maximum
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)    # row sum
    O_i = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)    # output

    # ── inner loop: iterate over K / V tiles ─────────────────────────────
    for j in range(tl.cdiv(SEQ_KV, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)    # [BLOCK_N]  kv rows
        kv_mask = offs_n < SEQ_KV                        # [BLOCK_N]  bool

        # K loaded transposed [HEAD_DIM, BLOCK_N] so we can do Q @ K directly
        K = tl.load(
            K_ptr + k_base
                  + offs_n[None, :] * stride_kn          # [1,       BLOCK_N]
                  + offs_k[:, None] * stride_kk,         # [HEAD_DIM, 1     ]
            mask=kv_mask[None, :],
            other=0.0,
        )  # [HEAD_DIM, BLOCK_N]

        V = tl.load(
            V_ptr + v_base
                  + offs_n[:, None] * stride_vn          # [BLOCK_N, 1      ]
                  + offs_k[None, :] * stride_vk,         # [1,       HEAD_DIM]
            mask=kv_mask[:, None],
            other=0.0,
        )  # [BLOCK_N, HEAD_DIM]

        # ── attention scores ─────────────────────────────────────────────
        S = tl.dot(Q, K) * scale                         # [BLOCK_M, BLOCK_N]
        S = tl.where(kv_mask[None, :], S, float("-inf")) # mask padding

        # ── online softmax (numerically stable) ──────────────────────────
        m_j   = tl.max(S, axis=1)                        # [BLOCK_M]  block max
        new_m = tl.maximum(m_i, m_j)                     # [BLOCK_M]  running max

        # rescale old accumulator before adding new block
        alpha = tl.exp(m_i - new_m)                      # [BLOCK_M]
        P     = tl.exp(S  - new_m[:, None])              # [BLOCK_M, BLOCK_N]

        l_i   = alpha * l_i + tl.sum(P, axis=1)
        O_i   = alpha[:, None] * O_i + tl.dot(P.to(tl.float16), V)
        m_i   = new_m

    # ── normalise and store output ────────────────────────────────────────
    O_i /= l_i[:, None]

    tl.store(
        O_ptr + o_base
              + offs_m[:, None] * stride_om
              + offs_k[None, :] * stride_ok,
        O_i.to(tl.float16),
        mask=q_mask,
    )

    # logsumexp = m + log(l), used by the backward pass
    tl.store(
        L_ptr + bh * SEQ_Q + offs_m,
        m_i + tl.log(l_i),
        mask=offs_m < SEQ_Q,
    )
