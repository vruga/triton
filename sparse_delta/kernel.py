import triton
import triton.language as tl


# ==============================================================================
# KERNEL 1: Block-Sparse Attention (forward)
# ==============================================================================
#
# Standard attention: compute ALL (Q-block i, KV-block j) pairs → O(N²)
# Sparse attention:   only compute pairs where block_mask[i, j] == 1
#
# block_mask shape: (num_q_blocks, num_kv_blocks)  int8 on device
# Grid: (num_q_blocks, B*H)  — one program per Q-block per (batch,head)
#
# Each program:
#   1. reads its row of block_mask to find which KV blocks to attend to
#   2. skips blocks where mask == 0 (entire block skipped, no HBM load)
#   3. runs online softmax only over active blocks
#
# Memory savings: if sparsity = 75%, only 25% of K/V blocks are ever loaded.
# ==============================================================================

@triton.jit
def sparse_attn_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr,
    L_ptr,
    block_mask_ptr,             # (num_q_blocks, num_kv_blocks)  int8
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_mask_row,            # = num_kv_blocks (stride across mask rows)
    SEQ_Q:    tl.constexpr,
    SEQ_KV:   tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_m = tl.program_id(0)   # which Q-row block
    bh     = tl.program_id(1)   # flattened (batch, head)

    q_base = bh * stride_qh
    k_base = bh * stride_kh
    v_base = bh * stride_vh
    o_base = bh * stride_oh

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    q_mask = offs_m[:, None] < SEQ_Q

    Q = tl.load(
        Q_ptr + q_base + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
        mask=q_mask, other=0.0, eviction_policy="evict_first",
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    O_i = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    num_kv_blocks = tl.cdiv(SEQ_KV, BLOCK_N)

    for j in range(num_kv_blocks):
        # ── sparse gate: skip this KV block if mask says 0 ──────────────
        # block_mask[tile_m, j]  — pointer arithmetic into the 2D mask
        active = tl.load(block_mask_ptr + tile_m * stride_mask_row + j)
        if active == 0:
            continue                    # entire block skipped — no HBM load

        offs_n  = j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < SEQ_KV

        K = tl.load(
            K_ptr + k_base + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk,
            mask=kv_mask[None, :], other=0.0, eviction_policy="evict_last",
        )
        V = tl.load(
            V_ptr + v_base + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk,
            mask=kv_mask[:, None], other=0.0, eviction_policy="evict_last",
        )

        S = tl.dot(Q, K) * scale
        S = tl.where(kv_mask[None, :], S, float("-inf"))

        m_j   = tl.max(S, axis=1)
        new_m = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - new_m)
        P     = tl.exp(S - new_m[:, None])

        l_i = alpha * l_i + tl.sum(P, axis=1)
        O_i = alpha[:, None] * O_i + tl.dot(P.to(tl.float16), V)
        m_i = new_m

    O_i /= l_i[:, None]

    tl.store(
        O_ptr + o_base + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
        O_i.to(tl.float16), mask=q_mask, eviction_policy="evict_first",
    )
    tl.store(L_ptr + bh * SEQ_Q + offs_m, m_i + tl.log(l_i), mask=offs_m < SEQ_Q)


# ==============================================================================
# KERNEL 2: Gated Delta Network (forward, chunkwise sequential)
# ==============================================================================
#
# The delta rule is a fast-weight / linear-attention variant:
#   retrieved_t = W_{t-1} @ k_t              ← recall from state
#   delta_t     = v_t - retrieved_t          ← prediction error
#   W_t         = g_t[:,None] * W_{t-1}  +  beta_t * outer(delta_t, k_t)
#   o_t         = W_t @ q_t
#
# W ∈ R^(D × D) is the recurrent state (fast-weight matrix).
# g_t ∈ R^D  is a per-row gate (sigmoid), controls forgetting.
# beta_t ∈ R  is the learning rate (scalar, sigmoid activated).
#
# Why chunkwise?
#   W lives in SRAM (D=64 → 64×64×4 = 16 KB). One CTA per (batch,head)
#   loops over ALL tokens sequentially — W stays in registers/SRAM the whole
#   time. Only Q,K,V,G,Beta and O touch HBM per token.
#
# Grid: (B * H,)  — one program owns the entire sequence for one head.
# ==============================================================================

@triton.jit
def gated_delta_fwd(
    Q_ptr, K_ptr, V_ptr,
    Beta_ptr,                   # (B, H, SEQ)       scalar gate per token
    G_ptr,                      # (B, H, SEQ, D)    per-dim forgetting gate
    O_ptr,
    H_ptr,                      # (B, H, D, D) fp32 — initial state (usually zeros)
                                #   updated in-place; contains final state after kernel
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_bb, stride_bh, stride_bs,            # beta strides
    stride_gb, stride_gh, stride_gs, stride_gd, # gate strides
    stride_ob, stride_oh, stride_os, stride_od,
    stride_hb, stride_hh, stride_hi, stride_hj, # state strides: (B,H,D,D)
    SEQ:      tl.constexpr,
    HEAD_DIM: tl.constexpr,     # D
):
    bh    = tl.program_id(0)    # flattened (batch, head) index

    q_base = bh * stride_qh
    k_base = bh * stride_kh
    v_base = bh * stride_vh
    b_base = bh * stride_bh
    g_base = bh * stride_gh
    o_base = bh * stride_oh
    h_base = bh * stride_hh    # start of this head's D×D state block

    offs_d = tl.arange(0, HEAD_DIM)   # [D]

    # ── load initial state W into SRAM  [D, D] ──────────────────────────
    # W[i, j] = H_ptr + h_base + i * stride_hi + j * stride_hj
    W = tl.load(
        H_ptr + h_base
              + offs_d[:, None] * stride_hi    # row index i
              + offs_d[None, :] * stride_hj,   # col index j
    )  # [D, D] fp32, lives in registers for the whole loop

    # ── sequential loop over tokens ──────────────────────────────────────
    for t in range(SEQ):
        t_off = t   # scalar token index

        # load per-token vectors [D]
        k = tl.load(K_ptr + k_base + t_off * stride_ks + offs_d * stride_kd)
        v = tl.load(V_ptr + v_base + t_off * stride_vs + offs_d * stride_vd)
        q = tl.load(Q_ptr + q_base + t_off * stride_qs + offs_d * stride_qd)
        g = tl.load(G_ptr + g_base + t_off * stride_gs + offs_d * stride_gd)

        # beta: scalar  →  load as 1-element then squeeze
        beta = tl.load(Beta_ptr + b_base + t_off * stride_bs)

        k = k.to(tl.float32)
        v = v.to(tl.float32)
        q = q.to(tl.float32)
        g = tl.sigmoid(g.to(tl.float32))         # gate ∈ (0,1)
        beta = tl.sigmoid(beta.to(tl.float32))   # learning rate ∈ (0,1)

        # ── delta rule update ────────────────────────────────────────────
        # retrieved = W @ k     [D, D] x [D] → [D]
        retrieved = tl.dot(W, k[:, None])         # [D, 1]
        retrieved = tl.reshape(retrieved, [HEAD_DIM])

        delta = v - retrieved                     # [D]  prediction error

        # W = g[:,None] * W  +  beta * outer(delta, k)
        # outer(delta, k): delta[i]*k[j] → [D, D]
        W = g[:, None] * W + beta * delta[:, None] * k[None, :]

        # ── output ───────────────────────────────────────────────────────
        # o = W @ q   [D, D] x [D] → [D]
        o = tl.dot(W, q[:, None])                 # [D, 1]
        o = tl.reshape(o, [HEAD_DIM])

        tl.store(
            O_ptr + o_base + t_off * stride_os + offs_d * stride_od,
            o.to(tl.float16),
        )

    # ── write final state back to HBM ────────────────────────────────────
    tl.store(
        H_ptr + h_base
              + offs_d[:, None] * stride_hi
              + offs_d[None, :] * stride_hj,
        W,
    )
