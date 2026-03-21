"""
Fused MLA + RoPE kernel  (DeepSeek-V2 style decoupled RoPE)
============================================================

mla_kernel.py fused:  C_kv → decompress K,V → flash-attn
This kernel fuses:    C_kv → decompress K_nope, K_r, V
                            → apply RoPE to K_r   (inline, no HBM write)
                            → split-dot attention  (no K concat needed)
                            → O

Three new ideas vs mla_kernel.py
---------------------------------
1.  W_kr is loaded in two halves (W_kr1, W_kr2) to avoid a rotate_half
    gather inside the hot loop.  RoPE becomes pure elementwise math:
        K_r1_rot = K_r1 * cos − K_r2 * sin
        K_r2_rot = K_r2 * cos + K_r1 * sin
    (complex multiplication on the two halves — no permutation needed.)

2.  cos/sin are loaded PER KV BLOCK from a prebuilt table indexed by the
    absolute sequence positions offs_n.  They change every j-iteration,
    which is why RoPE must be inside the loop.

3.  K is never concatenated.  The dot product decomposes:
        S = Q_nope @ K_nope^T
          + Q_r1   @ K_r1_rot^T
          + Q_r2   @ K_r2_rot^T
    Three tl.dot calls on smaller matrices — each uses tensor cores,
    and no intermediate [BLOCK_N, D_nope+D_rope] tensor is ever formed.

Shapes
------
  Q:       [B, H, SEQ_Q, D_NOPE + D_ROPE]   (Q_rope part already RoPE-rotated)
  C_kv:    [B, SEQ_KV, LATENT_DIM]           (shared latent — no head dim!)
  W_k_nope:[H, LATENT_DIM, D_NOPE]
  W_kr:    [H, LATENT_DIM, D_ROPE]           (loaded as two halves internally)
  W_v:     [H, LATENT_DIM, HEAD_DIM]
  cos/sin: [MAX_SEQ, D_ROPE//2]
  O:       [B, H, SEQ_Q, HEAD_DIM]
"""

import triton
import triton.language as tl


@triton.jit
def mla_rope_fwd(
    # ── inputs ───────────────────────────────────────────────────────────────
    Q_ptr,          # [B, H, SEQ_Q, D_NOPE + D_ROPE]  Q_rope part pre-rotated
    C_kv_ptr,       # [B, SEQ_KV, LATENT_DIM]          shared KV latent, no head dim
    W_k_nope_ptr,   # [H, LATENT_DIM, D_NOPE]          content-key projection
    W_kr_ptr,       # [H, LATENT_DIM, D_ROPE]           RoPE-key projection (both halves)
    W_v_ptr,        # [H, LATENT_DIM, HEAD_DIM]
    cos_ptr,        # [MAX_SEQ, D_ROPE//2]
    sin_ptr,        # [MAX_SEQ, D_ROPE//2]
    # ── outputs ──────────────────────────────────────────────────────────────
    O_ptr,          # [B, H, SEQ_Q, HEAD_DIM]
    L_ptr,          # [B, H, SEQ_Q]  log-sum-exp for backward
    # ── Q strides ────────────────────────────────────────────────────────────
    stride_qb, stride_qh, stride_qm, stride_qd,
    # ── C_kv strides  (no head stride — shared) ──────────────────────────────
    stride_cb, stride_cn, stride_cl,
    # ── W_k_nope strides [H, L, D_NOPE] ─────────────────────────────────────
    stride_wknh, stride_wknl, stride_wknd,
    # ── W_kr strides     [H, L, D_ROPE] ─────────────────────────────────────
    stride_wkrh, stride_wkrl, stride_wkrd,
    # ── W_v strides      [H, L, HEAD_DIM] ───────────────────────────────────
    stride_wvh, stride_wvl, stride_wvd,
    # ── cos/sin strides  [MAX_SEQ, D_ROPE//2] ────────────────────────────────
    stride_cos_s, stride_sin_s,
    # ── O strides ────────────────────────────────────────────────────────────
    stride_ob, stride_oh, stride_om, stride_od,
    # ── problem shape ─────────────────────────────────────────────────────────
    H:           tl.constexpr,
    SEQ_Q:       tl.constexpr,
    SEQ_KV:      tl.constexpr,
    LATENT_DIM:  tl.constexpr,
    D_NOPE:      tl.constexpr,   # content key/query dimension  (e.g. 128)
    D_ROPE:      tl.constexpr,   # RoPE key/query dimension     (e.g. 64)
    HEAD_DIM:    tl.constexpr,   # V output dimension           (e.g. 128)
    scale,
    BLOCK_M:     tl.constexpr,
    BLOCK_N:     tl.constexpr,
):
    D_ROPE_HALF: tl.constexpr = D_ROPE // 2

    # ── program IDs  (grid identical to all other kernels) ────────────────────
    tile_m = tl.program_id(0)
    bh     = tl.program_id(1)
    b      = bh // H
    h      = bh % H

    # ── base pointers ─────────────────────────────────────────────────────────
    q_base   = b * stride_qb  + h * stride_qh
    c_base   = b * stride_cb                    # NO head offset — shared latent
    wkn_base = h * stride_wknh
    wkr_base = h * stride_wkrh
    wv_base  = h * stride_wvh
    o_base   = b * stride_ob  + h * stride_oh

    offs_m   = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_nope= tl.arange(0, D_NOPE)                        # [D_NOPE]
    offs_rh  = tl.arange(0, D_ROPE_HALF)                   # [D_ROPE//2]
    offs_v   = tl.arange(0, HEAD_DIM)                      # [HEAD_DIM]
    offs_l   = tl.arange(0, LATENT_DIM)                    # [LATENT_DIM]
    q_mask   = offs_m[:, None] < SEQ_Q

    # ── load Q in three slices ─────────────────────────────────────────────────
    #
    # Q is stored as [D_NOPE | D_ROPE//2 | D_ROPE//2] contiguously.
    # Q_rope is pre-rotated outside the kernel (query position is fixed
    # for the whole inner loop, so no benefit fusing it here).
    #
    # We need the three slices separately to use the split-dot trick later.
    #
    Q_nope = tl.load(                                       # [BLOCK_M, D_NOPE]
        Q_ptr + q_base
              + offs_m[:, None] * stride_qm
              + offs_nope[None, :] * stride_qd,
        mask=q_mask, other=0.0,
        eviction_policy="evict_first",
    )
    Q_r1 = tl.load(                                         # [BLOCK_M, D_ROPE//2]
        Q_ptr + q_base
              + offs_m[:, None] * stride_qm
              + (offs_rh + D_NOPE)[None, :] * stride_qd,
        mask=q_mask, other=0.0,
        eviction_policy="evict_first",
    )
    Q_r2 = tl.load(                                         # [BLOCK_M, D_ROPE//2]
        Q_ptr + q_base
              + offs_m[:, None] * stride_qm
              + (offs_rh + D_NOPE + D_ROPE_HALF)[None, :] * stride_qd,
        mask=q_mask, other=0.0,
        eviction_policy="evict_first",
    )

    # ── load weight matrices ONCE before the KV loop ──────────────────────────
    #
    # Same strategy as mla_kernel.py: evict_last pins them in L2.
    # Every other tile_m CTA for the same head shares them via L2 cache.
    #
    # NEW: W_kr is loaded as two halves to enable the split-dot trick.
    # W_kr[h, :, :half]  → W_kr1  [LATENT_DIM, D_ROPE//2]
    # W_kr[h, :, half:]  → W_kr2  [LATENT_DIM, D_ROPE//2]
    # These are adjacent in memory, just different column offsets.
    #
    W_k_nope = tl.load(                                     # [LATENT_DIM, D_NOPE]
        W_k_nope_ptr + wkn_base
                     + offs_l[:, None] * stride_wknl
                     + offs_nope[None, :] * stride_wknd,
        eviction_policy="evict_last",
    )
    W_kr1 = tl.load(                                        # [LATENT_DIM, D_ROPE//2]
        W_kr_ptr + wkr_base
                 + offs_l[:, None] * stride_wkrl
                 + offs_rh[None, :] * stride_wkrd,
        eviction_policy="evict_last",
    )
    W_kr2 = tl.load(                                        # [LATENT_DIM, D_ROPE//2]
        W_kr_ptr + wkr_base
                 + offs_l[:, None] * stride_wkrl
                 + (offs_rh + D_ROPE_HALF)[None, :] * stride_wkrd,
        eviction_policy="evict_last",
    )
    W_v = tl.load(                                          # [LATENT_DIM, HEAD_DIM]
        W_v_ptr + wv_base
                + offs_l[:, None] * stride_wvl
                + offs_v[None, :] * stride_wvd,
        eviction_policy="evict_last",
    )

    # ── accumulators (identical to every other kernel) ────────────────────────
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    O_i = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    for j in range(tl.cdiv(SEQ_KV, BLOCK_N)):
        offs_n  = j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < SEQ_KV

        # ── load latent C ──────────────────────────────────────────────────────
        # [BLOCK_N, LATENT_DIM] — same as mla_kernel.py
        C = tl.load(
            C_kv_ptr + c_base
                     + offs_n[:, None] * stride_cn
                     + offs_l[None, :] * stride_cl,
            mask=kv_mask[:, None], other=0.0,
            eviction_policy="evict_first",
        )

        # ── decompress ────────────────────────────────────────────────────────
        # Same structure as mla_kernel.py, but K is now three tensors not one.
        K_nope = tl.dot(C.to(tl.float16), W_k_nope.to(tl.float16))  # [BLOCK_N, D_NOPE]
        K_r1   = tl.dot(C.to(tl.float16), W_kr1.to(tl.float16))     # [BLOCK_N, D_ROPE//2]
        K_r2   = tl.dot(C.to(tl.float16), W_kr2.to(tl.float16))     # [BLOCK_N, D_ROPE//2]
        V      = tl.dot(C.to(tl.float16), W_v.to(tl.float16))       # [BLOCK_N, HEAD_DIM]

        # ── NEW: load cos/sin for these KV positions ───────────────────────────
        #
        # offs_n are the absolute sequence positions of this KV block.
        # cos and sin change every j-iteration — this is why RoPE cannot be
        # hoisted outside the loop (unlike the weight matrices).
        #
        # mask other=1.0 for cos, other=0.0 for sin so that padding tokens
        # (offs_n >= SEQ_KV) get identity rotation: x*1 + rotate(x)*0 = x.
        #
        cos = tl.load(                                       # [BLOCK_N, D_ROPE//2]
            cos_ptr + offs_n[:, None] * stride_cos_s + offs_rh[None, :],
            mask=kv_mask[:, None], other=1.0,
        )
        sin = tl.load(                                       # [BLOCK_N, D_ROPE//2]
            sin_ptr + offs_n[:, None] * stride_sin_s + offs_rh[None, :],
            mask=kv_mask[:, None], other=0.0,
        )

        # ── NEW: apply split-half RoPE to K_r ─────────────────────────────────
        #
        # Standard split-half RoPE (same as LLaMA / GPT-NeoX):
        #   K_rot[:half]  = K[:half]  * cos − K[half:] * sin
        #   K_rot[half:]  = K[half:]  * cos + K[:half] * sin
        #
        # Because W_kr was loaded in two halves, K_r1 = K[:half] and
        # K_r2 = K[half:] are already separate tensors. No gather/permute
        # needed — this is the whole point of the split-weight design.
        #
        # This is equivalent to complex multiplication:
        #   (K_r1 + i*K_r2) * exp(i*theta) = (K_r1 + i*K_r2)(cos + i*sin)
        #
        K_r1_rot = K_r1 * cos - K_r2 * sin   # [BLOCK_N, D_ROPE//2]
        K_r2_rot = K_r2 * cos + K_r1 * sin   # [BLOCK_N, D_ROPE//2]

        # ── NEW: split-dot attention score ─────────────────────────────────────
        #
        # S = Q_total @ K_total^T
        #   = Q_nope @ K_nope^T  +  Q_r1 @ K_r1_rot^T  +  Q_r2 @ K_r2_rot^T
        #
        # We never form the full concatenated K [BLOCK_N, D_NOPE+D_ROPE].
        # Three partial tl.dot calls, each on tensor-core-friendly dimensions.
        #
        S = (tl.dot(Q_nope, tl.trans(K_nope))      # [BLOCK_M, BLOCK_N]
           + tl.dot(Q_r1,   tl.trans(K_r1_rot))    # [BLOCK_M, BLOCK_N]
           + tl.dot(Q_r2,   tl.trans(K_r2_rot))    # [BLOCK_M, BLOCK_N]
           ) * scale
        S = tl.where(kv_mask[None, :], S, float("-inf"))

        # ── online softmax — identical to all other kernels ───────────────────
        m_j   = tl.max(S, axis=1)
        new_m = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - new_m)
        P     = tl.exp(S - new_m[:, None])

        l_i   = alpha * l_i + tl.sum(P, axis=1)
        O_i   = alpha[:, None] * O_i + tl.dot(P.to(tl.float16), V)
        m_i   = new_m

    O_i /= l_i[:, None]

    # ── store output — identical to all other kernels ─────────────────────────
    tl.store(
        O_ptr + o_base
              + offs_m[:, None] * stride_om
              + offs_v[None, :] * stride_od,
        O_i.to(tl.float16),
        mask=q_mask,
        eviction_policy="evict_first",
    )
    tl.store(
        L_ptr + bh * SEQ_Q + offs_m,
        m_i + tl.log(l_i),
        mask=offs_m < SEQ_Q,
    )
