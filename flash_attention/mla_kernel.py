"""
Fused Multi-Head Latent Attention (MLA) kernel.

Standard attention:   Q [B,H,S,D]  K [B,H,S,D]  V [B,H,S,D]
MLA:                  Q [B,H,S,D]  C_kv [B,S,L]  W_k_up [H,L,D]  W_v_up [H,L,D]

where L = LATENT_DIM (512) << H*D (16*128 = 2048)

The KV-cache stores C_kv instead of full K and V:
    4x smaller cache  (512 vs 2048 dims)

Inside this kernel K and V are never written to HBM.
They are computed on the fly from C and the weight matrices and
live only in registers.
"""

import triton
import triton.language as tl


@triton.jit
def mla_flash_attn_fwd(
    # ── inputs ──────────────────────────────────────────────────────────────
    Q_ptr,      # [B, H, SEQ_Q,  HEAD_DIM]  — per-head queries (same as standard)
    C_kv_ptr,   # [B,    SEQ_KV, LATENT_DIM] — shared latent, NO head dim!
    W_k_up_ptr, # [H, LATENT_DIM, HEAD_DIM]  — per-head up-projection for K
    W_v_up_ptr, # [H, LATENT_DIM, HEAD_DIM]  — per-head up-projection for V
    # ── outputs ─────────────────────────────────────────────────────────────
    O_ptr,      # [B, H, SEQ_Q, HEAD_DIM]
    L_ptr,      # [B, H, SEQ_Q]  log-sum-exp for backward recomputation
    # ── Q strides ───────────────────────────────────────────────────────────
    stride_qb, stride_qh, stride_qm, stride_qd,
    # ── C_kv strides (B, SEQ, LATENT — no head stride) ──────────────────────
    stride_cb, stride_cn, stride_cl,
    # ── W_k_up strides [H, LATENT, HEAD] ────────────────────────────────────
    stride_wkh, stride_wkl, stride_wkd,
    # ── W_v_up strides [H, LATENT, HEAD] ────────────────────────────────────
    stride_wvh, stride_wvl, stride_wvd,
    # ── O strides ───────────────────────────────────────────────────────────
    stride_ob, stride_oh, stride_om, stride_od,
    # ── problem shape ───────────────────────────────────────────────────────
    H:          tl.constexpr,
    SEQ_Q:      tl.constexpr,
    SEQ_KV:     tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    LATENT_DIM: tl.constexpr,
    scale,
    BLOCK_M:    tl.constexpr,
    BLOCK_N:    tl.constexpr,
):
    # ── program IDs ──────────────────────────────────────────────────────────
    # Grid: (ceil(SEQ_Q / BLOCK_M),  B * H)   ← same shape as standard kernel
    tile_m = tl.program_id(0)
    bh     = tl.program_id(1)
    b      = bh // H   # batch index
    h      = bh % H    # head index

    # ── base pointers ────────────────────────────────────────────────────────
    q_base  = b * stride_qb + h * stride_qh
    c_base  = b * stride_cb                   # ← NO head offset: C_kv is shared!
    wk_base = h * stride_wkh
    wv_base = h * stride_wvh
    o_base  = b * stride_ob + h * stride_oh

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, HEAD_DIM)                     # [HEAD_DIM]
    offs_l = tl.arange(0, LATENT_DIM)                   # [LATENT_DIM]
    q_mask = offs_m[:, None] < SEQ_Q

    # ── load Q block ─────────────────────────────────────────────────────────
    # Identical to standard kernel — Q is a normal per-head tensor.
    Q = tl.load(
        Q_ptr + q_base
              + offs_m[:, None] * stride_qm
              + offs_d[None, :] * stride_qd,
        mask=q_mask,
        other=0.0,
        eviction_policy="evict_first",
    )  # [BLOCK_M, HEAD_DIM]

    # ── DIFFERENCE 1: load weight matrices once, before the KV loop ──────────
    #
    # In standard flash-attn there are NO weight matrices; K and V are already
    # fully materialized per-head tensors in HBM.
    #
    # Here W_k and W_v are [LATENT_DIM, HEAD_DIM] = [512, 128] each.
    # They are fixed for the whole inner loop — loaded once into registers,
    # and also pinned in L2 (evict_last) so every other tile_m CTA for the
    # same head gets an L2 hit instead of an HBM fetch.
    #
    W_k = tl.load(
        W_k_up_ptr + wk_base
                   + offs_l[:, None] * stride_wkl
                   + offs_d[None, :] * stride_wkd,
        eviction_policy="evict_last",
    )  # [LATENT_DIM, HEAD_DIM]

    W_v = tl.load(
        W_v_up_ptr + wv_base
                   + offs_l[:, None] * stride_wvl
                   + offs_d[None, :] * stride_wvd,
        eviction_policy="evict_last",
    )  # [LATENT_DIM, HEAD_DIM]

    # ── online-softmax accumulators (same as standard) ───────────────────────
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    O_i = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    for j in range(tl.cdiv(SEQ_KV, BLOCK_N)):
        offs_n  = j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < SEQ_KV

        # ── DIFFERENCE 2: load C (latent) instead of K and V ─────────────────
        #
        # Standard kernel loads:
        #   K [HEAD_DIM, BLOCK_N]  (pre-transposed, 64×128 fp16 = 16 KB)
        #   V [BLOCK_N, HEAD_DIM]  (64×128 fp16 = 16 KB)
        #   Total: 32 KB per iteration
        #
        # MLA kernel loads:
        #   C [BLOCK_N, LATENT_DIM] (64×512 fp16 = 64 KB)  ← 2× more bandwidth!
        #   Total: 64 KB per iteration
        #
        # The extra bandwidth is the price of the 4× smaller KV cache.
        #
        C = tl.load(
            C_kv_ptr + c_base
                     + offs_n[:, None] * stride_cn
                     + offs_l[None, :] * stride_cl,
            mask=kv_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",   # C is used once per iteration
        )  # [BLOCK_N, LATENT_DIM]

        # ── DIFFERENCE 3: on-the-fly decompression ────────────────────────────
        #
        # K and V are never in HBM. They are computed here in registers and
        # discarded at the end of each iteration. No extra memory allocated.
        #
        #   K = C @ W_k :  [BLOCK_N, LATENT_DIM] @ [LATENT_DIM, HEAD_DIM]
        #                → [BLOCK_N, HEAD_DIM]     (2 × 64 × 512 × 128 FLOPs)
        #   V = C @ W_v :  same shape, same cost
        #
        # This adds ~2× 8M FLOPs per block vs the original kernel's
        # ~1M FLOPs for Q@K (64×128×64×2). Compute is the new bottleneck.
        #
        K_block = tl.dot(C.to(tl.float16), W_k.to(tl.float16))  # [BLOCK_N, HEAD_DIM]
        V_block = tl.dot(C.to(tl.float16), W_v.to(tl.float16))  # [BLOCK_N, HEAD_DIM]

        # ── standard flash-attention from here (identical to original) ────────
        #
        # K_block is [BLOCK_N, HEAD_DIM] — transpose before dot with Q.
        # (The standard kernel avoids this by loading K pre-transposed from HBM;
        #  here we can't because K was just computed, not loaded.)
        #
        S = tl.dot(Q, tl.trans(K_block)) * scale   # [BLOCK_M, BLOCK_N]
        S = tl.where(kv_mask[None, :], S, float("-inf"))

        m_j   = tl.max(S, axis=1)
        new_m = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - new_m)
        P     = tl.exp(S - new_m[:, None])

        l_i   = alpha * l_i + tl.sum(P, axis=1)
        O_i   = alpha[:, None] * O_i + tl.dot(P.to(tl.float16), V_block)
        m_i   = new_m

    O_i /= l_i[:, None]

    tl.store(
        O_ptr + o_base
              + offs_m[:, None] * stride_om
              + offs_d[None, :] * stride_od,
        O_i.to(tl.float16),
        mask=q_mask,
        eviction_policy="evict_first",
    )

    tl.store(
        L_ptr + bh * SEQ_Q + offs_m,
        m_i + tl.log(l_i),
        mask=offs_m < SEQ_Q,
    )
