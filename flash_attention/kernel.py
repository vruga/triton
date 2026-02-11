import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr,
    L_ptr,
    # strides — for (B, H, S, D) contiguous: stride_h = S*D, stride_s = D, stride_d = 1
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    SEQ_Q:    tl.constexpr,
    SEQ_KV:   tl.constexpr,
    HEAD_DIM: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (ceil(SEQ_Q / BLOCK_M),  B*H)
    # [1]
    tile_m = tl.program_id(0)
    bh     = tl.program_id(1)

    q_base = bh * stride_qh
    k_base = bh * stride_kh
    v_base = bh * stride_vh
    o_base = bh * stride_oh

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_k = tl.arange(0, HEAD_DIM)                     # [HEAD_DIM]
    q_mask = offs_m[:, None] < SEQ_Q

    # Q: one-time use per program → evict_first keeps L2 free for K/V
    Q = tl.load(
        Q_ptr + q_base
              + offs_m[:, None] * stride_qm
              + offs_k[None, :] * stride_qk,
        mask=q_mask,
        other=0.0,
        eviction_policy="evict_first",   # ← frees L2 for K/V reuse
    )  # [BLOCK_M, HEAD_DIM]  fp16

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M],              dtype=tl.float32)
    O_i = tl.zeros([BLOCK_M, HEAD_DIM],    dtype=tl.float32)

    for j in range(tl.cdiv(SEQ_KV, BLOCK_N)):
        offs_n  = j * BLOCK_N + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < SEQ_KV

        # K, V: every Q-tile program for this head loads the same blocks.
        # evict_last → hardware keeps them in L2 for the next Q-tile program.
        K = tl.load(
            K_ptr + k_base
                  + offs_n[None, :] * stride_kn    # pre-transposed: [HEAD_DIM, BLOCK_N]
                  + offs_k[:, None] * stride_kk,
            mask=kv_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",           # ← shared across Q-tile programs via L2
        )

        V = tl.load(
            V_ptr + v_base
                  + offs_n[:, None] * stride_vn
                  + offs_k[None, :] * stride_vk,
            mask=kv_mask[:, None],
            other=0.0,
            eviction_policy="evict_last",           # ← same
        )

        S = tl.dot(Q, K) * scale                   # [BLOCK_M, BLOCK_N]
        S = tl.where(kv_mask[None, :], S, float("-inf"))

        m_j   = tl.max(S, axis=1)
        new_m = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - new_m)
        P     = tl.exp(S - new_m[:, None])

        l_i   = alpha * l_i + tl.sum(P, axis=1)
        O_i   = alpha[:, None] * O_i + tl.dot(P.to(tl.float16), V)
        m_i   = new_m

    O_i /= l_i[:, None]

    # O: written once, no one reads it back → don't pollute L2
    tl.store(
        O_ptr + o_base
              + offs_m[:, None] * stride_om
              + offs_k[None, :] * stride_ok,
        O_i.to(tl.float16),
        mask=q_mask,
        eviction_policy="evict_first",
    )

    tl.store(
        L_ptr + bh * SEQ_Q + offs_m,
        m_i + tl.log(l_i),
        mask=offs_m < SEQ_Q,
    )
