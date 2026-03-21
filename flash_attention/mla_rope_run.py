"""
Runner + correctness check for the fused MLA+RoPE kernel.

Verifies against a naive reference that does each step separately:
  1. decompress:  K_nope = C @ W_k_nope,  K_r = C @ W_kr
  2. RoPE:        K_r_rot = rope(K_r, abs_positions)
  3. concat:      K = cat([K_nope, K_r_rot], dim=-1)
  4. Q RoPE:      apply rope to Q_r part at query positions
  5. attention:   softmax(Q @ K^T * scale) @ V

The kernel fuses all 5 steps — K never touches HBM.

python mla_rope_run.py
"""

import torch
import math
from mla_rope_kernel import mla_rope_fwd

# ── sizes ─────────────────────────────────────────────────────────────────────
B, H, SEQ  = 1, 4, 256
D_NOPE     = 64    # content key/query dim
D_ROPE     = 32    # RoPE key/query dim  (D_ROPE_HALF = 16)
D_ROPE_HALF= D_ROPE // 2
HEAD_DIM   = 64    # V output dim
LATENT_DIM = 64    # compressed latent dim (small for test; DeepSeek uses 512)

BLOCK_M, BLOCK_N = 32, 32
NUM_WARPS  = 4
NUM_STAGES = 2

scale = 1.0 / math.sqrt(D_NOPE + D_ROPE)
grid  = (math.ceil(SEQ / BLOCK_M), B * H)

dev = "cuda"
dt  = torch.float16


# ── build RoPE tables ─────────────────────────────────────────────────────────
def build_rope_tables(max_seq, d_rope_half, base=10000.0, device="cuda"):
    """cos/sin tables, shape [max_seq, d_rope_half]"""
    theta = 1.0 / (base ** (torch.arange(0, d_rope_half, device=device).float()
                             / d_rope_half))
    pos   = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(pos, theta)   # [max_seq, d_rope_half]
    return freqs.cos().to(dt), freqs.sin().to(dt)

cos_table, sin_table = build_rope_tables(SEQ, D_ROPE_HALF, device=dev)


# ── apply split-half RoPE (used on Q outside the kernel) ─────────────────────
def apply_rope(x, cos, sin):
    """x: [..., D_ROPE]  cos/sin: [seq, D_ROPE//2]"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos, sin
    return torch.cat([x1 * c - x2 * s,
                      x2 * c + x1 * s], dim=-1)


# ── tensors ───────────────────────────────────────────────────────────────────
# raw Q before RoPE: [B, H, SEQ, D_NOPE + D_ROPE]
Q_raw   = torch.randn(B, H, SEQ, D_NOPE + D_ROPE, device=dev, dtype=dt)
# C_kv: shared latent — no head dimension
C_kv    = torch.randn(B, SEQ, LATENT_DIM, device=dev, dtype=dt)
W_k_nope= torch.randn(H, LATENT_DIM, D_NOPE, device=dev, dtype=dt)
W_kr    = torch.randn(H, LATENT_DIM, D_ROPE, device=dev, dtype=dt)
W_v     = torch.randn(H, LATENT_DIM, HEAD_DIM, device=dev, dtype=dt)

# Pre-apply RoPE to the rope portion of Q at query positions [0, SEQ)
# (This happens outside the kernel — query position is fixed per CTA.)
Q_nope_raw = Q_raw[..., :D_NOPE]                       # [B,H,SEQ,D_NOPE]
Q_r_raw    = Q_raw[..., D_NOPE:]                       # [B,H,SEQ,D_ROPE]
# Each query token i gets cos/sin[i]
cos_q = cos_table[:SEQ]                                # [SEQ, D_ROPE_HALF]
sin_q = sin_table[:SEQ]                                # [SEQ, D_ROPE_HALF]
Q_r_rot = apply_rope(Q_r_raw, cos_q[None, None, :, :], sin_q[None, None, :, :])
Q        = torch.cat([Q_nope_raw, Q_r_rot], dim=-1).contiguous()  # [B,H,SEQ,D_NOPE+D_ROPE]

O   = torch.empty(B, H, SEQ, HEAD_DIM, device=dev, dtype=dt)
Lse = torch.empty(B, H, SEQ,            device=dev, dtype=torch.float32)

# ── launch ────────────────────────────────────────────────────────────────────
mla_rope_fwd[grid](
    Q, C_kv, W_k_nope, W_kr, W_v,
    cos_table, sin_table,
    O, Lse,
    *Q.stride(),           # stride_qb, qh, qm, qd
    *C_kv.stride(),        # stride_cb, cn, cl
    *W_k_nope.stride(),    # stride_wknh, wknl, wknd
    *W_kr.stride(),        # stride_wkrh, wkrl, wkrd
    *W_v.stride(),         # stride_wvh, wvl, wvd
    cos_table.stride(0), sin_table.stride(0),   # stride_cos_s, stride_sin_s
    *O.stride(),           # stride_ob, oh, om, od
    H=H,
    SEQ_Q=SEQ, SEQ_KV=SEQ,
    LATENT_DIM=LATENT_DIM,
    D_NOPE=D_NOPE, D_ROPE=D_ROPE,
    HEAD_DIM=HEAD_DIM,
    scale=scale,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    num_warps=NUM_WARPS, num_stages=NUM_STAGES,
)

# ── reference: naive step-by-step ─────────────────────────────────────────────
# 1. decompress K and V
K_nope_ref = torch.einsum("bsl,hld->bhsd", C_kv.float(), W_k_nope.float())
K_r_ref    = torch.einsum("bsl,hld->bhsd", C_kv.float(), W_kr.float())
V_ref      = torch.einsum("bsl,hld->bhsd", C_kv.float(), W_v.float())

# 2. apply RoPE to K_r at KV positions [0, SEQ)
K_r_rot_ref = apply_rope(K_r_ref,
                          cos_table[None, None, :SEQ, :].float(),
                          sin_table[None, None, :SEQ, :].float())

# 3. full K
K_ref = torch.cat([K_nope_ref, K_r_rot_ref], dim=-1)   # [B,H,SEQ,D_NOPE+D_ROPE]

# 4. attention with pre-rotated Q
Q_ref = Q.float()   # already rotated
S_ref = torch.einsum("bhmd,bhnd->bhmn", Q_ref, K_ref) * scale
P_ref = torch.softmax(S_ref, dim=-1)
O_ref = torch.einsum("bhmn,bhnd->bhmd", P_ref, V_ref).half()

max_diff = (O - O_ref).abs().max().item()
print(f"Forward MLA+RoPE: max |O_triton - O_ref| = {max_diff:.5f}  "
      f"({'PASS' if max_diff < 0.05 else 'FAIL'})")

# ── what was fused ────────────────────────────────────────────────────────────
bytes_saved = B * H * SEQ * (D_NOPE + D_ROPE) * 2   # intermediate K never written
print(f"\nHBM writes eliminated (intermediate K): {bytes_saved / 1024:.1f} KB")
print(f"  (at SEQ=4096 this would be {B*H*4096*(D_NOPE+D_ROPE)*2 / 1024:.0f} KB)")
