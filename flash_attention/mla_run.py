"""
Runner for the fused MLA kernel.

Verifies against naive MLA:
    K = C_kv @ W_k_up   (per head)
    V = C_kv @ W_v_up   (per head)
    S = Q @ K^T * scale
    O = softmax(S) @ V

python mla_run.py
"""

import torch
import math
from mla_kernel import mla_flash_attn_fwd

# ── problem size ──────────────────────────────────────────────────────────────
B, H, SEQ, D = 1, 4, 512, 64
L = 128   # LATENT_DIM — in DeepSeek-V2 this is 512; use 128 here so W fits in SRAM

BLOCK_M, BLOCK_N = 64, 64
NUM_WARPS  = 4
NUM_STAGES = 2

scale = 1.0 / math.sqrt(D)
grid  = (math.ceil(SEQ / BLOCK_M), B * H)

# ── tensors ───────────────────────────────────────────────────────────────────
# Q: standard per-head query
q     = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)

# C_kv: shared latent — NO head dimension (that's the whole point)
c_kv  = torch.randn(B, SEQ, L, device="cuda", dtype=torch.float16)

# Per-head up-projections (weight matrices, not sequence tensors)
w_k   = torch.randn(H, L, D, device="cuda", dtype=torch.float16)
w_v   = torch.randn(H, L, D, device="cuda", dtype=torch.float16)

O     = torch.empty(B, H, SEQ, D, device="cuda", dtype=torch.float16)
Lse   = torch.empty(B, H, SEQ,    device="cuda", dtype=torch.float32)

# ── launch ────────────────────────────────────────────────────────────────────
mla_flash_attn_fwd[grid](
    q, c_kv, w_k, w_v, O, Lse,
    *q.stride(),            # stride_qb, stride_qh, stride_qm, stride_qd
    *c_kv.stride(),         # stride_cb, stride_cn, stride_cl
    *w_k.stride(),          # stride_wkh, stride_wkl, stride_wkd
    *w_v.stride(),          # stride_wvh, stride_wvl, stride_wvd
    *O.stride(),            # stride_ob, stride_oh, stride_om, stride_od
    H=H,
    SEQ_Q=SEQ, SEQ_KV=SEQ,
    HEAD_DIM=D, LATENT_DIM=L,
    scale=scale,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    num_warps=NUM_WARPS, num_stages=NUM_STAGES,
)

# ── reference: naive MLA ──────────────────────────────────────────────────────
# c_kv: [B, SEQ, L]
# w_k:  [H, L, D]  →  K[b,h] = c_kv[b] @ w_k[h]
K_ref = torch.einsum("bsl,hld->bhsd", c_kv.float(), w_k.float())  # [B,H,SEQ,D]
V_ref = torch.einsum("bsl,hld->bhsd", c_kv.float(), w_v.float())  # [B,H,SEQ,D]

S_ref = torch.einsum("bhmd,bhnd->bhmn", q.float(), K_ref) * scale
P_ref = torch.softmax(S_ref, dim=-1)
O_ref = torch.einsum("bhmn,bhnd->bhmd", P_ref, V_ref).half()

max_diff = (O - O_ref).abs().max().item()
print(f"Forward MLA: max |O_triton - O_ref| = {max_diff:.5f}  "
      f"({'PASS' if max_diff < 0.05 else 'FAIL'})")

# ── memory comparison ─────────────────────────────────────────────────────────
kv_cache_standard = B * H * SEQ * D * 2   # K and V, fp16 bytes
kv_cache_mla      = B * SEQ * L * 2       # just C_kv, fp16 bytes
print(f"\nKV cache comparison (B={B}, H={H}, SEQ={SEQ}):")
print(f"  Standard (K+V):  {kv_cache_standard / 1024:.1f} KB")
print(f"  MLA (C_kv only): {kv_cache_mla      / 1024:.1f} KB")
print(f"  Reduction:       {kv_cache_standard / kv_cache_mla:.1f}x")
