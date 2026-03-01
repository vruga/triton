"""
python run.py
"""

import torch
import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from kernel import sparse_attn_fwd, gated_delta_fwd

B, H, SEQ, D = 1, 4, 512, 64
BLOCK_M, BLOCK_N = 128, 64
NUM_WARPS, NUM_STAGES = 4, 3
scale = 1.0 / math.sqrt(D)

# ── helpers ───────────────────────────────────────────────────────────────────
def naive_attn(q, k, v, mask_2d=None):
    """mask_2d: (Sq, Skv) bool, True = attend."""
    S = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * scale
    if mask_2d is not None:
        S = S.masked_fill(~mask_2d[None, None], float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", P, v.float()).half()


# ==============================================================================
# TEST 1: Block-Sparse Attention
# ==============================================================================
print("=" * 60)
print("Block-Sparse Attention")
print("=" * 60)

q = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)

num_q_blocks  = SEQ // BLOCK_M       # 4
num_kv_blocks = SEQ // BLOCK_N       # 8

# Sparsity pattern: sliding window — each Q block attends to 3 neighbouring KV blocks.
# block_mask[i, j] = 1 if Q-block i attends to KV-block j
block_mask = torch.zeros(num_q_blocks, num_kv_blocks, device="cuda", dtype=torch.int8)
for i in range(num_q_blocks):
    for j in range(num_kv_blocks):
        if abs(i * (BLOCK_M // BLOCK_N) - j) <= 1:  # ±1 KV block window
            block_mask[i, j] = 1

sparsity = 1.0 - block_mask.float().mean().item()
print(f"  sparsity = {sparsity*100:.0f}%  "
      f"({block_mask.sum().item()} / {num_q_blocks * num_kv_blocks} blocks active)")

O_sp = torch.zeros(B, H, SEQ, D, device="cuda", dtype=torch.float16)
L_sp = torch.zeros(B, H, SEQ, device="cuda", dtype=torch.float32)

grid = (num_q_blocks, B * H)
sparse_attn_fwd[grid](
    q, k, v, O_sp, L_sp,
    block_mask,
    *q.stride(), *k.stride(), *v.stride(), *O_sp.stride(),
    num_kv_blocks,                # stride_mask_row
    SEQ_Q=SEQ, SEQ_KV=SEQ, HEAD_DIM=D, scale=scale,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    num_warps=NUM_WARPS, num_stages=NUM_STAGES,
)

# Reference: build the equivalent dense token-level mask
token_mask = torch.zeros(SEQ, SEQ, device="cuda", dtype=torch.bool)
for i in range(num_q_blocks):
    for j in range(num_kv_blocks):
        if block_mask[i, j]:
            r0, r1 = i * BLOCK_M, (i + 1) * BLOCK_M
            c0, c1 = j * BLOCK_N, (j + 1) * BLOCK_N
            token_mask[r0:r1, c0:c1] = True

O_ref = naive_attn(q, k, v, mask_2d=token_mask)
diff = (O_sp - O_ref).abs().max().item()
print(f"  max |O_sparse - O_ref| = {diff:.5f}  "
      f"({'PASS' if diff < 0.05 else 'FAIL'})")


# ==============================================================================
# TEST 2: Gated Delta Network
# ==============================================================================
print()
print("=" * 60)
print("Gated Delta Network")
print("=" * 60)

q2    = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
k2    = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
v2    = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
beta  = torch.randn(B, H, SEQ,    device="cuda", dtype=torch.float16)
gate  = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
H_state = torch.zeros(B, H, D, D, device="cuda", dtype=torch.float32)
O_delta = torch.zeros(B, H, SEQ, D, device="cuda", dtype=torch.float16)

gated_delta_fwd[(B * H,)](
    q2, k2, v2, beta, gate, O_delta, H_state,
    *q2.stride(), *k2.stride(), *v2.stride(),
    *beta.stride(), *gate.stride(),
    *O_delta.stride(), *H_state.stride(),
    SEQ=SEQ, HEAD_DIM=D,
    num_warps=4, num_stages=1,    # sequential recurrence → 1 stage
)

# Reference: Python loop
def naive_gated_delta(q, k, v, beta, gate):
    B, H, S, D = q.shape
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            W = torch.zeros(D, D, device=q.device, dtype=torch.float32)
            for t in range(S):
                kt  = k[b, h, t].float()
                vt  = v[b, h, t].float()
                qt  = q[b, h, t].float()
                bt  = torch.sigmoid(beta[b, h, t].float())
                gt  = torch.sigmoid(gate[b, h, t].float())
                ret = W @ kt
                delta = vt - ret
                W = gt[:, None] * W + bt * delta[:, None] * kt[None, :]
                out[b, h, t] = W @ qt
    return out.half()

O_ref2 = naive_gated_delta(q2, k2, v2, beta, gate)
diff2  = (O_delta - O_ref2).abs().max().item()
print(f"  max |O_delta - O_ref| = {diff2:.5f}  "
      f"({'PASS' if diff2 < 0.05 else 'FAIL'})")
print(f"  final state norm      = {H_state.norm().item():.3f}")
