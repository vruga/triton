"""
Minimal runner: allocate Q/K/V, launch the kernel, verify against naive attn.

python run.py
"""

import torch
import math
from kernel import flash_attn_fwd

# ── problem size ─────────────────────────────────────────────────────────────
B, H, SEQ, D = 1, 4, 512, 64

# ── block sizes for RTX 4090  (see SRAM math at top of kernel.py) ────────────
#  Q tile : 128 × 64 × 2 = 16 KB
#  K tile :  64 × 64 × 2 =  8 KB
#  V tile :  64 × 64 × 2 =  8 KB
#  O acc  : 128 × 64 × 4 = 32 KB  (fp32)
#  ─────────────────────────────────────────
#  Total  :              64 KB  ✓ < 100 KB usable SRAM
BLOCK_M, BLOCK_N = 128, 64
NUM_WARPS  = 4   # 4 × 32 = 128 threads per CTA
NUM_STAGES = 3   # software-pipeline depth for prefetch

# ── inputs ────────────────────────────────────────────────────────────────────
q = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
O = torch.empty_like(q)
L = torch.empty(B, H, SEQ, device="cuda", dtype=torch.float32)

scale = 1.0 / math.sqrt(D)

# ── grid:  (number of Q-row tiles,  B*H) ────────────────────────────────────
grid = (math.ceil(SEQ / BLOCK_M), B * H)

flash_attn_fwd[grid](
    q, k, v, O, L,
    *q.stride(), *k.stride(), *v.stride(), *O.stride(),
    SEQ_Q    = SEQ,
    SEQ_KV   = SEQ,
    HEAD_DIM = D,
    scale    = scale,
    BLOCK_M  = BLOCK_M,
    BLOCK_N  = BLOCK_N,
    num_warps  = NUM_WARPS,
    num_stages = NUM_STAGES,
)

# ── reference (naive fp32 softmax) ───────────────────────────────────────────
S_ref = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * scale
P_ref = torch.softmax(S_ref, dim=-1)
O_ref = torch.einsum("bhmn,bhnd->bhmd", P_ref, v.float()).half()

max_diff = (O - O_ref).abs().max().item()
print(f"max |O_triton - O_ref| = {max_diff:.5f}  "
      f"({'PASS' if max_diff < 0.05 else 'FAIL'})")
