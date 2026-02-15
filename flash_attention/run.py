"""
Minimal runner: allocate Q/K/V, launch the kernel, verify against naive attn.
Forward + backward; gradient check against reference.

python run.py
"""

import torch
import math
from kernel import flash_attn_fwd, flash_attn_bwd

# ── problem size ─────────────────────────────────────────────────────────────
B, H, SEQ, D = 1, 4, 512, 64

# ── block sizes for RTX 4090  (see SRAM math at top of kernel.py) ────────────
BLOCK_M, BLOCK_N = 128, 64
NUM_WARPS  = 4
NUM_STAGES = 3

scale = 1.0 / math.sqrt(D)
grid = (math.ceil(SEQ / BLOCK_M), B * H)

# ── forward ─────────────────────────────────────────────────────────────────
q = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
O = torch.empty_like(q)
L = torch.empty(B, H, SEQ, device="cuda", dtype=torch.float32)

flash_attn_fwd[grid](
    q, k, v, O, L,
    *q.stride(), *k.stride(), *v.stride(), *O.stride(),
    SEQ_Q=SEQ, SEQ_KV=SEQ, HEAD_DIM=D, scale=scale,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    num_warps=NUM_WARPS, num_stages=NUM_STAGES,
)

S_ref = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * scale
P_ref = torch.softmax(S_ref, dim=-1)
O_ref = torch.einsum("bhmn,bhnd->bhmd", P_ref, v.float()).half()

max_diff = (O - O_ref).abs().max().item()
print(f"Forward: max |O_triton - O_ref| = {max_diff:.5f}  "
      f"({'PASS' if max_diff < 0.05 else 'FAIL'})")

# ── backward ────────────────────────────────────────────────────────────────
dO = torch.randn(B, H, SEQ, D, device="cuda", dtype=torch.float16)
dQ = torch.empty_like(q)
dK = torch.zeros(B, H, SEQ, D, device="cuda", dtype=torch.float32)
dV = torch.zeros(B, H, SEQ, D, device="cuda", dtype=torch.float32)

flash_attn_bwd[grid](
    q, k, v, dO, L, dQ, dK, dV,
    *q.stride(), *k.stride(), *v.stride(),
    *dO.stride(),
    *dQ.stride(), *dK.stride(), *dV.stride(),
    SEQ_Q=SEQ, SEQ_KV=SEQ, HEAD_DIM=D, scale=scale,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    num_warps=NUM_WARPS, num_stages=NUM_STAGES,
)

# Reference backward (naive)
dP_ref = torch.einsum("bhmd,bhnd->bhmn", dO.float(), v.float())
dS_ref = P_ref * (dP_ref - (P_ref * dP_ref).sum(dim=-1, keepdim=True))
dQ_ref = torch.einsum("bhmn,bhnd->bhmd", dS_ref, k.float()) * scale
dK_ref = torch.einsum("bhmn,bhmd->bhnd", dS_ref, q.float()) * scale
dV_ref = torch.einsum("bhmn,bhmd->bhnd", P_ref, dO.float())

for name, triton_grad, ref_grad in [
    ("dQ", dQ.float(), dQ_ref),
    ("dK", dK, dK_ref),
    ("dV", dV, dV_ref),
]:
    diff = (triton_grad - ref_grad).abs().max().item()
    print(f"Backward {name}: max |triton - ref| = {diff:.5f}  "
          f"({'PASS' if diff < 0.1 else 'FAIL'})")
