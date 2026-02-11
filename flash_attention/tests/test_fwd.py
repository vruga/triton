"""
Correctness test: compare Triton Flash Attention against naive PyTorch.

Run:
    python -m pytest tests/test_fwd.py -v
or:
    python tests/test_fwd.py
"""

import torch
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ops.flash_attn import flash_attn_forward


# ── reference (naive, exact) ──────────────────────────────────────────────────
def naive_attention(q, k, v):
    """Standard scaled dot-product attention.  O(N^2) memory."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    # promote to fp32 for numerical accuracy in the reference
    qf = q.float()
    kf = k.float()
    vf = v.float()
    S  = torch.einsum("bhmd,bhnd->bhmn", qf, kf) * scale  # (B,H,Sq,Skv)
    P  = torch.softmax(S, dim=-1)
    O  = torch.einsum("bhmn,bhnd->bhmd", P, vf)            # (B,H,Sq,D)
    return O.half()


# ── test cases ────────────────────────────────────────────────────────────────
CASES = [
    # (B, H, Sq,  Skv, D)
    (1, 1, 64,  64,  64),
    (2, 4, 128, 128, 64),
    (1, 8, 512, 512, 64),
    (2, 4, 256, 512, 128),   # non-square seq lens
]

def run_test(B, H, Sq, Skv, D, atol=1e-2, rtol=1e-2):
    q = torch.randn(B, H, Sq,  D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, Skv, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, Skv, D, device="cuda", dtype=torch.float16)

    ref  = naive_attention(q, k, v)
    out, _ = flash_attn_forward(q, k, v)

    max_diff = (ref - out).abs().max().item()
    match    = torch.allclose(ref, out, atol=atol, rtol=rtol)
    print(f"  B={B} H={H} Sq={Sq} Skv={Skv} D={D}  |  "
          f"max_diff={max_diff:.5f}  {'PASS' if match else 'FAIL'}")
    return match


if __name__ == "__main__":
    all_pass = True
    print("Flash Attention forward — correctness tests")
    print("=" * 60)
    for case in CASES:
        ok = run_test(*case)
        all_pass = all_pass and ok
    print("=" * 60)
    print("ALL PASS" if all_pass else "SOME TESTS FAILED")
    sys.exit(0 if all_pass else 1)
