"""
Benchmark: Flash Attention forward vs naive PyTorch.

Measures:
  - Triton kernel latency (ms)
  - Naive attention latency (ms)
  - Memory bandwidth utilisation (GB/s)
  - FLOP/s achieved

Run:
    python benchmark/bench_fwd.py
"""

import torch
import triton
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ops.flash_attn import flash_attn_forward


def naive_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    S = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * scale
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", P, v.float()).half()


def bench(fn, warmup=25, rep=100):
    """Triton's built-in benchmarking helper."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)   # returns ms


def flops(B, H, Sq, Skv, D):
    """Approximate FLOPs for one forward pass (2× for multiply-add)."""
    # QK^T: B*H * Sq * Skv * 2D
    # PV:   B*H * Sq * Skv * 2D
    return 4 * B * H * Sq * Skv * D


def hbm_bytes(B, H, Sq, Skv, D):
    """Bytes read/written from GDDR (fp16 = 2 bytes)."""
    return 2 * (B * H * (Sq + Skv + Skv + Sq) * D)   # Q + K + V + O


CONFIGS = [
    # (B, H, Sq,  Skv,  D,  label)
    (1, 8, 512,  512,  64, "512-sq"),
    (1, 8, 1024, 1024, 64, "1k-sq"),
    (1, 8, 2048, 2048, 64, "2k-sq"),
    (2, 8, 4096, 4096, 64, "4k-sq"),
]

if __name__ == "__main__":
    print(f"{'config':<12} {'triton (ms)':>12} {'naive (ms)':>12} "
          f"{'speedup':>10} {'bandwidth (GB/s)':>18} {'TFLOP/s':>10}")
    print("-" * 80)

    for B, H, Sq, Skv, D, label in CONFIGS:
        q = torch.randn(B, H, Sq,  D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, Skv, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, Skv, D, device="cuda", dtype=torch.float16)

        t_triton = bench(lambda: flash_attn_forward(q, k, v))
        t_naive  = bench(lambda: naive_attention(q, k, v))

        speedup  = t_naive / t_triton
        bw_GBs   = hbm_bytes(B, H, Sq, Skv, D) / (t_triton * 1e-3) / 1e9
        tflops   = flops(B, H, Sq, Skv, D)     / (t_triton * 1e-3) / 1e12

        print(f"{label:<12} {t_triton:>12.3f} {t_naive:>12.3f} "
              f"{speedup:>10.2f}x {bw_GBs:>18.1f} {tflops:>10.3f}")
