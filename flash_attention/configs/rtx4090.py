"""
RTX 4090 (Ada Lovelace AD102) memory hierarchy facts
------------------------------------------------------
  SRAM per SM      : 128 KB  (usable ~100 KB after compiler overhead)
  L2 cache total   : 72 MB
  GDDR6X bandwidth : ~1 TB/s
  SM count         : 128
  Warp size        : 32 threads
  Max threads/block: 1024

Block size math (fp16, 2 bytes/element)
  One iteration holds in SRAM:
    Q tile : BLOCK_M × HEAD_DIM × 2 bytes
    K tile : BLOCK_N × HEAD_DIM × 2 bytes
    V tile : BLOCK_N × HEAD_DIM × 2 bytes
    O acc  : BLOCK_M × HEAD_DIM × 4 bytes  (fp32 accumulator)

  Example  BLOCK_M=128, BLOCK_N=64, HEAD_DIM=64:
    Q : 128×64×2 = 16 KB
    K :  64×64×2 =  8 KB
    V :  64×64×2 =  8 KB
    O : 128×64×4 = 32 KB
    ──────────────────────
    Total          64 KB   ✓ fits in 100 KB usable SRAM

  Rule of thumb:
    (BLOCK_M + 2*BLOCK_N) * HEAD_DIM * 2  +  BLOCK_M * HEAD_DIM * 4  < 100 KB
"""

# Keyed by head dimension.
# Each entry: (BLOCK_M, BLOCK_N, num_warps, num_stages)
#
#  num_warps  — threads = num_warps × 32
#               more warps → better latency hiding for memory ops
#               fewer warps → more registers per warp
#
#  num_stages — software-pipeline depth for tl.load prefetching
#               2–3 is the sweet spot on Ada; higher wastes SRAM for buffers
BLOCK_CONFIG = {
    # head_dim: (BLOCK_M, BLOCK_N, num_warps, num_stages)
    16:  (128, 64,  2, 3),
    32:  (128, 64,  4, 3),
    64:  (128, 64,  4, 3),   # most common (BERT, GPT-style)
    128: (64,  64,  4, 3),   # larger head → tile must shrink to fit SRAM
    256: (32,  32,  4, 3),   # very large head, tight fit
}


def get_block_config(head_dim: int):
    """Return (BLOCK_M, BLOCK_N, num_warps, num_stages) for this head dim."""
    if head_dim not in BLOCK_CONFIG:
        # round down to nearest known config
        known = sorted(BLOCK_CONFIG.keys())
        hd = max(d for d in known if d <= head_dim)
        return BLOCK_CONFIG[hd]
    return BLOCK_CONFIG[head_dim]
