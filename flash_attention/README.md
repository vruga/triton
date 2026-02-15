# changes

**[1] KV eviction policy** — all Q-tile programs for the same head iterate over identical K/V blocks, causing redundant HBM reads.
Added `eviction_policy="evict_last"` on K/V loads so the 4090's 72 MB L2 serves subsequent programs instead of re-fetching from GDDR6X.
Q gets `evict_first` (used once per program) and O gets `evict_first` (write-only), keeping L2 free for K/V.
L2 → SM: ~200 cycles
earluer ~600 cycles

---

## run — B=1 H=4 SEQ=512 D=64

| | |
|---|---|
| correctness | max \|O_triton − O_ref\| = 0.00024 ✓ |
| latency | 0.0291 ms / run (100 iters) |
| grid | (4 Q-tiles, 4 heads) = 16 programs |
| BLOCK_M / BLOCK_N | 128 / 64 |
| SRAM per CTA | ~64 KB (Q 16 KB + K 8 KB + V 8 KB + O_acc 32 KB) |
| num_warps / num_stages | 4 / 3 |
| device | RTX 4090, GDDR6X 1 TB/s, L2 72 MB |
