# changes

**[1] KV eviction policy** — all Q-tile programs for the same head iterate over identical K/V blocks, causing redundant HBM reads.
Added `eviction_policy="evict_last"` on K/V loads so the 4090's 72 MB L2 serves subsequent programs instead of re-fetching from GDDR6X.
Q gets `evict_first` (used once per program) and O gets `evict_first` (write-only), keeping L2 free for K/V.
