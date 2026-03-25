# Iteration Log

## Baseline (sm_100, original V3)

| B | Qwen(Triton) | V3 | V3 vs Qwen |
|---|-------------|-----|-----------|
| 1 | 29.22 | 5.15 | 5.7× faster |
| 32 | 28.02 | 12.29 | 2.3× faster |
| 128 | 43.11 | 43.05 | 1.0× faster |
| 512 | 160.08 | 160.95 | 0.5% slower |
| 1024 | 317.62 | 324.00 | 2.0% slower |

NCU profile (B=512, H=8, HV=16): DRAM BW=4.49 TB/s (55% of peak 8 TB/s), 46 regs/thread, 55% warp occupancy, grid=(8, 8192), block=128.

## Iterations

### Iter 1 — BV=8 v-rows per warp (REGRESSION)

- **Hypothesis:** More v-rows per warp amortizes q/k load and gate computation overhead.
- **Changes:** BV: 4→8, added `__launch_bounds__(128, 2)`, simplified V load.
- **Bench:** B=512: 183.70 us (was 160.95), B=1024: 362.94 (was 324.00). **13% regression across large B.**
- **Analysis:** BV=8 doubles state register usage (32 regs → 64 regs) hurting occupancy. Grid.x halves (8→4) reducing block-level parallelism. For memory-bound kernels, more independent blocks beats more work per block.
- **Next:** Revert BV=4. Try more warps or different parallelization strategy.

### Iter 2 — 8 warps per block (REGRESSION)

- **Hypothesis:** 8 warps (256 threads/block) with BV=4 gives same grid.x but more warps per SM for latency hiding.
- **Changes:** NUM_WARPS: 4→8, BLOCK_SIZE: 128→256, `__launch_bounds__(256, 1)`.
- **Bench:** B=512: 171.96 us (was 160.95), B=1024: 338.89 (was 324.00). **7% regression.**
- **Analysis:** Grid.x halved (8→4), so total blocks halved = less parallelism. Even though more warps/block, the reduced grid kills performance at high B. Block-level parallelism > warp-level for memory-bound kernels.
- **Next:** Keep original config (4 warps, BV=4, grid.x=8). Focus on reducing compute overhead or improving memory access patterns.

### Iter 3 — Code cleanup + remove boundary checks

- **Hypothesis:** Remove unnecessary boundary checks (V=128 is always divisible by BV_TOTAL=16), simplify pointer math, fuse store+output loop.
- **Changes:** Removed if-guards on state load/V load (V always divisible), simplified pointer arithmetic, fused delta+store+output into single loop.
- **Bench:**

| B | Baseline V3 | Iter 3 | Qwen | Status |
|---|-----------|--------|------|--------|
| 1 | 5.15 | 6.05 | 30.53 | ✓ faster |
| 8 | 6.15 | 5.62 | 30.19 | ✓ improved |
| 16 | 8.20 | 7.49 | 32.82 | ✓ improved |
| 32 | 12.29 | 10.25 | 28.82 | ✓ improved |
| 128 | 43.05 | 43.76 | 43.11 | ~same |
| 512 | 160.95 | 163.70 | 160.50 | 2% slower than Qwen |
| 1024 | 324.00 | 324.89 | 319.47 | 1.7% slower than Qwen |

- **Analysis:** Slight improvement at small B (5-20% faster at B=8-32) from removing branches. Large B unchanged — the bottleneck is DRAM bandwidth, not compute overhead. The 2% gap at B≥512 is within measurement noise but consistent.
- **Next:** The gap at large B is ~2%. Need to improve DRAM bandwidth utilization from 55% to ~58%+. Try: (1) cp.async prefetch, (2) different state memory layout, (3) reducing L1/L2 cache thrashing.
