# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed -->

| Iter | Title | Config | Runtime | Tokens/s (M) | Status |
|------|-------|--------|---------|-------------|--------|
| 0 | Baseline | 4×8192 QK4V8 | 1.380 ms | 23.75 | baseline |
| 0 | Baseline | 4×8192 QK8V16 | 1.226 ms | 26.73 | baseline |
| 0 | Baseline | 1×65536 QK4V8 | 4.301 ms | 15.24 | baseline |
| 0 | Baseline | 1×65536 QK8V16 | 3.569 ms | 18.36 | baseline |

## NCU Baseline Profile (4×8192, QK4V8)

| Kernel | Duration | % of Total | Limiter |
|--------|----------|-----------|---------|
| merge_16x16_to_64x64_inverse | 381.9 µs | 33.3% | 255 regs → 12.5% occ → latency-bound |
| chunk_gated_delta_rule_fwd_h | 360.4 µs | 31.4% | 78KB shmem + small grid → 6.3% occ |
| chunk_fwd_kernel_o | 160.9 µs | 14.0% | 80 regs + shmem → 32.3% occ |
| recompute_w_u_fwd | 123.4 µs | 10.8% | 138 regs → 16.9% occ |
| chunk_scaled_dot_kkt_fwd | 53.1 µs | 4.6% | 145 regs → 17.2% occ |
| Others | 67.7 µs | 5.9% | — |
| **Total** | **~1,147 µs** | **100%** | — |

## Iterations

### Iter 1 — Split solve_tril 64×64 into two-phase (reduce register pressure)

- **Hypothesis:** The monolithic `merge_16x16_to_64x64_inverse_kernel` uses 255 regs/thread → 12.5% occupancy → latency-bound. Splitting into phase1 (solve two 32×32 diagonal blocks) + phase2 (merge off-diagonal) reduces peak live registers.
- **Changes:** `solve_tril.py` — added `solve_tril_64x64_phase1_kernel` and `merge_32x32_to_64x64_inverse_kernel`. Phase1 solves top-left and bottom-right 32×32 independently. Phase2 reads the diagonal inverses from Ai and computes only the 4 off-diagonal 16×16 blocks.
- **Bench:**
  - Correct: True (PASS)
  - Results:

| Config | Baseline | Iter 1 | Speedup |
|--------|----------|--------|---------|
| 1×8192 QK4V8 | 0.529 ms | 0.409 ms | 1.29× |
| 4×8192 QK4V8 | 1.380 ms | 0.646 ms | 2.14× |
| 1×65536 QK4V8 | 4.301 ms | 2.539 ms | 1.69× |
| 1×8192 QK8V16 | 0.486 ms | 0.423 ms | 1.15× |
| 4×8192 QK8V16 | 1.226 ms | 0.934 ms | 1.31× |
| 1×65536 QK8V16 | 3.569 ms | 2.973 ms | 1.20× |

- **Analysis:** Huge win, especially on 4×8192 QK4V8 (2.14×). Register pressure reduction worked as expected. The two-phase approach trades 1 kernel launch for much better occupancy.
- **Next:** Target the second bottleneck: `chunk_gated_delta_rule_fwd_kernel_h` (31.4% of time, 6.3% occupancy, 78KB shmem, grid < SMs).

### Iter 2 — Expand autotune configs for h, o, and wy kernels

- **Hypothesis:** Wider autotune search (BV=128, num_warps=8, num_stages=1 for h kernel; asymmetric BK/BV for o kernel) may find better configs on B200.
- **Changes:** `chunk_delta_h.py` — added BV=128, num_warps=8, num_stages=1. `chunk_o.py` — added BK/BV asymmetric configs. `wy_fast.py` — added num_stages=1.
- **Bench:**
  - Correct: True (PASS)
  - Results:

| Config | Iter 1 | Iter 2 | Δ |
|--------|--------|--------|---|
| 1×8192 QK4V8 | 0.409 ms | 0.414 ms | ~same |
| 4×8192 QK4V8 | 0.646 ms | 0.640 ms | ~same |
| 1×65536 QK4V8 | 2.539 ms | 2.542 ms | ~same |
| 1×8192 QK8V16 | 0.423 ms | 0.421 ms | ~same |
| 4×8192 QK8V16 | 0.934 ms | 0.922 ms | ~same |
| 1×65536 QK8V16 | 2.973 ms | 2.949 ms | ~same |

- **Analysis:** No significant change. Autotune likely already converged on the best num_warps/stages. The bottleneck is likely structural (shmem usage, grid size) rather than tuning parameters.
- **Next:** Run NCU post-iter1 to get updated profile. Focus on structural changes to h kernel.

<!-- Template — copy for each new iteration:

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next
-->
