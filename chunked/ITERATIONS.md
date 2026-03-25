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

### Iter 3 — Tune h kernel (maxnreg + stages) and bf16 A matrix

- **Hypothesis:** (a) `maxnreg=128` on h kernel could improve occupancy by trading register spill; (b) bfloat16 A matrix halves memory for KKT+solve_tril pipeline; (c) `num_stages=1` reduces shmem.
- **Changes:** Tried `maxnreg` (reverted — regression), tried bf16 A matrix (reverted — mixed, some configs regressed). Kept autotune config with `num_stages=[2,3,4]` and `empty` instead of `zeros` for final_state.
- **Bench:**
  - Correct: True (PASS)
  - Results: ~same as Iter 1/2 after reverts. `maxnreg` and bf16 A both caused regressions on some configs.
- **Analysis:** The h kernel bottleneck is structural — limited by sequential time-loop and state size, not tuning params. bf16 A hurts precision-sensitive autotune paths. `new_empty` for final_state saves ~5µs (trivial).
- **Next:** Focus on reducing total kernel count by fusing operations, or try `allow_tf32=True` in wy_fast dot products for speed.

### Iter 4 — Enable TF32 in wy_fast dot product

- **Hypothesis:** The wy_fast kernel uses `allow_tf32=False`, forcing IEEE precision. Removing this constraint lets the compiler use TF32 tensor cores on B200, which are ~2× faster for matrix multiply.
- **Changes:** `wy_fast.py` — removed `allow_tf32=False` from `tl.dot(b_A, b_vb)`.
- **Bench:**
  - Correct: True (PASS, state max_abs_err=0.005 still well under 1.0)
  - Results: ~same as Iter 1/2 (within noise). TF32 didn't measurably change wy_fast timing since it's not the bottleneck.
- **Analysis:** The wy_fast kernel is only ~97µs (11% of total). TF32 may save a few µs but it's within measurement noise. However, no regression, so keep the change.
- **Next:** Try fusing `fused_gdn_gating` inline or investigate chunk_size=32 to reduce solve_tril overhead.

### Iter 5 — Attempted gdn_gating kernel rewrite and reverted

- **Hypothesis:** The gdn_gating kernel launches 32768 tiny blocks (1 warp each). Vectorizing across tokens with larger blocks should reduce launch overhead and improve utilization.
- **Changes:** Rewrote gdn_gating with BLOCK_T=1024, loop over heads. Reverted — sequential head loop was worse than original's embarrassingly parallel approach.
- **Bench:** Slight regression on some configs. Reverted to original.
- **Analysis:** The gating kernel is only 33µs (4% of total). Even 2× speedup saves only 16µs. Not worth the complexity. Focus on bigger targets.
- **Next:** Try `input_precision='tf32'` in solve_tril dot products. Profile the o kernel more carefully.

### Iter 6 — Enable TF32 precision in solve_tril autotune

- **Hypothesis:** solve_tril's DOT_PRECISION was fixed to 'ieee' on B200 due to default env var. Adding 'tf32' to the autotune list lets the compiler use TF32 tensor cores for the 12+ dot products in the merge kernel.
- **Changes:** `solve_tril.py` — changed `FLA_TRIL_PRECISION` default from 'ieee' to 'tf32', making autotune search over both 'ieee' and 'tf32' on TMA-capable GPUs.
- **Bench:**
  - Correct: True (PASS, max_abs_err=0.007812 — slightly higher but well under 1.0)
  - Results:

| Config | Iter 4 | Iter 6 | Δ |
|--------|--------|--------|---|
| 1×8192 QK4V8 | 0.408 | 0.395 | -3.2% |
| 4×8192 QK4V8 | 0.638 | 0.610 | -4.4% |
| 1×65536 QK4V8 | 2.552 | 2.492 | -2.4% |
| 1×8192 QK8V16 | 0.422 | 0.409 | -3.1% |
| 4×8192 QK8V16 | 0.915 | 0.895 | -2.2% |
| 1×65536 QK8V16 | 2.938 | 2.888 | -1.7% |

- **Analysis:** Consistent 2-4% improvement across all configs. TF32 allows the tensor cores to work faster on 16×16 matrix multiplies. The slight precision loss (0.004→0.008 max abs err) is acceptable.
- **Next:** Look at the chunk_kkt kernel (47µs) — try adding TF32 autotune there too. Investigate if the `o` kernel can be improved.

### Iter 7 — Refine kkt autotune configs

- **Hypothesis:** Narrowing kkt autotune to BK=[64,128], adding num_stages=1 might find better config on B200.
- **Changes:** `chunk_scaled_dot_kkt.py` — narrowed autotune search space.
- **Bench:** ~same as Iter 6 (within noise). kkt is only 47µs (5% of total).
- **Analysis:** Diminishing returns on small kernels. The h kernel at 55% of total time is the hard bottleneck.
- **Next:** Try reducing `o` kernel time by improving its memory access pattern, or try persistent kernel approach for h.

### Iter 9 — Fused kkt + solve_tril kernel (from upstream FLA)

- **Hypothesis:** Fusing chunk_scaled_dot_kkt + solve_tril into one kernel avoids HBM round-trip for intermediate A matrix. Upstream FLA has this optimization.
- **Changes:** New `chunk_fwd_intra.py` with `chunk_gated_delta_rule_fwd_kkt_solve_kernel`. Updated `chunk.py` to use fused path.
- **Bench:**
  - Correct: True (PASS)
  - Results:

| Config | Iter 6 | Iter 9 | Δ |
|--------|--------|--------|---|
| 1×8192 QK4V8 | 0.395 | 0.362 | **-8.4%** |
| 4×8192 QK4V8 | 0.610 | 0.620 | ~same |
| 1×65536 QK4V8 | 2.492 | 2.763 | +10.9% regression |
| 1×8192 QK8V16 | 0.409 | 0.399 | -2.4% |
| 4×8192 QK8V16 | 0.895 | 0.903 | ~same |
| 1×65536 QK8V16 | 2.888 | 2.945 | ~same |

- **Analysis:** Mixed results. Short sequences benefit from fewer kernel launches and less HBM traffic. Long sequences (65536) regress — possibly due to higher register pressure in the fused kernel (holds 10+ BC×BC matrices simultaneously). The fused kernel replaces 3 separate kernels (kkt + phase1 + merge) with 1, but the monolithic approach may not autotune as well.
- **Next:** Keep fused kernel for now. Try improving the long-sequence performance, perhaps by tuning the fused kernel autotune configs or falling back to separate kernels for long sequences.

### Iter 10 — Widen fused kernel autotune (BK=32, num_warps=1)

- **Hypothesis:** The upstream FLA uses BK=32 and num_warps=1 for the fused kernel. Adding these to autotune may find better configs, especially for long sequences.
- **Changes:** `chunk_fwd_intra.py` — added BK=32 and num_warps=1 to autotune search.
- **Bench:**
  - Correct: True (PASS)
  - Results:

| Config | Baseline | Iter 6 | **Iter 10** | **Total Speedup** |
|--------|----------|--------|-------------|-------------------|
| 1×8192 QK4V8 | 0.529 | 0.395 | **0.347** | **1.52×** |
| 4×8192 QK4V8 | 1.380 | 0.610 | **0.542** | **2.55×** |
| 1×65536 QK4V8 | 4.301 | 2.492 | **2.477** | **1.74×** |
| 1×8192 QK8V16 | 0.486 | 0.409 | **0.370** | **1.31×** |
| 4×8192 QK8V16 | 1.226 | 0.895 | **0.789** | **1.55×** |
| 1×65536 QK8V16 | 3.569 | 2.888 | **2.712** | **1.32×** |

- **Analysis:** Significant improvement across all configs! The fused kernel with wider autotune now matches or beats the separate-kernel approach even on long sequences. The autotune likely chose smaller BK/warps for better register efficiency. This is our best result: **2.55× on 4×8192 QK4V8**.
- **Next:** Continue optimizing. Run NCU to see new profile. Target remaining bottlenecks.

### Iter 11 — BK=BV=128 for recompute_w_u_fwd

- **Hypothesis:** With K=V=128, using BK=BV=128 eliminates inner loops (1 iteration vs 2 with BK=64). This reduces loop overhead and register pressure from keeping multiple tile pointers alive.
- **Changes:** `wy_fast.py` — changed BK=min(K,128), BV=min(V,128).
- **Bench:**
  - Correct: True (PASS)
  - Best results yet: **2.67× on 4×8192 QK4V8**, **1.87× on 1×65536 QK4V8**
- **Analysis:** Eliminating inner loops dramatically reduced the w_u kernel time. Single-iteration K/V loops reduce register spill and branch overhead.
- **Next:** Continue optimizing chunk_fwd_kernel_o and explore other improvements.

### Iter 12 — Wider chunk_o autotune (reverted)

- **Hypothesis:** Adding num_stages=1 and num_warps=2 to chunk_o autotune may find better configs.
- **Changes:** Added wider search space. Reverted — regression on QK4V8 configs (autotune picked worse configs).
- **Bench:** 0.366/0.621 vs 0.311/0.541 on QK4V8 1×8192/4×8192. Reverted.
- **Analysis:** The previous autotune configs were already well-tuned for chunk_o. Adding too many options can cause autotune to pick locally suboptimal configs. Stick with previous settings.
- **Next:** Try other approaches — investigate if we can reduce memory traffic by using smaller intermediate types.

### Iter 8 — Forced h kernel BV=64, stages=1 (reverted)

- **Hypothesis:** BV=64 with stages=1 reduces shmem. Combined with 4 warps might help occupancy.
- **Changes:** Fixed h kernel config to BV=64, num_stages=1, num_warps=4.
- **Bench:** MAJOR REGRESSION (1.155ms vs 0.610ms). BV=64 halves grid to 64 blocks → even worse occupancy. stages=1 kills pipelining. Reverted immediately.
- **Analysis:** BV=32 is strongly preferred for this kernel because it provides 2× more blocks (128 vs 64). The grid dimension cdiv(V, BV) directly impacts parallelism. stages≥2 is needed for pipelining.
- **Next:** The h kernel is at its limit with current architecture. Look at reducing the `o` kernel or trying to eliminate intermediate allocations.

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
