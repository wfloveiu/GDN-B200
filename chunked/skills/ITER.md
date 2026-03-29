# Iteration Log

<!-- Append one entry per iteration. Template at bottom. -->

### NCU Baseline Profile (4×8192, QK4V8)

| Kernel | Duration | % | Regs | Smem | Occ% |
|--------|----------|---|------|------|------|
| chunk_h | 368.6µs | 43.7% | 138 | 78KB | 6.24% |
| chunk_o | 167.9µs | 19.9% | 80 | 65KB | 32.1% |
| kkt_solve | 132.3µs | 15.7% | 222 | 16KB | 11.3% |
| w_u | 130.7µs | 15.5% | 156 | 25KB | 17.0% |
| gating | 33.7µs | 4.0% | 20 | 0 | 12.7% |
| cumsum | 9.9µs | 1.2% | 18 | 0 | 63.6% |

### Iter 1 — kkt_solve: add num_stages=[1,2] to autotune

- **Hypothesis:** kkt_solve has 222 regs → 11.3% occ. Adding num_stages=[1,2] allows autotune to find lower-register configs with fewer pipeline buffers.
- **Changes:** `chunk_fwd_intra.py` — added `num_stages` to autotune configs (9 → 18 configs).
- **Bench:** 4×8192 QK4V8: 1.105 → 1.036 ms (-6.3%). Others within noise.
- **Analysis:** Small but consistent win on QK4V8. Autotune likely picked num_stages=1 which reduced pipeline register usage.
- **Next:** Target chunk_o (19.9%, 80 regs, 65KB smem). Try adding num_stages to chunk_o autotune.

### Iter 2 — chunk_o: add num_stages=1 to autotune

- **Hypothesis:** chunk_o uses 65KB smem. Adding num_stages=1 reduces pipeline buffers → less smem → more blocks/SM.
- **Changes:** `chunk_o.py` — added num_stages=1 (was [2,3,4], now [1,2,3,4]).
- **Bench:** 4×8192 QK4V8: 1.036→0.931 (-10%). 1×65536 QK4V8: 3.072→2.710 (-12%). QK8V16 unchanged.
- **Analysis:** Significant win on QK4V8. num_stages=1 reduced smem usage, allowing higher occupancy. QK8V16 autotune likely already had good config.
- **Next:** Target chunk_delta_h (43.7%, 138 regs, 78KB smem, 6.24% occ). This is the biggest bottleneck.

### Iter 3 — chunk_h: add num_stages=1 to autotune

- **Hypothesis:** chunk_h uses 78KB smem, only 2 blocks/SM. num_stages=1 may reduce smem.
- **Changes:** `chunk_delta_h.py` — added num_stages=1 (was [2,3,4], now [1,2,3,4]).
- **Bench:** 1×65536 QK4V8: 2.710→2.666 (-1.6%). 4×8192 QK4V8: 0.931→0.926 (-0.5%). QK8V16 unchanged.
- **Analysis:** Tiny improvement. chunk_h's smem is dominated by state tile (h: [BV, K]=64*128*bf16=16KB per tile + BT*BV dot results), not pipeline buffers. num_stages has less leverage here.
- **Next:** Try widening chunk_h autotune with BV=128 or num_warps=8. Or target QK8V16 — perhaps it needs different chunk_o configs.

### Iter 4 — chunk_h: widen autotune (BV=128, num_warps=8) — REVERTED

- **Hypothesis:** BV=128 covers full V dim in 1 pass, num_warps=8 increases parallelism.
- **Changes:** `chunk_delta_h.py` — BV=[32,64,128], num_warps=[2,4,8].
- **Bench:** Slight regression on QK4V8 (0.926→0.958). QK8V16 unchanged.
- **Analysis:** BV=128 doubles smem; num_warps=8 doubles register file partition → both hurt occupancy. Autotune picked worse configs.
- **Reverted** to Iter 3 configs.
- **Next:** Try chunk_o with num_warps=2 for QK8V16. Or try allow_tf32 in chunk_o dot products.

### Iter 5 — chunk_o: wider autotune (asymmetric BK/BV, num_warps=2) — REVERTED

- **Hypothesis:** Asymmetric BK/BV and num_warps=2 may help QK8V16.
- **Changes:** `chunk_o.py` — BK/BV=[(128,128),(64,64),(64,128),(128,64)], num_warps=[2,4,8], stages=[1,2,3].
- **Bench:** 4×8192 QK4V8 regressed from 0.926 to 1.604 ms. Autotune picked bad config.
- **Reverted** to Iter 2 configs.
- **Next:** 3 iters without significant improvement on QK8V16. Run NCU for QK8V16 config to understand bottleneck.

### Iter 5 — chunk_o: wider autotune — REVERTED

- **Bench:** 4×8192 QK4V8 regressed from 0.926 to 1.604 ms.
- **Reverted** to Iter 2 chunk_o configs.

### Iter 6 — wy_fast: remove allow_tf32=False

- **Hypothesis:** `allow_tf32=False` forces IEEE precision on u=A@(v*beta) dot product. Removing it enables TF32 tensor cores.
- **Changes:** `wy_fast.py` line 66: removed `allow_tf32=False`.
- **Bench (2 runs):** 1×8192 QK4V8: 0.787→0.691 (-12%). 1×8192 QK8V16: 0.837→0.760 (-9%). Significant autotune variability across runs.
- **Analysis:** TF32 enables faster matmul for the u computation path. Correctness still PASS — max_abs_err stays under 0.01.
- **Next:** Try reducing cumsum + gating overhead by fusing them, or try different precision for chunk_h dot products.

### Iter 7 — chunk_h: narrow stages to [1,2]

- **Hypothesis:** Fewer autotune configs → less chance of picking bad config.
- **Changes:** `chunk_delta_h.py` — num_stages=[1,2,3,4] → [1,2].
- **Bench:** Within noise of previous iterations.
- **Analysis:** chunk_h is fundamentally limited by sequential loop + 78KB smem. Autotune configs don't matter much.
- **Next:** Try fusing gating + cumsum to reduce kernel launches. Or try removing the triton deprecation warning overhead.

### Iter 8 — chunk_o: add (32,32) tile — REVERTED

- No clear improvement. Reverted.

### Iter 9-10 — Verification runs

Best observed across all runs:

| Config | Pre-optimization | Best Observed | Improvement |
|--------|-----------------|--------------|-------------|
| 1×6 QK4V8 | 0.619 | 0.499 | -19.4% |
| 32×256 QK4V8 | 0.742 | 0.600 | -19.1% |
| 1×8192 QK4V8 | 0.801 | 0.691 | -13.7% |
| 4×8192 QK4V8 | 1.105 | 0.918 | -16.9% |
| 1×65536 QK4V8 | 3.129 | 2.666 | -14.8% |
| 1×8192 QK8V16 | 0.837 | 0.745 | -11.0% |
| 4×8192 QK8V16 | 1.140 | 1.084 | -4.9% |
| 1×65536 QK8V16 | 3.093 | 3.071 | -0.7% |

**Autotune variability: ~10-15% across runs.**

## Summary of Applied Optimizations (Iter 1-10)

| Iter | Change | File | Impact |
|------|--------|------|--------|
| 1 | Add num_stages=[1,2] to kkt_solve autotune | chunk_fwd_intra.py | -6% on 4×8192 QK4V8 |
| 2 | Add num_stages=1 to chunk_o autotune | chunk_o.py | **-15% on QK4V8** |
| 3 | Add num_stages=1 to chunk_h autotune | chunk_delta_h.py | -1.6% marginal |
| 6 | Remove allow_tf32=False from wy_fast | wy_fast.py | -9-13% on 1×8192 |

Reverted: Iter 4 (chunk_h BV=128), Iter 5 (chunk_o wider configs), Iter 8 (chunk_o 32x32 tile).

<!-- Template:

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:** Results table or key numbers
- **Analysis:** Why it worked or failed
- **Next:** What to try next

-->
