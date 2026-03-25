# Chunked GDN Kernel Optimization

Optimize the Triton kernels in `FLA/` for maximum performance, measured by `bench.py` on a remote B200 GPU via Modal. The optimized kernels must pass the correctness test (max_abs_err < 1.0).

## Project Structure

```
chunked/
  bench.py              # Benchmark & correctness test (entry point)
  ncu_profile.py        # NCU profiling script (used by run_on_modal.py)
  run_on_modal.py       # Remote execution on Modal (B200 GPU)
  FLA/                  # Triton kernels (optimization target)
    chunk.py            #   Main orchestrator: chunk_gated_delta_rule
    fused_gdn_gating.py #   Gate computation (g, beta)
    cumsum.py           #   Chunk-local cumulative sum
    chunk_scaled_dot_kkt.py # beta * K * K^T
    solve_tril.py       #   Lower-triangular inverse (I+A)^-1
    wy_fast.py          #   Recompute w, u (WY representation)
    chunk_delta_h.py    #   State recurrence (typically the bottleneck)
    chunk_o.py          #   Output: o = scale * (q @ h) + attention * v
    utils.py            #   Helpers, device flags, autotune cache
  HINTS.md              # Optimization hints & stall rules
  ITERATIONS.md         # Iteration log
  README.md             # Baseline performance numbers
  requirements.txt      # Python dependencies
```

## Setup

1. **Understand the code:** Read all files in `FLA/`, `bench.py`, `HINTS.md`, and this file.
2. **Verify environment:** Run `modal run run_on_modal.py --which bench`. Expected output: correctness test `PASS` and benchmark numbers printed. If it fails, diagnose and fix before proceeding. Then `git add -A && git commit -m "[baseline] Verify environment and baseline performance"`.

## Benchmark

- **Run benchmark:** `modal run run_on_modal.py --which bench`
- **Run NCU profiling:** `modal run run_on_modal.py --which ncu`
- **Correctness:** `max_abs_err < 1.0` for both output and state (compared to the Python baseline `baseline_run` in bench.py)
- **Performance metric:** Time (ms) and Tokens/s (M) for multiple configs
- **Input shapes:** `head_size=128`, two head configurations:
  - `num_q_heads=4, num_k_heads=4, num_v_heads=8` (GQA: 2 v-heads per q/k-head)
  - `num_q_heads=8, num_k_heads=8, num_v_heads=16` (GQA: 2 v-heads per q/k-head)
- **Hardware:** B200 (Blackwell, SM 10.0) — accessed remotely via Modal

## Optimization

- Run `modal run run_on_modal.py --which bench` to measure performance.
- Run `modal run run_on_modal.py --which ncu` to profile and identify bottlenecks — do not optimize blindly.
- Leverage all available information: `HINTS.md`, prior attempts in `ITERATIONS.md`, web search, etc.
- All optimizations go into `FLA/` files only. Do NOT modify `bench.py`.
- Follow stall rules defined in `HINTS.md`.

### Iteration Protocol

Every modification to `FLA/` code followed by a benchmark run counts as one iteration — regardless of whether the result is an improvement, regression, or failure. Number iterations sequentially (1, 2, 3, ...).

**Do NOT start the next iteration until ALL steps below are completed:**

1. **Run benchmark** — `modal run run_on_modal.py --which bench` and record the results.
2. **Update `ITERATIONS.md`** — append a new entry with hypothesis, changes, bench results, analysis, and next steps.
3. **Git commit** — `[iter N] Short description of optimization direction`.
