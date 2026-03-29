# Chunked GDN Kernel Optimization

Optimize the Triton kernels in `MY/` for maximum performance, measured by `bench.py` on a remote B200 GPU via Modal. The optimized kernels must pass the official correctness test (atol=0.01, rtol=0.01).

## Permissions

**You have FULL read/write permissions on ALL files under `chunked/`.** This includes `MY/`, `bench.py`, `ncu_profile.py`, `run_on_modal.py`, `skills/`, and any other files. You may create, modify, or delete any file as needed. **No user confirmation is required** — iterate autonomously until completion.

## Project Structure

```
chunked/
  bench.py              # Benchmark (entry point): MY vs SGLang vs FlashInfer
  ncu_profile.py        # NCU profiling script
  run_on_modal.py       # Remote execution on Modal (B200 GPU)
  MY/                   # Triton kernels (optimization target)
    chunk.py            #   Main orchestrator: chunk_gated_delta_rule
    fused_gdn_gating.py #   Gate computation (g, beta)
    cumsum.py           #   Chunk-local cumulative sum
    chunk_fwd_intra.py  #   Fused kkt + solve_tril + recompute_w_u
    wy_fast.py          #   Recompute w, u (WY representation)
    chunk_delta_h.py    #   State recurrence (the bottleneck, ~45-60% of total time)
    chunk_o.py          #   Output: o = scale * (q @ h) + attention * v
    utils.py            #   Helpers, device flags, autotune cache
    tmp/                #   Unused separate kernels (kkt, solve_tril) for reference
  sglang_chunked_gdn/   # SGLang's GDN implementation (comparison baseline)
  flashinfer/            # FlashInfer CuTe-DSL implementation (comparison baseline)
  skills/               # This directory
    TASK.md             # This file
    HINTS.md            # Optimization hints & rules
    ITER.md             # Iteration log
  ITERATIONS.md         # Historical optimization log (completed iterations)
  MASK_FIX_FUSED_KKT+SOLVE-TRIL.md  # Padding fix analysis
  FLASHINFER_VS_MY_ANALYSIS.md       # Performance comparison analysis
  requirements.txt      # Python dependencies
```

## Current State

MY/ code already has these optimizations applied (see ITERATIONS.md for details):
1. **GQA support (Hk != H)** — all kernels handle num_k_heads != num_v_heads natively
2. **Fused kkt + solve_tril** — `chunk_fwd_intra.py` fuses 3 separate kernels into 1
3. **Off-diagonal mask fix** — prevents exp overflow in padding regions
4. **A matrix torch.zeros** — prevents garbage values in padding
5. **BK=BV=128 for wy_fast** — eliminates inner loop when K=V=128
6. **Reduced autotune stages** — wy_fast uses num_stages=[1,2], num_warps=[4,8]

## Benchmark

- **Run MY benchmark:** `modal run run_on_modal.py --which opt` (correctness + performance, MY only)
- **Run full benchmark:** `modal run run_on_modal.py --which bench` (MY vs SGLang vs FlashInfer)
- **Run NCU profiling:** `modal run run_on_modal.py --which ncu`
- **Correctness:** Official test uses atol=0.01, rtol=0.01, 100% elements must pass
- **Performance metric:** Time (ms), cold cache, compared against SGLang and FlashInfer
- **Input shapes:** `head_size=128`, two head configurations:
  - `num_q_heads=4, num_k_heads=4, num_v_heads=8` (GQA: 2 v-heads per q/k-head)
  - `num_q_heads=8, num_k_heads=8, num_v_heads=16` (GQA: 2 v-heads per q/k-head)
- **Hardware:** B200 (Blackwell, SM 10.0) — accessed remotely via Modal

### Current Performance (Cold Cache, B200)

| Config | MY (ms) | SGLang (ms) | FlashInfer (ms) | MY/SG | MY/FI |
|--------|---------|-------------|-----------------|-------|-------|
| 1×6 QK4V8 | 0.612 | 0.735 | 0.158 | 1.20× | 0.26× |
| 32×256 QK4V8 | 0.709 | 0.907 | 0.216 | 1.28× | 0.30× |
| 1×8192 QK4V8 | 0.780 | 0.968 | 0.869 | 1.24× | 1.11× |
| 4×8192 QK4V8 | 0.981 | 1.202 | 0.926 | 1.23× | 0.94× |
| 1×65536 QK4V8 | 2.809 | 3.374 | 6.116 | 1.20× | 2.18× |
| 1×8192 QK8V16 | 0.821 | 1.037 | 0.898 | 1.26× | 1.09× |
| 4×8192 QK8V16 | 1.172 | 1.528 | 0.910 | 1.30× | 0.78× |
| 1×65536 QK8V16 | 3.110 | 4.034 | 6.217 | 1.30× | 2.00× |

### Remaining Bottleneck

`chunk_gated_delta_rule_fwd_kernel_h` (state recurrence) accounts for ~45-60% of total time:
- Sequential time-loop over chunks (cannot be parallelized)
- 78KB shared memory per block
- 142 registers/thread
- Limited by memory bandwidth (state is [B, NT, H, V, K] in bf16)

## Optimization

- Run `modal run run_on_modal.py --which bench` to measure performance.
- Run `modal run run_on_modal.py --which ncu` to profile and identify bottlenecks.
- Leverage: `skills/HINTS.md`, `ITERATIONS.md`, `FLASHINFER_VS_MY_ANALYSIS.md`, web search.
- Follow stall rules defined in `skills/HINTS.md`.

### Iteration Protocol

**Total iterations: N = 20.** Execute all 20 iterations autonomously without asking for user confirmation.

Every modification to `MY/` code followed by a benchmark run counts as one iteration. Number iterations sequentially in `skills/ITER.md`.

**Do NOT start the next iteration until ALL steps below are completed:**

1. **Run benchmark** — `modal run run_on_modal.py --which opt` and record the results.
2. **Update `skills/ITER.md`** — append a new entry with hypothesis, changes, bench results, analysis, and next steps.
3. **Git commit** — `[iter N] Short description of optimization direction`.

### Decision Rules

- **Before Iter 1:** Run NCU to establish current profile and identify the first optimization target.
- **If 3 consecutive iterations show no improvement:** Stop and run NCU to re-profile. Use web search for new ideas. Review `ITER.md` for patterns. Plan before continuing.
- **If a change breaks correctness:** Revert immediately, record the failure in `ITER.md`, and try a different approach.
- **Prioritize the biggest bottleneck first** — `chunk_delta_h.py` (state recurrence kernel).
