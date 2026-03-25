# Recurrent GDN CUDA Kernel Optimization

Optimize the CUDA kernel `deltanet_recurrent_cuda_v3.cu` for maximum decode performance on B200, measured by `bench.py` via Modal. The optimized kernel must pass correctness (compared against the PyTorch reference `ref()` in bench.py).

## Permissions & Autonomy

**You have MAXIMUM permissions on this project. You are fully autonomous.**

- You have FULL read/write/delete permissions on ALL files under `recurrent/`.
- You may modify any file at any time: `.cu`, `.py`, `bench.py`, `ncu_profile.py`, `run_on_modal.py`, `HINTS.md`, `ITERATIONS.md`, etc.
- **No user confirmation is required for ANY action** — code changes, running benchmarks, running NCU, git commits, reverting changes, etc.
- **Assume the user is NOT at the computer.** You must make ALL decisions independently: what to optimize, when to profile, when to revert, when to try a new direction.
- Do NOT ask questions. Do NOT wait for approval. Do NOT pause for feedback. Just iterate: modify code → benchmark → record → commit → repeat.
- If something fails (compilation error, correctness regression, performance regression), diagnose and fix it yourself. You have full authority to do so.

## Project Structure

```
recurrent/
  bench.py                          # Benchmark & correctness test (entry point)
  ncu_profile.py                    # NCU profiling script
  run_on_modal.py                   # Remote execution on Modal (B200 GPU)
  deltanet_recurrent_cuda_v3.cu     # CUDA kernel V3 (optimization target, best so far)
  deltanet_recurrent_cuda_v2.cu     # CUDA kernel V2 (reference, predecessor)
  deltanet_recurrent_cuda_v1.cu     # CUDA kernel V1 (original baseline)
  CUDA_recurrent_v3.py              # Python wrapper for V3 kernel
  CUDA_recurrent_v2.py              # Python wrapper for V2 kernel
  CUDA_recurrent_v1.py              # Python wrapper for V1 kernel
  Triton_recurrent.py               # Triton kernel (Qwen/FLA-based, reference)
  cutedsl_gdn.py                    # CuTe DSL kernel (reference)
  HINTS.md                          # Optimization hints & hardware info
  ITERATIONS.md                     # Iteration log
  readme.md                         # Benchmark results & optimization history
  requirements.txt                  # Python dependencies
```

## Kernel Architecture

This is a **single-step recurrent decode kernel** (T=1). Each kernel launch processes one decode step for all batch elements.

**Algorithm (per head):**
```
g = exp(-exp(A_log) * softplus(a + dt_bias))      # decay gate
beta = sigmoid(b)                                    # interpolation gate
state_decayed = g * state_old                        # decay old state
old_v = k @ state_decayed                            # project old state
new_v = beta * v + (1-beta) * old_v                  # interpolate
state_new = state_decayed - k^T @ old_v + k^T @ new_v  # delta update
output = scale * q @ state_new                       # query output
```

**State:** `[B, HV, V, K]` float32 — the dominant memory cost. Each head's state is 128×128×4 = 64 KB.

**Current V3 design:**
- Grid: `(V/(NUM_WARPS*BV), B*HV)` = `(8, B*HV)`
- Block: 128 threads (4 warps × 32 lanes)
- Each warp: BV=4 v-rows, KPT=4 floats/thread
- float4 vectorized state load/store (128-bit)
- uint2 vectorized q/k/v bf16 loads (64-bit)
- No shared memory, no `__syncthreads` — warps fully independent
- Pure register-based computation

## Setup

1. **Understand the code:** Read `deltanet_recurrent_cuda_v3.cu`, `bench.py`, `HINTS.md`, and this file.
2. **Verify environment:** Run `modal run run_on_modal.py --which bench`. Check correctness and baseline numbers.

## Benchmark

- **Run benchmark:** `modal run run_on_modal.py --which bench`
- **Run NCU profiling:** `modal run run_on_modal.py --which ncu --batch-size 128`
- **Correctness:** Compare against `ref()` in bench.py — max absolute error should be small (< 0.1 for output, < 0.01 for state)
- **Performance metric:** Latency in microseconds (us) across batch sizes [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
- **Input shapes:** `H=8, HV=16, K=128, V=128` (GQA: 2 v-heads per k-head)
- **Hardware:** B200 (Blackwell, SM 10.0) — accessed remotely via Modal

## Optimization Target

**Directly modify `deltanet_recurrent_cuda_v3.cu` in-place.** Do NOT create new v4/v5 files — always overwrite V3. V1 and V2 are frozen references and must not be changed.

### Performance Goal

**CUDA V3 must beat the Triton kernel (`Qwen(Triton)`) at ALL batch sizes [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024].** This is the success criterion. Current status from bench.py:
- B ≤ 64: V3 is already **faster** than Triton (3-7× faster)
- B ≥ 128: V3 is **slower** than Triton (3-5% slower) — **this must be fixed**

The goal is to make V3 faster than Triton at every single batch size.

**Key bottleneck:** This is a **memory-bandwidth-bound** kernel. The state tensor dominates: reading + writing `B * HV * V * K * 4` bytes per step. On B200 with ~8 TB/s bandwidth, the theoretical minimum latency at B=128 is ~32 us. Current V3 achieves ~45 us — there is room to improve.

**Optimization directions to explore:**
1. **Reduce memory traffic** — avoid redundant state reads/writes, fuse gating computation
2. **Improve memory access patterns** — better coalescing, alignment, prefetching
3. **Increase parallelism** — more warps, better grid mapping, warp specialization
4. **Use hardware features** — TMA, cp.async, warp-level matrix ops
5. **Algorithmic improvements** — restructure computation to reduce total work

## Iteration Protocol

**Total iterations: N = 50.** Execute all 50 iterations autonomously without asking for user confirmation.

Every modification to `.cu` code followed by a benchmark run counts as one iteration. Number iterations sequentially (1, 2, 3, ..., 20).

**Do NOT start the next iteration until ALL steps below are completed:**

1. **Run benchmark** — `modal run run_on_modal.py --which bench` and record the results.
2. **Update `ITERATIONS.md`** — append a new entry with hypothesis, changes, bench results, analysis, and next steps.
3. **Git commit** — `[iter N] Short description of optimization direction`.

### Decision Rules

- **Before Iter 1:** Run NCU to establish a baseline profile and identify the first optimization target.
- **If 3 consecutive iterations show no improvement:** Stop and run NCU to re-profile. Use web search for new ideas. Review `ITERATIONS.md` for patterns. Plan before continuing.
- **If a change breaks correctness or compilation:** Revert immediately, record the failure in `ITERATIONS.md`, and try a different approach.
- **Focus on large batch sizes (128-1024)** where the kernel is slowest relative to theoretical minimum, but don't regress small batch sizes.
