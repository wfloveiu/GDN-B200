# Hints

- Before Iter 1, run `ncu` on the baseline V3 kernel to guide the first direction.
- If 3 consecutive iterations show no improvement, run `ncu` to re-profile, use WebSearch for new ideas, and review `ITERATIONS.md` for patterns. Plan before continuing.

## Hardware

- **Test machine:** B200 (Blackwell, SM 10.0, 192 GB HBM3e, 192 SMs, 228 KB shared mem/SM) — accessed remotely via Modal.
- Profile and iterate directly on B200 through `modal run run_on_modal.py`.

## B200 CUDA Optimization Guidelines

- **HBM bandwidth:** B200 has ~8 TB/s. This recurrent kernel is **memory-bandwidth-bound** — state read+write dominates. The key metric is achieved memory bandwidth as a fraction of peak.
- **L2 cache:** B200 has 96 MB L2. For small batch sizes, state may partially fit in L2. Optimize access patterns to maximize L2 hit rate.
- **cp.async / TMA:** B200 supports asynchronous memory copies. Use `cp.async` to overlap memory loads with computation. TMA (Tensor Memory Accelerator) provides even higher throughput for structured accesses.
- **Warp-level parallelism:** B200 supports up to 32 warps/SM (2048 threads). More active warps = better latency hiding for memory-bound kernels. Target ≥50% occupancy.
- **Vectorized access:** float4 (128-bit) loads/stores are critical for coalesced access. Ensure state accesses use float4 or wider when possible.
- **Register pressure:** Each SM has 65536 registers. With 128 threads/block, that's 512 regs/thread max. Keep register usage low enough to allow ≥2 blocks/SM for better occupancy.
- **Shared memory budget:** 228 KB per SM. The current V3 kernel uses NO shared memory — all register-based. Shared memory could be used for q/k broadcasting across warps, or for inter-warp reduction.
- **Compile flags:** Use `-arch=sm_100` for B200 (Blackwell). Currently using `-arch=sm_90` (Hopper) — updating may enable new instructions.

## Kernel-Specific Analysis

### Memory Traffic Analysis

For B=128, H=8, HV=16, K=128, V=128:
- **State read:** B × HV × V × K × 4 = 128 × 16 × 128 × 128 × 4 = 128 MB
- **State write:** same = 128 MB
- **Total state I/O:** 256 MB
- **Other tensors (q, k, v, gates, output):** ~2 MB (negligible)
- **Theoretical minimum time @ 8 TB/s:** 256 MB / 8 TB/s = 32 us
- **Current V3 time:** ~45 us → achieving ~71% of peak bandwidth

### Current V3 Performance Characteristics

- **Small batch (B ≤ 8):** ~5-6 us, limited by kernel launch overhead and grid underutilization (too few blocks for 192 SMs)
- **Medium batch (B=16-64):** 8-25 us, scaling approximately linearly with batch size
- **Large batch (B ≥ 128):** 45-335 us, memory-bandwidth-bound, ~71% of peak BW

### Optimization Priorities

1. **Large batch bandwidth efficiency:** Getting from 71% to ≥85% of peak BW would save ~15-20% latency
2. **Medium batch parallelism:** Better grid/block mapping to keep all 192 SMs busy
3. **Small batch launch overhead:** Already good at 5-6 us, less room to improve

## Coding Guidelines

- **Directly modify `deltanet_recurrent_cuda_v3.cu` in-place.** Do NOT create v4/v5 files. V1 and V2 are frozen.
- `CUDA_recurrent_v3.py` is the wrapper — update it if needed (e.g., change compile flags).
- Do NOT modify `bench.py`, `deltanet_recurrent_cuda_v1.cu`, `deltanet_recurrent_cuda_v2.cu`, or their wrappers.
- Use `-O3 --use_fast_math -arch=sm_100` for compilation on B200 (Blackwell).
- Do not install additional packages. Only use what's in `requirements.txt` plus PyTorch.
