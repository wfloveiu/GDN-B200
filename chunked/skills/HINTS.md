# Hints

- Run MY-only test: `modal run run_on_modal.py --which opt`
- Before Iter 1, run `ncu` on the current kernel to guide the first direction.
- If 3 consecutive iterations show no improvement, run `ncu` to re-profile, use WebSearch for new ideas, and review `skills/ITER.md` for patterns. Plan before continuing.

## Hardware

- **Test machine:** B200 (Blackwell, SM 10.0, 192 GB HBM3e, 192 SMs, 228 KB shared mem/SM) — accessed remotely via Modal.
- Profile and iterate directly on B200 through `modal run run_on_modal.py`.

## B200 Optimization Guidelines

- **Prefer TMA (Tensor Memory Accelerator):** B200 supports TMA. Use `tl.make_tensor_descriptor` / TMA loads where possible — they leverage B200's larger L2 and higher bandwidth.
- **Target larger tile sizes:** B200 has 192 SMs. Choose tile configs that scale well to 192 SMs.
- **Maximize warp-level parallelism:** B200 supports up to 32 warps/SM. Prefer `num_warps=4` or `num_warps=8` configs.
- **Shared memory budget:** B200 has ~228 KB shared mem per SM. Do not push beyond this limit.
- **Autotune configs should cover B200 sweet spots:** Include `BV=64`, `BV=128` and `num_warps` in {2, 4, 8} in autotune search spaces.
- **HBM bandwidth:** B200 has ~8 TB/s. Memory-bound kernels benefit automatically, so focus optimization effort on **compute-bound** or **latency-bound** kernels.
- **safe_dot for Blackwell:** On SM100 (Blackwell), there is a known Triton compiler bug where the TritonGPUHoistTMEMAlloc pass incorrectly fuses add and dot operations. Use `tl.inline_asm_elementwise` wrapper (see FLA upstream `safe_dot`) if encountering miscompilation. Track: https://github.com/triton-lang/triton/issues/8695

## Code Rules

- All changes go into `MY/` only. Do NOT modify `bench.py`, `sglang_chunked_gdn/`, `flashinfer/`, or `FLA/`.
- Do not install any additional packages. Only use what's in `requirements.txt` plus PyTorch and Triton.
- chunk_size=64 is the proven optimal. CS=128 was tested and rejected (see ITERATIONS.md Iter 22).

## Key Performance Facts

- **Bottleneck:** `chunk_gated_delta_rule_fwd_kernel_h` (state recurrence) is 45-60% of total time. It has a sequential time-loop over chunks that cannot be parallelized — this is an algorithmic limitation.
- **FlashInfer beats MY on short sequences** (T<8192) due to single kernel launch + TMA + zero HBM round-trip. See FLASHINFER_VS_MY_ANALYSIS.md.
- **MY beats FlashInfer on long sequences** (T>8192) due to CS=64 being lighter weight than CS=128.
- **MY beats SGLang by 20-35%** across all configs.
- **exp vs exp2:** No measurable performance difference on B200. USE_EXP2 currently set to False.
- **Autotune variability:** ~10-15% across Modal runs due to non-deterministic autotune. Best results not guaranteed on every run.

## Previously Attempted Optimizations (diminishing returns)

These were tried and either reverted or showed marginal gain:
- Expanding autotune search space for h, o, wy kernels — autotune already converged
- `maxnreg=128` on h kernel — regression from register spilling
- bf16 A matrix — precision-sensitive path affected
- Rewriting gdn_gating kernel — only 4% of total, not worth complexity
- Wider chunk_o autotune — autotune picked worse configs
- chunk_size=16, 32, 128 — all worse than 64 (see ITERATIONS.md)
- exp2 optimization — no measurable difference on B200
- Fusing kkt_solve + wy_fast — not worth it: A matrix only 8KB, register pressure too high
