# Hints

- Before Iter 1, run `ncu` on the baseline kernel to guide the first direction.
- If 3 consecutive iterations show no improvement, run `ncu` to re-profile, use WebSearch for new ideas, and review `ITERATIONS.md` for patterns. Plan before continuing.

## Hardware

- **Test machine:** B200 (Blackwell, SM 10.0, 192 GB HBM3e, 192 SMs, 228 KB shared mem/SM) — accessed remotely via Modal.
- Profile and iterate directly on B200 through `modal run run_on_modal.py`.

## B200 Optimization Guidelines

- **Prefer TMA (Tensor Memory Accelerator):** B200 supports TMA. Use `tl.make_tensor_descriptor` / TMA loads where possible — they leverage B200's larger L2 and higher bandwidth.
- **Target larger tile sizes:** B200 has 192 SMs. Choose tile configs that scale well to 192 SMs.
- **Maximize warp-level parallelism:** B200 supports up to 32 warps/SM. Prefer `num_warps=4` or `num_warps=8` configs.
- **Use FP8 where safe:** B200 has 2nd-gen FP8 Tensor Cores. If accumulation precision allows, consider FP8 compute paths — they are ~2x faster. However, correctness comes first.
- **Shared memory budget:** B200 has ~228 KB shared mem per SM. Do not push beyond this limit.
- **Autotune configs should cover B200 sweet spots:** Include `BV=64`, `BV=128` and `num_warps` in {2, 4, 8} in autotune search spaces.
- **HBM bandwidth:** B200 has ~8 TB/s. Memory-bound kernels benefit automatically, so focus optimization effort on **compute-bound** or **latency-bound** kernels.

## General

- Do not install any additional packages. Only use what's in `requirements.txt` plus PyTorch and Triton.
- All changes go into `FLA/` only. Do NOT modify `bench.py`.


