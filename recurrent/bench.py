import sys
import os
import torch
import math
import torch.nn.functional as F

# Add subdirectories to path
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, "Triton"))
sys.path.insert(0, os.path.join(_dir, "Cuda"))
sys.path.insert(0, os.path.join(_dir, "CuteDSL"))

from Triton_recurrent import kernel as triton_kernel
from CUDA_recurrent import kernel as cuda_kernel
from cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
NUM_KHEADS = 4
NUM_VHEADS = 8
DK = 128
DV = 128
B200_PEAK_BW_GBS = 8000  # B200 HBM peak bandwidth in GB/s


def get_inputs(batch_size=1):
    B, T, H, K, V = batch_size, 1, NUM_KHEADS, DK, DV
    HV = NUM_VHEADS
    return dict(
        q=torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
        k=torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
        v=torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16),
        state=torch.zeros(B, HV, V, K, device="cuda", dtype=torch.float32),
        A_log=torch.randn(HV, device="cuda", dtype=torch.float32) * -1.0,
        a=torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16) * 0.1,
        dt_bias=torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
        b=torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16),
        scale=1.0,
    )


def compute_data_bytes(batch_size):
    """Compute total HBM data movement for one decode step."""
    B = batch_size
    HV = NUM_VHEADS
    K, V = DK, DV
    H = NUM_KHEADS

    # State: read + write = 2 * B * HV * V * K * 4 bytes (float32)
    state_bytes = 2 * B * HV * V * K * 4
    # Q: B * H * K * 2 (bf16), K: same, V: B * HV * V * 2 (bf16)
    qkv_bytes = B * (H * K * 2 + H * K * 2 + HV * V * 2)
    # Output: B * HV * V * 2 (bf16)
    output_bytes = B * HV * V * 2
    # Gate params: a, b (bf16) + A_log, dt_bias (f32, per-head, negligible)
    gate_bytes = B * HV * 2 * 2 + HV * 4 * 2

    return state_bytes + qkv_bytes + output_bytes + gate_bytes


# ===================== Benchmark utils (cold cache) =====================

def _clone_kwargs(kwargs):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}


def benchmark_fn(fn, kwargs, warmup=100, repeat=1000):
    """Cold cache benchmark: flush L2 + clone inputs each iteration."""
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

    for _ in range(warmup):
        cache.zero_()
        cloned = _clone_kwargs(kwargs)
        fn(**cloned)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        cache.zero_()
        cloned = _clone_kwargs(kwargs)
        torch.cuda.synchronize()
        start_events[i].record()
        fn(**cloned)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times) * 1000  # microseconds


def bench_kernel(kernel_fn, batch_size):
    torch.manual_seed(42)
    inputs = get_inputs(batch_size)
    B, T, H, K = inputs["q"].shape
    HV, V = inputs["v"].shape[2], inputs["v"].shape[3]

    out = torch.empty(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    st = torch.empty(B, HV, V, K, device="cuda", dtype=torch.float32)

    kernel_fn(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"], out, st,
    )
    inputs["output"] = out
    inputs["new_state"] = st
    return benchmark_fn(kernel_fn, inputs)


def bench_cutedsl(batch_size):
    import cuda.bindings.driver as cuda_drv
    from cutlass.cute.runtime import from_dlpack
    from cutedsl_gdn import _get_compiled_kernel, SMALL_BATCH_THRESHOLD

    torch.manual_seed(42)
    B, T, H, K, V = batch_size, 1, NUM_KHEADS, DK, DV
    HV = NUM_VHEADS
    N = B

    q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HV, device="cuda", dtype=torch.float32) * -1.0
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1
    o = torch.empty(B, T, HV, V, device="cuda", dtype=torch.bfloat16)

    state = torch.zeros(N, HV, K, V, device="cuda", dtype=torch.float32)
    indices = torch.arange(N, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(N + 1, dtype=torch.int32, device="cuda")

    use_small = N < SMALL_BATCH_THRESHOLD

    cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b,
        state, indices, scale=1.0, use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()

    compiled = _get_compiled_kernel(N, H, HV, K, V, N, use_small, is_varlen_decode=False)

    cu_t = from_dlpack(cu_seqlens, assumed_align=16)
    q_t = from_dlpack(q, assumed_align=16)
    k_t = from_dlpack(k, assumed_align=16)
    v_t = from_dlpack(v, assumed_align=16)
    a_t = from_dlpack(a, assumed_align=16)
    b_t = from_dlpack(b, assumed_align=16)
    A_log_t = from_dlpack(A_log, assumed_align=16)
    dt_bias_t = from_dlpack(dt_bias, assumed_align=16)
    h0_t = from_dlpack(state, assumed_align=16)
    idx_t = from_dlpack(indices, assumed_align=16)
    o_t = from_dlpack(o, assumed_align=16)

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

    for _ in range(100):
        cache.zero_()
        compiled(cu_t, q_t, k_t, v_t, a_t, b_t, A_log_t, dt_bias_t, h0_t, idx_t, o_t, stream)
    torch.cuda.synchronize()

    repeat = 1000
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        cache.zero_()
        torch.cuda.synchronize()
        start_events[i].record()
        compiled(cu_t, q_t, k_t, v_t, a_t, b_t, A_log_t, dt_bias_t, h0_t, idx_t, o_t, stream)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times) * 1000  # us


# ===================== Main =====================

if __name__ == "__main__":
    torch.manual_seed(42)

    # Kernels to benchmark
    std_kernels = {
        "Triton": triton_kernel,
        "CUDA":   cuda_kernel,
    }
    kernel_names = list(std_kernels.keys()) + ["CuTeDSL"]

    # Collect results
    results = {}
    for bs in BATCH_SIZES:
        results[bs] = {}
        for name, fn in std_kernels.items():
            us = bench_kernel(fn, bs)
            results[bs][name] = us
            print(f"  B={bs:>4d}  {name:>14s}: {us:>8.2f} us", flush=True)
        us = bench_cutedsl(bs)
        results[bs]["CuTeDSL"] = us
        print(f"  B={bs:>4d}  {'CuTeDSL':>14s}: {us:>8.2f} us", flush=True)

    # ===== Print summary table =====
    print()
    print("=" * 120)
    print(f"  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H={NUM_KHEADS}, HV={NUM_VHEADS}, K={DK}, V={DV})")
    print("=" * 120)

    col_w = 12
    header = (f"{'Batch':>6} | {'Data(MB)':>9} | "
              f"{'Triton':>{col_w}} | {'CUDA':>{col_w}} | {'CuTeDSL':>{col_w}} | "
              f"{'CUDA BW':>10} | {'Util%':>6} | {'CUDA/Triton':>12}")
    print(header)
    print("-" * len(header))

    for bs in BATCH_SIZES:
        data_bytes = compute_data_bytes(bs)
        data_mb = data_bytes / 1e6

        triton_us = results[bs]["Triton"]
        cuda_us = results[bs]["CUDA"]
        cutedsl_us = results[bs]["CuTeDSL"]

        # CUDA bandwidth
        bw_gbs = (data_bytes / 1e9) / (cuda_us * 1e-6)
        util_pct = bw_gbs / B200_PEAK_BW_GBS * 100

        # Speedup
        speedup = triton_us / cuda_us
        if speedup >= 1.0:
            tag = f"{speedup:.1f}x faster"
        else:
            tag = f"{(1-speedup)*100:.1f}% slower"

        print(f"{bs:>6} | {data_mb:>8.2f} | "
              f"{triton_us:>{col_w}.2f} | {cuda_us:>{col_w}.2f} | {cutedsl_us:>{col_w}.2f} | "
              f"{bw_gbs:>8.0f} GB/s | {util_pct:>5.1f}% | {tag:>12}")

    print("-" * len(header))
    print(f"{'(us)':>6} | {'':>9} | {'lower=better':>{col_w}} | {'lower=better':>{col_w}} | {'lower=better':>{col_w}} | "
          f"{'':>10} | {'':>6} | {'':>12}")
    print()
