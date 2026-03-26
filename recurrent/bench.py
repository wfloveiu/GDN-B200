import torch
from Triton_recurrent import kernel as qwen_kernel
from CUDA_recurrent import kernel as cuda_kernel
from cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update
import math
import torch.nn.functional as F

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
NUM_KHEADS = 8
NUM_VHEADS = 16
DK = 128
DV = 128


def get_inputs(batch_size=1):
    B, T, H, K, V = batch_size, 1, NUM_KHEADS, DK, DV
    HV = NUM_VHEADS  # HV (value heads)
    return dict(
        q=torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
        k=torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
        v=torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16),
        state=torch.zeros(B, HV, V, K, device="cuda", dtype=torch.float32),  # k-last [B, HV, V, K]
        A_log=torch.randn(HV, device="cuda", dtype=torch.float32) * -1.0,
        a=torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16) * 0.1,
        dt_bias=torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
        b=torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16),
        scale=1.0,
    )

@torch.no_grad()
def ref(q, k, v, state, A_log, a, dt_bias, b, scale):
    """
    Gated Delta Net decode reference implementation (k-last layout).

    State layout: [B, H, V, K] (k-last, K dimension at the end)

    Gate computation:
    g = exp(-exp(A_log) * softplus(a + dt_bias))
    beta = sigmoid(b)

    Delta rule update:
    state_new = g * state_old + k^T @ (beta * v + (1-beta) * k @ state_old) - k^T @ (k @ state_old)
    output = scale * q @ state_new
    """
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    num_heads = num_v_heads
    device = q.device

    assert K == 128 and V == 128
    assert T == 1

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()  # [B, 1, HV]
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))  # [B, 1, HV]
    beta = torch.sigmoid(b.float())  # [B, 1, HV]

    q_f32 = q.squeeze(1).float()
    k_f32 = k.squeeze(1).float()
    v_f32 = v.squeeze(1).float()
    g_f32 = g.squeeze(1).float()
    beta_f32 = beta.squeeze(1).float()

    if state is not None:
        state_f32 = state.float()
    else:
        state_f32 = torch.zeros(B, num_heads, V, K, dtype=torch.float32, device=device)

    q_exp = q_f32.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k_f32.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    new_state = torch.zeros_like(state_f32)
    output = torch.zeros(B, num_heads, V, dtype=torch.float32, device=device)

    for b_idx in range(B):
        for h_idx in range(num_heads):
            q_h = q_exp[b_idx, h_idx]
            k_h = k_exp[b_idx, h_idx]
            v_h = v_f32[b_idx, h_idx]
            h_state = state_f32[b_idx, h_idx].clone().transpose(-1, -2)  # [V,K] -> [K,V]
            g_val = g_f32[b_idx, h_idx]
            beta_val = beta_f32[b_idx, h_idx]

            old_state = g_val * h_state
            old_v = k_h @ old_state
            new_v = beta_val * v_h + (1 - beta_val) * old_v
            state_remove = k_h.unsqueeze(1) @ old_v.unsqueeze(0)
            state_update = k_h.unsqueeze(1) @ new_v.unsqueeze(0)
            h_state = old_state - state_remove + state_update

            output[b_idx, h_idx] = scale * (q_h @ h_state)
            new_state[b_idx, h_idx] = h_state.transpose(-1, -2)  # [K,V] -> [V,K]

    output = output.unsqueeze(1).to(torch.bfloat16)
    return output, new_state


def bench_cutedsl(batch_size):
    """Benchmark CuTe DSL kernel — compile once, then benchmark pure compiled kernel calls."""
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

    # CuTe DSL state: [pool_size, HV, K, V], pool_size=N, indices=[0..N-1]
    state = torch.zeros(N, HV, K, V, device="cuda", dtype=torch.float32)
    indices = torch.arange(N, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.arange(N + 1, dtype=torch.int32, device="cuda")

    use_small = N < SMALL_BATCH_THRESHOLD

    # Trigger JIT compilation via high-level API
    cutedsl_fused_sigmoid_gating_delta_rule_update(
        A_log, dt_bias, q, k, v, a, b,
        state, indices, scale=1.0, use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()

    # Get the compiled kernel object
    compiled = _get_compiled_kernel(N, H, HV, K, V, N, use_small, is_varlen_decode=False)

    # Wrap torch tensors as CuTe tensors (once, outside the loop)
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

    # Use the DEFAULT stream so CUDA Events can correctly measure timing
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    # Warmup
    for _ in range(100):
        compiled(cu_t, q_t, k_t, v_t, a_t, b_t, A_log_t, dt_bias_t, h0_t, idx_t, o_t, stream)
    torch.cuda.synchronize()

    # Benchmark
    repeat = 1000
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        compiled(cu_t, q_t, k_t, v_t, a_t, b_t, A_log_t, dt_bias_t, h0_t, idx_t, o_t, stream)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat * 1000  # us


def benchmark_fn(fn, kwargs, warmup=100, repeat=1000):
    for _ in range(warmup):
        fn(**kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn(**kwargs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat * 1000  # microseconds


def bench_kernel(kernel_fn, batch_size):
    """Run a single kernel benchmark, return latency in us."""
    torch.manual_seed(42)
    inputs = get_inputs(batch_size)
    B, T, H, K = inputs["q"].shape
    HV, V = inputs["v"].shape[2], inputs["v"].shape[3]

    out = torch.empty(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    st = torch.empty(B, HV, V, K, device="cuda", dtype=torch.float32)

    # warmup run
    kernel_fn(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"], out, st,
    )
    inputs["output"] = out
    inputs["new_state"] = st
    return benchmark_fn(kernel_fn, inputs)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Kernels to benchmark
    std_kernels = {
        "Qwen(Triton)": qwen_kernel,
        "CUDA":      cuda_kernel,
    }
    kernel_names = list(std_kernels.keys()) + ["CuTeDSL"]

    # Collect all results: {bs: {name: latency_us}}
    results = {}
    for bs in BATCH_SIZES:
        results[bs] = {}
        for name, fn in std_kernels.items():
            us = bench_kernel(fn, bs)
            results[bs][name] = us
            print(f"  B={bs:>4d}  {name:>14s}: {us:>8.2f} us", flush=True)
        # CuTe DSL (different interface, separate bench function)
        us = bench_cutedsl(bs)
        results[bs]["CuTeDSL"] = us
        print(f"  B={bs:>4d}  {'CuTeDSL':>14s}: {us:>8.2f} us", flush=True)

    # ===== Print summary table =====
    print()
    print("=" * 100)
    print(f"  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H={NUM_KHEADS}, HV={NUM_VHEADS}, K={DK}, V={DV})")
    print("=" * 100)

    # Header
    col_w = 14  # column width
    header = f"{'Batch':>6}"
    for name in kernel_names:
        header += f" | {name:>{col_w}}"
    header += f" | {'CUDA vs Qwen':>{col_w}}"
    print(header)
    print("-" * len(header))

    # Rows
    for bs in BATCH_SIZES:
        row = f"{bs:>6}"
        for name in kernel_names:
            us = results[bs][name]
            row += f" | {us:>{col_w}.2f}"
        # CUDA vs Qwen speedup
        qwen_us = results[bs]["Qwen(Triton)"]
        v3_us = results[bs]["CUDA"]
        speedup = qwen_us / v3_us
        if speedup >= 1.0:
            tag = f"{speedup:.1f}x faster"
        else:
            pct = (1.0 - speedup) * 100
            tag = f"{pct:.1f}% slower"
        row += f" | {tag:>{col_w}}"
        print(row)

    print("-" * len(header))
    print(f"{'(us)':>6}", end="")
    for _ in kernel_names:
        print(f" | {'lower=better':>{col_w}}", end="")
    print(f" | {'':>{col_w}}")
    print()
