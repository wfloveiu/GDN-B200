import torch
from FLA_recurrent import kernel as fla_kernel
from Qwen_recurrent import kernel as qwen_kernel
from CUDA_recurrent import kernel as cuda_kernel
import math
import torch.nn.functional as F

BatchSize = 1
NUM_KHEADS = 48
NUM_VHEADS = 48
DK = 128
DV = 128


def get_inputs():
    B, T, H, K, V = BatchSize, 1, NUM_KHEADS, DK, DV
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
    
    # assert num_q_heads == 4
    # assert num_k_heads == 4
    # assert num_v_heads == 8
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


def _print_diff(label, test_out, ref_out, test_st, ref_st):
    """Print absolute/relative error table for output and state."""
    abs_diff_out = (test_out - ref_out).abs()
    rel_diff_out = abs_diff_out / (ref_out.abs() + 1e-6)
    abs_diff_st = (test_st - ref_st).abs()
    rel_diff_st = abs_diff_st / (ref_st.abs() + 1e-6)

    print(f"--- Correctness: {label} ---")
    print(f"{'':>28} {'Output':>14} {'State':>14}")
    print("-" * 58)
    print(f"{'Max  absolute error':>28} {abs_diff_out.max().item():>14.6f} {abs_diff_st.max().item():>14.6f}")
    print(f"{'Mean absolute error':>28} {abs_diff_out.mean().item():>14.6f} {abs_diff_st.mean().item():>14.6f}")
    print(f"{'Max  relative error':>28} {rel_diff_out.max().item():>14.6f} {rel_diff_st.max().item():>14.6f}")
    print(f"{'Mean relative error':>28} {rel_diff_out.mean().item():>14.6f} {rel_diff_st.mean().item():>14.6f}")


def fla_accuracy():
    """Compare fla_kernel vs ref() (PyTorch reference)."""
    torch.manual_seed(42)
    inputs = get_inputs()
    B, T, H, K = inputs["q"].shape
    HV, V = inputs["v"].shape[2], inputs["v"].shape[3]

    # --- Run ref (PyTorch reference) ---
    ref_output, ref_state = ref(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"],
    )

    # --- Run fla_kernel ---
    fla_output = torch.empty(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    fla_state = torch.empty(B, HV, V, K, device="cuda", dtype=torch.float32)
    fla_kernel(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"],
        fla_output, fla_state,
    )

    _print_diff("FLA vs Ref",
                fla_output.float(), ref_output.float(),
                fla_state.float(), ref_state.float())

    inputs["output"] = fla_output
    inputs["new_state"] = fla_state
    us = benchmark_fn(fla_kernel, inputs)
    print(f"FLA kernel latency: {us:.2f} us")


def qwen_accuracy():
    """Compare qwen_kernel vs ref() (PyTorch reference)."""
    torch.manual_seed(42)
    inputs = get_inputs()
    B, T, H, K = inputs["q"].shape
    HV, V = inputs["v"].shape[2], inputs["v"].shape[3]

    # --- Run ref (PyTorch reference) ---
    ref_output, ref_state = ref(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"],
    )

    # --- Run qwen_kernel ---
    qwen_output = torch.empty(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    qwen_state = torch.empty(B, HV, V, K, device="cuda", dtype=torch.float32)
    qwen_kernel(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"],
        qwen_output, qwen_state,
    )

    _print_diff("Qwen vs Ref",
                qwen_output.float(), ref_output.float(),
                qwen_state.float(), ref_state.float())

    inputs["output"] = qwen_output
    inputs["new_state"] = qwen_state
    us = benchmark_fn(qwen_kernel, inputs)
    print(f"Qwen kernel latency: {us:.2f} us")





def cuda_accuracy():
    """Compare cuda_kernel vs ref() (PyTorch reference)."""
    torch.manual_seed(42)
    inputs = get_inputs()
    B, T, H, K = inputs["q"].shape
    HV, V = inputs["v"].shape[2], inputs["v"].shape[3]

    # --- Run ref (PyTorch reference) ---
    ref_output, ref_state = ref(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"],
    )

    # --- Run cuda_kernel ---
    cuda_output = torch.empty(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    cuda_state = torch.empty(B, HV, V, K, device="cuda", dtype=torch.float32)
    cuda_kernel(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"],
        cuda_output, cuda_state,
    )

    _print_diff("CUDA vs Ref",
                cuda_output.float(), ref_output.float(),
                cuda_state.float(), ref_state.float())

    inputs["output"] = cuda_output
    inputs["new_state"] = cuda_state
    us = benchmark_fn(cuda_kernel, inputs)
    print(f"CUDA kernel latency: {us:.2f} us")


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


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("FLA vs Qwen DeltaNet Recurrent Kernel Comparison")
    print(f"B={BatchSize}, T=1 (decode), H={NUM_KHEADS}, HV={NUM_VHEADS}, K={DK}, V={DV}")
    print("=" * 60)

    # Correctness comparison
    print()
    # fla_accuracy()
    qwen_accuracy()
    cuda_accuracy()
    print()

