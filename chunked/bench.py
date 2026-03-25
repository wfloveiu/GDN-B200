import sys
import os
import torch
import torch.nn.functional as F
import triton
import math

# Triton >= 3.6.0 requires an explicit memory allocator
if hasattr(triton, 'set_allocator'):
    def _torch_allocator(size: int, alignment: int, stream: int):
        return torch.empty(size, dtype=torch.uint8, device='cuda')
    triton.set_allocator(_torch_allocator)

# Add FLA directory to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "FLA"))

from chunk import chunk_gated_delta_rule


# ===================== Baseline reference =====================

def matmul(a: torch.Tensor, b: torch.Tensor):
    """Float32 matmul for numerical stability."""
    return a.float() @ b.float()


@torch.no_grad()
def baseline_run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Inputs:
        q: [total_seq_len, num_q_heads, head_size] bfloat16
        k: [total_seq_len, num_k_heads, head_size] bfloat16
        v: [total_seq_len, num_v_heads, head_size] bfloat16
        state: [num_seqs, num_v_heads, head_size, head_size] float32
        A_log: [num_v_heads]  float32
        a: [total_seq_len, num_v_heads] bfloat16
        dt_bias: [num_v_heads]  float32
        b: [total_seq_len, num_v_heads] bfloat16
        cu_seqlens: [num_seqs+1] int64
        scale: scalar float32
    Outputs:
        output: [total_seq_len, num_v_heads, head_size] bfloat16
        new_state: [num_seqs, num_v_heads, head_size, head_size] float32
    """
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    assert num_q_heads == 4
    assert num_k_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    # Compute g and beta from raw parameters
    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(
        (total_seq_len, num_sab_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start

        if seq_len <= 0:
            continue

        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)
        else:
            state_HKV = torch.zeros(
                (num_sab_heads, head_size, head_size), dtype=torch.float32, device=device
            )

        for i in range(seq_len):
            t = seq_start + i
            q_H1K = q_exp[t].unsqueeze(1).float()
            k_H1K = k_exp[t].unsqueeze(1).float()
            v_H1V = v[t].unsqueeze(1).float()
            g_H11 = g[t].unsqueeze(1).unsqueeze(2)
            beta_H11 = beta[t].unsqueeze(1).unsqueeze(2)

            old_state_HKV = g_H11 * state_HKV
            old_v_H1V = matmul(k_H1K, old_state_HKV)
            new_v_H1V = beta_H11 * v_H1V + (1 - beta_H11) * old_v_H1V
            state_remove = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), old_v_H1V)
            state_update = torch.einsum('hkl,hlv->hkv', k_H1K.transpose(-1, -2), new_v_H1V)
            state_HKV = old_state_HKV - state_remove + state_update

            o_H1V = scale * matmul(q_H1K, state_HKV)
            output[t] = o_H1V.squeeze(1).to(torch.bfloat16)

        new_state[seq_idx] = state_HKV.transpose(-1, -2)

    return output, new_state


# ===================== Shared input generation =====================

def make_inputs(num_seqs, seq_len, num_q_heads=4, num_k_heads=4, num_v_heads=8,
                head_size=128, dtype=torch.bfloat16, device="cuda"):
    """Create inputs matching baseline_run / chunk_gated_delta_rule interface."""
    total_seq_len = num_seqs * seq_len

    q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=dtype, device=device)
    k = F.normalize(torch.randn(total_seq_len, num_k_heads, head_size, dtype=dtype, device=device), p=2, dim=-1)
    v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=dtype, device=device)
    state = torch.randn(num_seqs, num_v_heads, head_size, head_size, dtype=torch.float32, device=device) * 0.01
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.5
    a_param = torch.randn(total_seq_len, num_v_heads, dtype=dtype, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1
    b_param = torch.randn(total_seq_len, num_v_heads, dtype=dtype, device=device)
    cu_seqlens = torch.arange(0, total_seq_len + 1, seq_len, dtype=torch.int64, device=device)
    scale = 1.0 / math.sqrt(head_size)

    return q, k, v, state, A_log, a_param, dt_bias, b_param, cu_seqlens, scale


# ===================== Correctness test =====================

def _report_errors(name, ref, tri):
    """Report max abs/rel error with values at the error positions."""
    ref_f, tri_f = ref.float(), tri.float()
    diff = (ref_f - tri_f).abs()

    # Max absolute error
    abs_err_flat = diff.reshape(-1)
    abs_idx = abs_err_flat.argmax().item()
    max_abs_err = abs_err_flat[abs_idx].item()
    ref_val_abs = ref_f.reshape(-1)[abs_idx].item()
    tri_val_abs = tri_f.reshape(-1)[abs_idx].item()

    # Max relative error
    denom = torch.max(ref_f.abs(), tri_f.abs()).clamp(min=1e-8)
    rel_err_flat = (diff / denom).reshape(-1)
    rel_idx = rel_err_flat.argmax().item()
    max_rel_err = rel_err_flat[rel_idx].item()
    ref_val_rel = ref_f.reshape(-1)[rel_idx].item()
    tri_val_rel = tri_f.reshape(-1)[rel_idx].item()

    print(f"  {name:7s} | max_abs_err: {max_abs_err:.6f}  (ref={ref_val_abs:.6f}, tri={tri_val_abs:.6f})")
    print(f"  {' ':7s} | max_rel_err: {max_rel_err:.6f}  (ref={ref_val_rel:.6f}, tri={tri_val_rel:.6f})")
    return max_abs_err, max_rel_err


@torch.no_grad()
def test_correctness():
    """Compare chunk_gated_delta_rule against baseline_run on short sequences."""
    print("=" * 70)
    print("Correctness Test: chunk_gated_delta_rule vs baseline_run")
    print("-" * 70)

    num_seqs, seq_len = 2, 64
    args = make_inputs(num_seqs, seq_len)

    ref_out, ref_state = baseline_run(*args)
    tri_out, tri_state = chunk_gated_delta_rule(*args)

    out_abs, _ = _report_errors("Output", ref_out, tri_out)
    st_abs, _ = _report_errors("State", ref_state, tri_state)

    passed = out_abs < 1.0 and st_abs < 1.0
    status = "PASS" if passed else "FAIL"
    print(f"  Status  | {status}")
    print("=" * 70)
    return passed


# ===================== Benchmark =====================

def benchmark():
    """Benchmark chunk_gated_delta_rule with various configs."""
    # (num_seqs, seq_len, num_q_heads, num_k_heads, num_v_heads)
    configs = [
        (1, 8192,  4, 4, 8),
        (4, 8192,  4, 4, 8),
        (1, 65536, 4, 4, 8),
        (1, 8192,  8, 8, 16),
        (4, 8192,  8, 8, 16),
        (1, 65536, 8, 8, 16),
    ]
    warmup = 5
    rep = 10

    print("\n" + "=" * 100)
    print(f"{'seqs':>5} {'seq_len':>8} {'total_T':>8} {'QK_h':>5} {'V_h':>5} | {'Time (ms)':>10} {'Tokens/s (M)':>14}")
    print("-" * 100)

    for num_seqs, seq_len, nq, nk, nv in configs:
        args = make_inputs(num_seqs, seq_len, num_q_heads=nq, num_k_heads=nk, num_v_heads=nv)

        fn = lambda: chunk_gated_delta_rule(*args)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        total_T = num_seqs * seq_len
        tokens_per_sec = total_T / ms * 1e-3  # M tokens/s

        print(f"{num_seqs:>5} {seq_len:>8} {total_T:>8} {nq:>5} {nv:>5} | {ms:>10.3f} {tokens_per_sec:>14.2f}")

    print("=" * 100)


# ===================== Main =====================

if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    test_correctness()
    benchmark()
