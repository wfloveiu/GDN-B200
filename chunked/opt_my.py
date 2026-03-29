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

# Import MY kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MY"))
from chunk import chunk_gated_delta_rule


# ===================== Baseline reference =====================

def matmul(a, b):
    return a.float() @ b.float()

@torch.no_grad()
def baseline_run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_k_heads = k.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    num_seqs = cu_seqlens.size(0) - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    x = a.float() + dt_bias.float()
    g = torch.exp(-torch.exp(A_log.float()) * F.softplus(x))
    beta = torch.sigmoid(b.float())

    q_exp = q.repeat_interleave(num_v_heads // num_q_heads, dim=1)
    k_exp = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    output = torch.zeros(total_seq_len, num_sab_heads, head_size, dtype=torch.bfloat16, device=device)
    new_state = torch.zeros(num_seqs, num_sab_heads, head_size, head_size, dtype=torch.float32, device=device)

    for seq_idx in range(num_seqs):
        seq_start = int(cu_seqlens[seq_idx].item())
        seq_end = int(cu_seqlens[seq_idx + 1].item())
        seq_len = seq_end - seq_start
        if seq_len <= 0:
            continue

        if state is not None:
            state_HKV = state[seq_idx].clone().float().transpose(-1, -2)
        else:
            state_HKV = torch.zeros(num_sab_heads, head_size, head_size, dtype=torch.float32, device=device)

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


# ===================== Input generation =====================

def make_inputs(num_seqs, seq_len, num_q_heads=4, num_k_heads=4, num_v_heads=8,
                head_size=128, dtype=torch.bfloat16, device="cuda"):
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

@torch.no_grad()
def test_correctness():
    test_configs = [
        (1, 6), (1, 17), (2, 23), (1, 64), (1, 65),
        (1, 127), (1, 128), (2, 64), (1, 8192), (4, 8192),
    ]
    REPEATS = 3

    print("=" * 78)
    print(f"Correctness: MY vs baseline ({REPEATS} repeats)")
    print("=" * 78)

    all_passed = True
    for num_seqs, seq_len in test_configs:
        total_T = num_seqs * seq_len
        worst_out = 0.0
        worst_st = 0.0
        has_fail = False

        for _ in range(REPEATS):
            args = make_inputs(num_seqs, seq_len)
            ref_out, ref_state = baseline_run(*args)
            tri_out, tri_state = chunk_gated_delta_rule(*args)

            if torch.isnan(tri_out).any() or torch.isnan(tri_state).any():
                has_fail = True
                break

            out_err = (ref_out.float() - tri_out.float()).abs().max().item()
            st_err = (ref_state.float() - tri_state.float()).abs().max().item()
            worst_out = max(worst_out, out_err)
            worst_st = max(worst_st, st_err)

            if out_err >= 1.0 or st_err >= 1.0:
                has_fail = True

        status = "FAIL" if has_fail else "PASS"
        if has_fail:
            all_passed = False
        print(f"  seqs={num_seqs}, T={seq_len:>5}, total={total_T:>5} | out={worst_out:.6f} st={worst_st:.6f} -> {status}")

    result = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"\n  Result: {result}")
    print("=" * 78)
    return all_passed


# ===================== Benchmark (cold cache) =====================

def _clone_args(args):
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


def do_bench_cold(fn, args, warmup=5, rep=10):
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for _ in range(warmup):
        cache.zero_()
        fn(*_clone_args(args))
    torch.cuda.synchronize()

    for i in range(rep):
        cache.zero_()
        cloned = _clone_args(args)
        torch.cuda.synchronize()
        start_events[i].record()
        fn(*cloned)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times)


def run_my(q, k, v, state, A_log, a_param, dt_bias, b_param, cu_seqlens, scale):
    return chunk_gated_delta_rule(q, k, v, state, A_log, a_param, dt_bias, b_param, cu_seqlens, scale)


def benchmark():
    configs = [
        (1, 6,     4, 4, 8),
        (32, 256,  4, 4, 8),
        (1, 8192,  4, 4, 8),
        (4, 8192,  4, 4, 8),
        (1, 65536, 4, 4, 8),
        (1, 8192,  8, 8, 16),
        (4, 8192,  8, 8, 16),
        (1, 65536, 8, 8, 16),
    ]
    warmup = 5
    rep = 10

    print("=" * 80)
    print(f"{'seqs':>5} {'seq_len':>8} {'total_T':>8} {'QK_h':>5} {'V_h':>5} | {'MY (ms)':>10}")
    print("-" * 80)

    for num_seqs, seq_len, nq, nk, nv in configs:
        args = make_inputs(num_seqs, seq_len, num_q_heads=nq, num_k_heads=nk, num_v_heads=nv)
        ms = do_bench_cold(run_my, args, warmup=warmup, rep=rep)
        print(f"{num_seqs:>5} {seq_len:>8} {num_seqs*seq_len:>8} {nq:>5} {nv:>5} | {ms:>10.3f}")

    print("=" * 80)


# ===================== Main =====================

if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    test_correctness()
    print()
    benchmark()
