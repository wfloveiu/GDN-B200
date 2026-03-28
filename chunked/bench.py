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
import chunk as my_chunk_module
from chunk import chunk_gated_delta_rule as my_chunk_gated_delta_rule

# Import SGLang kernel
sglang_path = os.path.dirname(__file__)
if sglang_path not in sys.path:
    sys.path.insert(0, sglang_path)
from sglang_chunked_gdn.chunk import chunk_gated_delta_rule_fwd as sglang_fwd
from sglang_chunked_gdn.fused_gdn_gating import fused_gdn_gating as sglang_gating


# ===================== Shared input generation =====================

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


# ===================== Benchmark utils (match official harness) =====================

def _clone_args(args):
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


def do_bench_cold(fn, args, warmup=5, rep=10, device="cuda"):
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device=device)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for _ in range(warmup):
        cache.zero_()
        cloned = _clone_args(args)
        fn(*cloned)
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


# ===================== Wrappers =====================

def run_my(q, k, v, state, A_log, a_param, dt_bias, b_param, cu_seqlens, scale):
    return my_chunk_gated_delta_rule(q, k, v, state, A_log, a_param, dt_bias, b_param, cu_seqlens, scale)


def run_sglang(q, k, v, state, A_log, a_param, dt_bias, b_param, cu_seqlens, scale):
    g_log, beta = sglang_gating(A_log, a_param, b_param, dt_bias)
    q_4d = q.unsqueeze(0)
    k_4d = k.unsqueeze(0)
    v_4d = v.unsqueeze(0)
    num_seqs = cu_seqlens.size(0) - 1
    initial_state_indices = torch.arange(num_seqs, dtype=torch.long, device=q.device)
    _, o, _, _, _, _ = sglang_fwd(
        q=q_4d, k=k_4d, v=v_4d,
        g=g_log, beta=beta, scale=scale,
        initial_state=state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
    )
    return o.squeeze(0), state


# ===================== Benchmark =====================

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

    print("=" * 130)
    print(f"{'seqs':>5} {'seq_len':>8} {'total_T':>8} {'QK_h':>5} {'V_h':>5} | "
          f"{'MY exp (ms)':>12} {'MY exp2 (ms)':>13} {'SGLang (ms)':>12} | "
          f"{'exp2/exp':>9} {'MY/SG':>7}")
    print("-" * 130)

    for num_seqs, seq_len, nq, nk, nv in configs:
        args = make_inputs(num_seqs, seq_len, num_q_heads=nq, num_k_heads=nk, num_v_heads=nv)

        # MY with exp (USE_EXP2=False)
        my_chunk_module.USE_EXP2 = False
        ms_exp = do_bench_cold(run_my, args, warmup=warmup, rep=rep)

        # MY with exp2 (USE_EXP2=True)
        my_chunk_module.USE_EXP2 = True
        ms_exp2 = do_bench_cold(run_my, args, warmup=warmup, rep=rep)

        # SGLang
        ms_sg = do_bench_cold(run_sglang, args, warmup=warmup, rep=rep)

        exp2_vs_exp = ms_exp / ms_exp2 if ms_exp2 > 0 else 0
        best_my = min(ms_exp, ms_exp2)
        my_vs_sg = ms_sg / best_my if best_my > 0 else 0

        print(f"{num_seqs:>5} {seq_len:>8} {num_seqs*seq_len:>8} {nq:>5} {nv:>5} | "
              f"{ms_exp:>12.3f} {ms_exp2:>13.3f} {ms_sg:>12.3f} | "
              f"{exp2_vs_exp:>8.2f}x {my_vs_sg:>6.2f}x")

    print("=" * 130)


# ===================== Main =====================

if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    benchmark()
