"""
NCU profiling script for chunked GDN (Gated Delta Net) kernel.

This script profiles ALL Triton kernels launched by chunk_gated_delta_rule.
It uses the CUDA profiler API (cudaProfilerStart/Stop) so that NCU only
captures the final "clean" invocation — skipping autotune + warmup entirely.

Kernels profiled (in one call to chunk_gated_delta_rule):
  1. fused_gdn_gating_kernel                       — gate computation (g, beta)
  2. chunk_local_cumsum_scalar_kernel               — chunk-local cumulative sum
  3. chunk_gated_delta_rule_fwd_kkt_solve_kernel    — fused kkt + solve_tril
  4. recompute_w_u_fwd_kernel                       — WY representation (w, u)
  5. chunk_gated_delta_rule_fwd_kernel_h_blockdim64 — state recurrence (bottleneck)
  6. chunk_fwd_kernel_o                             — output computation

Usage:
    # Profile all kernels (fast — only captures one clean invocation):
    ncu --set detailed --target-processes all --replay-mode kernel \
        --profile-from-start off \
        python ncu_profile.py

    # Profile a specific kernel:
    ncu --set detailed --target-processes all --replay-mode kernel \
        --profile-from-start off \
        --kernel-name "chunk_gated_delta_rule_fwd_kernel_h.*" \
        python ncu_profile.py

    # Save to file for GUI analysis:
    ncu --set detailed --target-processes all --replay-mode kernel \
        --profile-from-start off \
        -o chunked_profile \
        python ncu_profile.py

Standalone (just runs the kernel, useful for testing):
    python ncu_profile.py [num_seqs] [seq_len]
"""
import sys
import os
import math

import torch
import torch.nn.functional as F
import triton

# Triton >= 3.6.0 requires an explicit memory allocator
if hasattr(triton, 'set_allocator'):
    def _torch_allocator(size: int, alignment: int, stream: int):
        return torch.empty(size, dtype=torch.uint8, device='cuda')
    triton.set_allocator(_torch_allocator)

# Add MY directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "MY"))

from chunk import chunk_gated_delta_rule

# ===================== Default config =====================
NUM_Q_HEADS = 4
NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_SIZE = 128


def make_inputs(num_seqs, seq_len, dtype=torch.bfloat16, device="cuda"):
    """Create inputs matching chunk_gated_delta_rule interface (same as bench.py)."""
    total_seq_len = num_seqs * seq_len

    q = torch.randn(total_seq_len, NUM_Q_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    k = F.normalize(
        torch.randn(total_seq_len, NUM_K_HEADS, HEAD_SIZE, dtype=dtype, device=device),
        p=2, dim=-1,
    )
    v = torch.randn(total_seq_len, NUM_V_HEADS, HEAD_SIZE, dtype=dtype, device=device)
    state = torch.randn(
        num_seqs, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE,
        dtype=torch.float32, device=device,
    ) * 0.01
    A_log = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device) * 0.5
    a = torch.randn(total_seq_len, NUM_V_HEADS, dtype=dtype, device=device)
    dt_bias = torch.randn(NUM_V_HEADS, dtype=torch.float32, device=device) * 0.1
    b = torch.randn(total_seq_len, NUM_V_HEADS, dtype=dtype, device=device)
    cu_seqlens = torch.arange(
        0, total_seq_len + 1, seq_len, dtype=torch.int64, device=device
    )
    scale = 1.0 / math.sqrt(HEAD_SIZE)

    return q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale


def main():
    # Parse optional args: num_seqs seq_len
    num_seqs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 8192

    torch.manual_seed(42)
    args = make_inputs(num_seqs, seq_len)
    total_T = num_seqs * seq_len

    print(f"[ncu_profile] Config: num_seqs={num_seqs}, seq_len={seq_len}, "
          f"total_T={total_T}")
    print(f"[ncu_profile] Heads: Q={NUM_Q_HEADS}, K={NUM_K_HEADS}, "
          f"V={NUM_V_HEADS}, head_size={HEAD_SIZE}")

    # ---- Warmup: triggers autotune + JIT compilation ----
    # NCU does NOT profile this region (--profile-from-start off)
    print("[ncu_profile] Warmup (autotune + JIT)...", flush=True)
    for _ in range(3):
        chunk_gated_delta_rule(*args)
    torch.cuda.synchronize()

    # ---- Profiled launch: only this call is captured by NCU ----
    print("[ncu_profile] Starting profiler...", flush=True)
    torch.cuda.cudart().cudaProfilerStart()

    output, final_state = chunk_gated_delta_rule(*args)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStop()
    print("[ncu_profile] Profiler stopped.", flush=True)

    print(f"[ncu_profile] Done. output shape={output.shape}, "
          f"state shape={final_state.shape}")


if __name__ == "__main__":
    main()
