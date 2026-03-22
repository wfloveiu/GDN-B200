import sys
import os
import torch
import torch.nn.functional as F
import triton

# Triton >= 3.6.0 requires an explicit memory allocator
if hasattr(triton, 'set_allocator'):
    def _torch_allocator(size: int, alignment: int, stream: int):
        return torch.empty(size, dtype=torch.uint8, device='cuda')
    triton.set_allocator(_torch_allocator)

# Add FLA directory to path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "FLA"))

from chunk import chunk_gated_delta_rule


def benchmark_chunk_gated_delta_rule():
    # ===================== Configuration =====================
    configs = [
        # (B, T, H, K, V)
        (4, 1024, 4, 128, 128),
        (4, 2048, 4, 128, 128),
        (4, 4096, 4, 128, 128),
        (2, 2048, 8, 128, 128),
        (2, 4096, 8, 128, 128),
        (1, 8192, 4, 128, 128),
    ]
    dtype = torch.bfloat16
    device = "cuda"
    warmup = 25
    rep = 100

    print("=" * 90)
    print(f"{'B':>3} {'T':>6} {'H':>3} {'K':>4} {'V':>4} | "
          f"{'Time (ms)':>10} {'TFLOPs':>8} {'GB/s':>8}")
    print("-" * 90)

    for B, T, H, K, V in configs:
        # ===================== Create Inputs =====================
        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        k = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device=device), p=2, dim=-1)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
        g = F.logsigmoid(torch.rand(B, T, H, dtype=dtype, device=device))
        h0 = torch.randn(B, H, K, V, dtype=dtype, device=device)

        # ===================== Correctness Smoke Test =====================
        o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
        )
        assert o.shape == (B, T, H, V), f"Output shape mismatch: {o.shape}"
        assert ht.shape == (B, H, K, V), f"Final state shape mismatch: {ht.shape}"

        # ===================== Benchmark =====================
        fn = lambda: chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        # Rough FLOP estimate:
        #   chunk_fwd_o: ~2*B*T*H*K*V (state-query) + 2*B*T*H*K*T_chunk (intra-chunk QK)
        #   chunk_fwd_h: ~2*B*NT*H*K*V (state update per chunk)
        #   chunk_scaled_dot_kkt: ~2*B*T*H*K*C (C=chunk_size=64)
        #   recompute_w_u: ~2*B*T*H*(K+V)*C
        C = 64
        NT = (T + C - 1) // C
        flops = (
            2 * B * T * H * K * V          # Q @ H -> O (state contribution)
            + 2 * B * T * H * K * C         # Q @ K^T intra-chunk
            + 2 * B * NT * H * K * V        # state recurrence (k^T @ v per chunk)
            + 2 * B * T * H * K * C         # scaled_dot_kkt
            + 2 * B * T * H * (K + V) * C   # recompute_w_u
        )
        tflops = flops / ms * 1e-9  # ms -> s -> TFLOP/s

        # Memory traffic (lower bound): read q,k,v,g,beta,h0; write o,ht
        elem_bytes = 2  # bf16
        mem_bytes = (
            B * T * H * (K + K + V) * elem_bytes     # q, k, v
            + B * T * H * elem_bytes                  # g
            + B * T * H * elem_bytes                  # beta
            + B * H * K * V * elem_bytes              # h0
            + B * T * H * V * elem_bytes              # o (write)
            + B * H * K * V * elem_bytes              # ht (write, fp32)
        )
        gb_s = mem_bytes / ms * 1e-6  # ms -> s -> GB/s

        print(f"{B:>3} {T:>6} {H:>3} {K:>4} {V:>4} | "
              f"{ms:>10.3f} {tflops:>8.2f} {gb_s:>8.1f}")

    print("=" * 90)


def benchmark_varying_seqlen():
    """Benchmark with varying sequence lengths to show scaling."""
    B, H, K, V = 2, 4, 128, 128
    dtype = torch.bfloat16
    device = "cuda"
    warmup = 25
    rep = 100

    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]

    print("\n" + "=" * 60)
    print(f"Varying T | B={B}, H={H}, K={K}, V={V}, dtype={dtype}")
    print("-" * 60)
    print(f"{'T':>8} | {'Time (ms)':>10} {'Tokens/s (M)':>14}")
    print("-" * 60)

    for T in seq_lens:
        q = torch.randn(B, T, H, K, dtype=dtype, device=device)
        k = F.normalize(torch.randn(B, T, H, K, dtype=dtype, device=device), p=2, dim=-1)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
        g = F.logsigmoid(torch.rand(B, T, H, dtype=dtype, device=device))
        h0 = torch.randn(B, H, K, V, dtype=dtype, device=device)

        fn = lambda: chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        tokens_per_sec = B * T / ms * 1e-3  # M tokens/s

        print(f"{T:>8} | {ms:>10.3f} {tokens_per_sec:>14.2f}")

    print("=" * 60)


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    benchmark_chunk_gated_delta_rule()
    benchmark_varying_seqlen()
