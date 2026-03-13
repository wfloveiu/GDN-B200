"""
DeltaNet Recurrent Step - CUDA kernel wrapper and benchmark.

Compares CUDA single-pass kernel vs Triton tiled kernel vs PyTorch reference.
"""
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# JIT compile the CUDA kernel
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_module = load(
    name="deltanet_recurrent_cuda",
    sources=[os.path.join(_cur_dir, "deltanet_recurrent_cuda.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
    verbose=False,
)


def deltanet_recurrent_step_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
) -> tuple:
    """Fused DeltaNet recurrent step using CUDA kernel.

    Args:
        q: [B, num_heads, Dk]
        k: [B, num_heads, Dk]
        v: [B, num_heads, Dv]
        beta: [B, num_heads]
        gate: [B, num_heads]
        state: [B, num_heads, Dk, Dv] (modified in-place)

    Returns:
        output: [B, num_heads, Dv]
        state: same tensor, updated in-place
    """
    output = _cuda_module.forward(q, k, v, beta, gate, state)
    return output, state


# =============================================================================
# Model wrapper (matches Triton file interface)
# =============================================================================

NUM_HEADS = 48
DK = 128
DV = 128


class CudaModel(nn.Module):
    def __init__(self, num_heads=NUM_HEADS, dk=DK, dv=DV):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.register_buffer("state", torch.zeros(1, num_heads, dk, dv, dtype=torch.bfloat16))

    def forward(self, q, k, v, beta, gate):
        output, self.state = deltanet_recurrent_step_cuda(
            q, k, v, beta, gate, self.state,
        )
        return output


# =============================================================================
# Benchmark
# =============================================================================

def get_inputs():
    B = 1
    return [
        torch.randn(B, NUM_HEADS, DK, device="cuda", dtype=torch.bfloat16),
        torch.randn(B, NUM_HEADS, DK, device="cuda", dtype=torch.bfloat16),
        torch.randn(B, NUM_HEADS, DV, device="cuda", dtype=torch.bfloat16),
        torch.rand(B, NUM_HEADS, device="cuda", dtype=torch.bfloat16),
        torch.randn(B, NUM_HEADS, device="cuda", dtype=torch.bfloat16) * 0.1,
    ]


def benchmark_fn(fn, inputs, warmup=100, repeat=1000):
    """Benchmark a function using CUDA events."""
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn(*inputs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat * 1000  # microseconds


if __name__ == "__main__":
    from recurrent import TritonModel

    torch.manual_seed(42)

    print("=" * 60)
    print("DeltaNet Recurrent Step - CUDA vs Triton")
    print("=" * 60)

    # ----- Correctness: CUDA vs Triton -----
    print("\n--- Correctness (CUDA vs Triton) ---")
    inputs = get_inputs()

    cuda_model = CudaModel(NUM_HEADS, DK, DV).cuda()
    triton_model = TritonModel(NUM_HEADS, DK, DV).cuda()

    out_cuda = cuda_model(*inputs).float()
    out_triton = triton_model(*inputs).float()

    abs_diff = (out_cuda - out_triton).abs()
    rel_diff = abs_diff / (out_triton.abs() + 1e-6)

    print(f"Max  absolute error: {abs_diff.max().item():.6f}")
    print(f"Mean absolute error: {abs_diff.mean().item():.6f}")
    print(f"Max  relative error: {rel_diff.max().item():.6f}")
    print(f"Mean relative error: {rel_diff.mean().item():.6f}")

    # ----- Performance: CUDA vs Triton -----
    print(f"\n--- Performance (B=1, H=48, Dk=Dv=128) ---")
    print(f"{'Kernel':<12} {'Latency (us)':>14}")
    print("-" * 28)

    triton_model2 = TritonModel(NUM_HEADS, DK, DV).cuda()
    us_triton = benchmark_fn(triton_model2, get_inputs())

    cuda_model2 = CudaModel(NUM_HEADS, DK, DV).cuda()
    us_cuda = benchmark_fn(cuda_model2, get_inputs())

    print(f"{'Triton':<12} {us_triton:>14.2f}")
    print(f"{'CUDA':<12} {us_cuda:>14.2f}")

    if us_cuda < us_triton:
        print(f"\nCUDA is {us_triton/us_cuda:.2f}x faster than Triton")
    else:
        print(f"\nTriton is {us_cuda/us_triton:.2f}x faster than CUDA")

    print()
