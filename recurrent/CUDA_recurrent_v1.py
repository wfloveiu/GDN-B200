"""
DeltaNet Recurrent Step - CUDA V3 kernel wrapper.

Matches the same kernel() interface as Triton kernels for bench.py.
No host-side precomputation — raw parameters go directly to CUDA kernel.
"""
import os
import math
import torch
from torch.utils.cpp_extension import load

# JIT compile the CUDA kernel
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_module = load(
    name="deltanet_recurrent_cuda_v1_ext",
    sources=[os.path.join(_cur_dir, "deltanet_recurrent_cuda_v1.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90"],
    verbose=False,
)


def kernel(
    q: torch.Tensor,        # [B, T, H, K] bf16
    k: torch.Tensor,        # [B, T, H, K] bf16
    v: torch.Tensor,        # [B, T, HV, V] bf16
    state: torch.Tensor,    # [B, HV, V, K] f32 (k-last)
    A_log: torch.Tensor,    # [HV] f32
    a: torch.Tensor,        # [B, T, HV] bf16
    dt_bias: torch.Tensor,  # [HV] f32
    b: torch.Tensor,        # [B, T, HV] bf16
    scale: float,           # float32 scalar
    output: torch.Tensor,   # [B, T, HV, V] bf16 (DPS pre-allocated)
    new_state: torch.Tensor # [B, HV, V, K] f32 (DPS pre-allocated)
):
    B, T, H, K = q.shape

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    # NO host-side precomputation — raw parameters go directly to CUDA kernel
    _cuda_module.forward(
        q, k, v, state,
        A_log, a, dt_bias, b,
        scale,
        output, new_state,
    )

    return output, new_state
