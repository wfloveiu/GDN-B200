"""
DeltaNet Recurrent Step - CUDA V3 multi-warp pipelined kernel wrapper.

V3: 4 warps per block + cp.async prefetch + float4 vectorized access.
"""
import os
import math
import torch
from torch.utils.cpp_extension import load

_cur_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_module = load(
    name="deltanet_recurrent_cuda_v3_ext",
    sources=[os.path.join(_cur_dir, "deltanet_recurrent_cuda_v3.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100"],
    verbose=False,
)


def kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: float,
    output: torch.Tensor,
    new_state: torch.Tensor,
):
    B, T, H, K = q.shape

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    _cuda_module.forward(
        q, k, v, state,
        A_log, a, dt_bias, b,
        scale,
        output, new_state,
    )

    return output, new_state
