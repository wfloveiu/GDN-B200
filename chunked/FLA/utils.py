# Unified utils shim - re-exports everything FLA kernels need from local files

# From fla_utils.py (was fla/utils.py)
from fla_utils import (
    autotune_cache_kwargs,
    check_shared_mem,
    input_guard,
    tensor_cache,
    IS_NVIDIA,
    IS_NVIDIA_HOPPER,
    IS_NVIDIA_BLACKWELL,
    IS_TF32_SUPPORTED,
    IS_TMA_SUPPORTED,
    IS_GATHER_SUPPORTED,
    USE_CUDA_GRAPH,
    contiguous,
)

# From op.py (was fla/ops/utils/op.py)
from op import exp, exp2

# From index.py (was fla/ops/utils/index.py)
from index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

# autocast helper (simple passthrough decorator)
import torch
import functools

def autocast_custom_fwd(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper
