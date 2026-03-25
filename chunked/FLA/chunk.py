import math

import torch
import torch.nn.functional as F

from utils import prepare_chunk_indices, input_guard, autocast_custom_fwd
from cumsum import chunk_local_cumsum
from chunk_fwd_intra import chunk_gated_delta_rule_fwd_intra
from chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from solve_tril import solve_tril
from wy_fast import recompute_w_u_fwd
from chunk_delta_h import chunk_gated_delta_rule_fwd_h
from chunk_o import chunk_fwd_o
from fused_gdn_gating import fused_gdn_gating

# Default chunk size — can be overridden in chunk_gated_delta_rule
CHUNK_SIZE = 64


def _intra_fused(k, v, g, beta, cu_seqlens, chunk_indices):
    """Fused kkt + solve_tril + recompute_w_u. Only supports chunk_size=64."""
    return chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g, beta=beta,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )


def _intra_separate(k, v, g, beta, cu_seqlens, chunk_indices, chunk_size=64):
    """Separate kkt → solve_tril → recompute_w_u. Supports chunk_size in {16, 32, 64}."""
    A = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
        chunk_indices=chunk_indices,
    )
    w, u = recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A, g=g,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    return w, u, A


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
):
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)

    # Select intra-chunk path based on chunk_size
    if chunk_size == 64:
        # Use fused kernel (fastest for chunk_size=64)
        w, u, _ = _intra_fused(k, v, g, beta, cu_seqlens, chunk_indices)
    else:
        # Fall back to separate kernels for other chunk sizes
        w, u, _ = _intra_separate(k, v, g, beta, cu_seqlens, chunk_indices, chunk_size=chunk_size)

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=transpose_state_layout,
    )
    return o, final_state



@torch.compiler.disable
def chunk_gated_delta_rule(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Chunked gated delta rule with the same interface as baseline_run.

    Inputs:
        q: [total_seq_len, num_q_heads, head_size] bfloat16
        k: [total_seq_len, num_k_heads, head_size] bfloat16
        v: [total_seq_len, num_v_heads, head_size] bfloat16
        state: [num_seqs, num_v_heads, head_size, head_size] float32  (H, V, K layout)
        A_log: [num_v_heads] float32
        a: [total_seq_len, num_v_heads] bfloat16
        dt_bias: [num_v_heads] float32
        b: [total_seq_len, num_v_heads] bfloat16
        cu_seqlens: [num_seqs+1] int64
        scale: scalar float32
    Returns:
        output: [total_seq_len, num_v_heads, head_size] bfloat16
        new_state: [num_seqs, num_v_heads, head_size, head_size] float32
    """
    _, _, head_size = q.shape

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)

    g_log, beta = fused_gdn_gating(A_log, a, b, dt_bias) # [1, T, HV]

    # ---------- reshape 3D -> 4D (B=1 for varlen mode) ----------
    q_4d = q.unsqueeze(0)                                   # [1, T, Hq, K]
    k_4d = k.unsqueeze(0)                                   # [1, T, Hk, K]
    v_4d = v.unsqueeze(0)                                   # [1, T, HV, V]

    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if cu_seqlens is not None else None
    o, final_state = chunk_gated_delta_rule_fwd(
        q=q_4d,
        k=k_4d,
        v=v_4d,
        g=g_log,
        beta=beta,
        scale=scale,
        initial_state=state,
        output_final_state=True,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state_layout=True,
    )

    output = o.squeeze(0)

    return output, final_state
