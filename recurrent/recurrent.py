"""
Fused DeltaNet Recurrent State Update  - THE FLAGSHIP KERNEL (v2: Tiled).

v2 optimization: Loads state in 2D tiles [BLOCK_DK, BLOCK_DV] instead of
row-by-row. This changes 128 sequential loads per pass into 8-16 coalesced
tile loads that the hardware can pipeline.

v1 problem: 128 sequential row loads × ~400ns HBM latency = 51μs per pass
v2 fix: 8 tile loads (BLOCK_DK=16) × ~400ns = 3.2μs per pass

Expected speedup: 3-5x on the recurrent kernel (from ~40μs to ~8-12μs/layer)

DeltaNet linear attention recurrence (per head, per decode step):
    S *= exp(g)                     # gate decay
    residual = v - S^T @ k          # delta rule residual
    delta = beta * residual
    S += outer(k, delta)            # rank-1 state update
    o = S^T @ q                     # output query

After Q/K repeat_interleave (16→48 heads), operates on 48 independent heads.
Per head: state S is [128, 128] = 32KB BF16. Fits in B200 SRAM.

Grid: (num_v_heads, B) = (48, 1) for decode.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # BLOCK_DK=16: 8 tile iterations per pass, good coalescing
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 16}, num_warps=8, num_stages=1),
        # BLOCK_DK=32: 4 tile iterations, more register pressure but fewer iterations
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 32}, num_warps=8, num_stages=1),
        # BLOCK_DK=8: 16 iterations, less register pressure, more occupancy
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_DV": 128, "BLOCK_DK": 8}, num_warps=8, num_stages=1),
    ],
    key=["Dk", "Dv"],
    restore_value=["S_ptr"],
)
@triton.jit
def _deltanet_recurrent_kernel(
    # Input vectors (per token, per head)
    Q_ptr, K_ptr, V_ptr, Beta_ptr, G_ptr,
    # State [B, num_heads, Dk, Dv]
    S_ptr,
    # Output [B, num_heads, Dv]
    O_ptr,
    # Dimensions
    Dk: tl.constexpr, Dv: tl.constexpr,
    # Strides for state
    stride_sb, stride_sh, stride_sdk, stride_sdv,
    # Strides for q/k/v/o (per head)
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    # Strides for beta/gate (scalar per head)
    stride_betab, stride_betah,
    stride_gb, stride_gh,
    BLOCK_DV: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    """Fused DeltaNet recurrent step for decode  - TILED version.

    Loads state as 2D tiles [BLOCK_DK, BLOCK_DV] instead of row-by-row.
    This coalesces memory access: 128 sequential loads → 8 coalesced tiles.

    Each program handles one (batch, head) pair.
    """
    head_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    # Load scalar inputs for this head
    beta = tl.load(Beta_ptr + batch_id * stride_betab + head_id * stride_betah).to(tl.float32)
    g = tl.load(G_ptr + batch_id * stride_gb + head_id * stride_gh).to(tl.float32)
    decay = tl.exp(g)

    # Dv offsets for tile columns
    dv_offs = tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < Dv

    # Pre-load full q and k vectors [Dk]  - these are small (128 elements)
    # Load them as 1D vectors, index by tile row offset later
    q_base = Q_ptr + batch_id * stride_qb + head_id * stride_qh
    k_base = K_ptr + batch_id * stride_kb + head_id * stride_kh

    # Load v vector [Dv]
    v = tl.load(V_ptr + batch_id * stride_vb + head_id * stride_vh + dv_offs * stride_vd,
                mask=dv_mask, other=0.0).to(tl.float32)

    s_base = S_ptr + batch_id * stride_sb + head_id * stride_sh

    # ===== Pass 1: Decay state + accumulate S^T @ k (TILED) =====
    # Instead of 128 sequential row loads, process BLOCK_DK rows at once
    accumulated = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    for r_start in range(0, Dk, BLOCK_DK):
        # Row indices for this tile
        dk_offs = r_start + tl.arange(0, BLOCK_DK)
        dk_mask = dk_offs < Dk

        # Load state tile [BLOCK_DK, BLOCK_DV]  - COALESCED 2D load
        s_ptrs = s_base + dk_offs[:, None] * stride_sdk + dv_offs[None, :] * stride_sdv
        s_mask = dk_mask[:, None] & dv_mask[None, :]
        s_tile = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        # Apply gate decay to entire tile
        s_tile = s_tile * decay

        # Load k values for these rows [BLOCK_DK]
        k_tile = tl.load(k_base + dk_offs * stride_kd, mask=dk_mask, other=0.0).to(tl.float32)

        # Accumulate S^T @ k: accumulated += sum_r(S[r, :] * k[r])
        # s_tile is [BLOCK_DK, BLOCK_DV], k_tile[:, None] is [BLOCK_DK, 1]
        accumulated += tl.sum(s_tile * k_tile[:, None], axis=0)

        # Store decayed state tile back
        tl.store(s_ptrs, s_tile.to(tl.bfloat16), mask=s_mask)

    # Compute delta = beta * (v - S^T @ k)
    delta = beta * (v - accumulated)

    # ===== Pass 2: State update + compute output S^T @ q (TILED) =====
    output = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    for r_start in range(0, Dk, BLOCK_DK):
        dk_offs = r_start + tl.arange(0, BLOCK_DK)
        dk_mask = dk_offs < Dk

        # Reload decayed state tile [BLOCK_DK, BLOCK_DV]
        # This data should still be in L2 cache from pass 1 (only 32KB/head)
        s_ptrs = s_base + dk_offs[:, None] * stride_sdk + dv_offs[None, :] * stride_sdv
        s_mask = dk_mask[:, None] & dv_mask[None, :]
        s_tile = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

        # Rank-1 update: S[r, :] += k[r] * delta[:]
        k_tile = tl.load(k_base + dk_offs * stride_kd, mask=dk_mask, other=0.0).to(tl.float32)
        s_tile += k_tile[:, None] * delta[None, :]

        # Store updated state tile
        tl.store(s_ptrs, s_tile.to(tl.bfloat16), mask=s_mask)

        # Accumulate output: o += sum_r(S_new[r, :] * q[r])
        q_tile = tl.load(q_base + dk_offs * stride_qd, mask=dk_mask, other=0.0).to(tl.float32)
        output += tl.sum(s_tile * q_tile[:, None], axis=0)

    # Store output [Dv]
    o_ptrs = O_ptr + batch_id * stride_ob + head_id * stride_oh + dv_offs * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=dv_mask)


def deltanet_recurrent_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
) -> tuple:
    """Fused DeltaNet recurrent step for decode.

    Args:
        q: [B, num_heads, Dk] query vector
        k: [B, num_heads, Dk] key vector
        v: [B, num_heads, Dv] value vector
        beta: [B, num_heads] scaling factor
        gate: [B, num_heads] log gate (applied as exp(gate))
        state: [B, num_heads, Dk, Dv] recurrent state (MODIFIED IN-PLACE)

    Returns:
        output: [B, num_heads, Dv] attention output
        state: same tensor, updated in-place
    """
    B, num_heads, Dk = q.shape
    Dv = v.shape[-1]

    output = torch.empty(B, num_heads, Dv, device=q.device, dtype=torch.bfloat16)

    grid = (num_heads, B)

    _deltanet_recurrent_kernel[grid](
        q, k, v, beta, gate,
        state, output,
        Dk, Dv,
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        beta.stride(0), beta.stride(1),
        gate.stride(0), gate.stride(1),
    )

    return output, state


# =============================================================================
# PyTorch reference for correctness testing
# =============================================================================

def deltanet_recurrent_step_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor,
) -> tuple:
    """Reference PyTorch DeltaNet recurrent step.
    
    Args:
        q: [B, num_heads, Dk] query vector
        k: [B, num_heads, Dk] key vector
        v: [B, num_heads, Dv] value vector
        beta: [B, num_heads] scaling factor
        gate: [B, num_heads] log gate (applied as exp(gate))
        state: [B, num_heads, Dk, Dv] recurrent state (MODIFIED IN-PLACE)

    Returns:
        output: [B, num_heads, Dv] attention output
        state: same tensor, updated in-place
    """
    B, H, Dk = q.shape
    Dv = v.shape[-1]

    decay = torch.exp(gate).unsqueeze(-1).unsqueeze(-1)
    state = state * decay

    stk = torch.einsum('bhkd,bhk->bhd', state, k) # [B, num_heads, Dv]
    residual = v - stk
    delta = beta.unsqueeze(-1) * residual

    state = state + torch.einsum('bhk,bhd->bhkd', k, delta)
    output = torch.einsum('bhkd,bhk->bhd', state, q)

    return output.to(torch.bfloat16), state


# =============================================================================
# Standalone validation structure
# =============================================================================

NUM_HEADS = 48
DK = 128
DV = 128


class PytorchModel(torch.nn.Module):
    """Reference PyTorch DeltaNet recurrent step."""
    def __init__(self, num_heads: int = NUM_HEADS, dk: int = DK, dv: int = DV):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.register_buffer("state", torch.zeros(1, num_heads, dk, dv, dtype=torch.bfloat16))

    def forward(self, q, k, v, beta, gate):
        output, self.state = deltanet_recurrent_step_pytorch(
            q, k, v, beta, gate, self.state.clone(),
        )
        return output


class TritonModel(torch.nn.Module):
    """Optimized DeltaNet recurrent step using fused Triton kernel."""
    def __init__(self, num_heads: int = NUM_HEADS, dk: int = DK, dv: int = DV):
        super().__init__()
        self.num_heads = num_heads
        self.dk = dk
        self.dv = dv
        self.register_buffer("state", torch.zeros(1, num_heads, dk, dv, dtype=torch.bfloat16))

    def forward(self, q, k, v, beta, gate):
        output, self.state = deltanet_recurrent_step(
            q, k, v, beta, gate, self.state,
        )
        return output


def get_inputs():
    B = 1
    return [
        torch.randn(B, NUM_HEADS, DK, device="cuda", dtype=torch.bfloat16),
        torch.randn(B, NUM_HEADS, DK, device="cuda", dtype=torch.bfloat16),
        torch.randn(B, NUM_HEADS, DV, device="cuda", dtype=torch.bfloat16),
        torch.rand(B, NUM_HEADS, device="cuda", dtype=torch.bfloat16),
        torch.randn(B, NUM_HEADS, device="cuda", dtype=torch.bfloat16) * 0.1,
    ]


def get_init_inputs():
    return [NUM_HEADS, DK, DV]


def benchmark_fn(model, inputs, warmup=100, repeat=1000):
    for _ in range(warmup):
        model(*inputs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        model(*inputs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat * 1000  # microseconds


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 60)
    print("B200 Recurrent Kernel Benchmark")
    print(f"B=1, T=1 (decode), H={NUM_HEADS}, K={DK}, V={DV}")
    print("=" * 60)
    
    model = TritonModel(*get_init_inputs()).cuda()
    inputs = get_inputs()

    us = benchmark_fn(model, inputs)
    print(f"Triton kernel latency: {us:.2f} us")