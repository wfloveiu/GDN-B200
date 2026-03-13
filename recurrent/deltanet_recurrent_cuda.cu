/*
 * Fused DeltaNet Recurrent State Update - CUDA Single-Pass Kernel
 *
 * DeltaNet linear attention recurrence (per head, per decode step):
 *     S *= exp(g)                     // gate decay
 *     residual = v - S^T @ k          // delta rule residual
 *     delta = beta * residual
 *     S += outer(k, delta)            // rank-1 state update
 *     o = S^T @ q                     // output query
 *
 * Key optimization vs Triton 2-pass version:
 *   - Single pass: state loaded to SMEM once, all ops done in-place, written back once
 *   - Explicit shared memory: state fully resident in dynamic SMEM (67KB fp32)
 *   - No L2 cache dependency between passes
 *
 * Grid: (num_heads, B)
 * Block: DV threads (128), each thread owns one column of S[DK, DV]
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Fixed dimensions for this model configuration
constexpr int DK = 128;
constexpr int DV = 128;
constexpr int BLOCK_SIZE = DV;  // 128 threads = one per column

// Dynamic shared memory layout (all float32):
//   [0, DK*DV)          : s_state[DK][DV]  (64KB)
//   [DK*DV, DK*DV+DK)   : s_k[DK]          (512B)
//   [DK*DV+DK, ...)      : s_q[DK]          (512B)
//   [DK*DV+2*DK, ...)    : s_delta[DV]      (512B)

__global__ void deltanet_recurrent_kernel(
    const __nv_bfloat16* __restrict__ Q,     // [B, H, DK]
    const __nv_bfloat16* __restrict__ K,     // [B, H, DK]
    const __nv_bfloat16* __restrict__ V,     // [B, H, DV]
    const __nv_bfloat16* __restrict__ Beta,  // [B, H]
    const __nv_bfloat16* __restrict__ Gate,  // [B, H]
    __nv_bfloat16* __restrict__ S,           // [B, H, DK, DV] in-place
    __nv_bfloat16* __restrict__ O,           // [B, H, DV]
    int stride_sb, int stride_sh,
    int stride_qb, int stride_qh,
    int stride_kb, int stride_kh,
    int stride_vb, int stride_vh,
    int stride_ob, int stride_oh,
    int stride_betab, int stride_betah,
    int stride_gb, int stride_gh
) {
    const int head_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int col = threadIdx.x;

    if (col >= DV) return;

    // Dynamic shared memory pointers
    extern __shared__ float smem[];
    float* s_state = smem;                          // [DK * DV]
    float* s_k     = smem + DK * DV;                // [DK]
    float* s_q     = smem + DK * DV + DK;           // [DK]
    float* s_delta = smem + DK * DV + DK + DK;      // [DV]

    // Load scalar inputs
    const float beta = __bfloat162float(
        Beta[batch_id * stride_betab + head_id * stride_betah]);
    const float gate = __bfloat162float(
        Gate[batch_id * stride_gb + head_id * stride_gh]);
    const float decay = expf(gate);

    // Base pointers for this (batch, head)
    const __nv_bfloat16* s_base = S + batch_id * stride_sb + head_id * stride_sh;
    __nv_bfloat16* s_out_base   = S + batch_id * stride_sb + head_id * stride_sh;

    // =========================================================================
    // Step 1: Load state column into SMEM + apply decay
    // Each thread loads its column (128 elements along DK)
    // =========================================================================
    #pragma unroll
    for (int r = 0; r < DK; r++) {
        float val = __bfloat162float(s_base[r * DV + col]);
        s_state[r * DV + col] = val * decay;
    }

    // Cooperatively load k and q vectors (128 threads, 128 elements)
    const __nv_bfloat16* k_base = K + batch_id * stride_kb + head_id * stride_kh;
    const __nv_bfloat16* q_base = Q + batch_id * stride_qb + head_id * stride_qh;

    if (col < DK) {
        s_k[col] = __bfloat162float(k_base[col]);
        s_q[col] = __bfloat162float(q_base[col]);
    }

    // Load v value for this thread's column
    const float v_col = __bfloat162float(
        V[batch_id * stride_vb + head_id * stride_vh + col]);

    __syncthreads();

    // =========================================================================
    // Step 2: Compute S^T @ k for this column
    // stk[col] = sum_r(S[r][col] * k[r])  -- thread-local reduction, no sync
    // =========================================================================
    float stk = 0.0f;
    #pragma unroll
    for (int r = 0; r < DK; r++) {
        stk += s_state[r * DV + col] * s_k[r];
    }

    // =========================================================================
    // Step 3: Compute delta = beta * (v - stk), broadcast via SMEM
    // =========================================================================
    float delta_val = beta * (v_col - stk);
    s_delta[col] = delta_val;
    __syncthreads();

    // =========================================================================
    // Step 4: Rank-1 update + Step 5: Compute S^T @ q (fused)
    // =========================================================================
    float output_val = 0.0f;
    #pragma unroll
    for (int r = 0; r < DK; r++) {
        s_state[r * DV + col] += s_k[r] * s_delta[col];
        output_val += s_state[r * DV + col] * s_q[r];
    }

    // =========================================================================
    // Step 6: Write back state to HBM and store output
    // =========================================================================
    #pragma unroll
    for (int r = 0; r < DK; r++) {
        s_out_base[r * DV + col] = __float2bfloat16(s_state[r * DV + col]);
    }

    O[batch_id * stride_ob + head_id * stride_oh + col] = __float2bfloat16(output_val);
}


// PyTorch C++ extension interface
torch::Tensor deltanet_recurrent_cuda_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor beta,
    torch::Tensor gate,
    torch::Tensor state
) {
    const int B = q.size(0);
    const int H = q.size(1);

    auto output = torch::empty({B, H, DV}, q.options());

    dim3 grid(H, B);
    dim3 block(BLOCK_SIZE);

    // Dynamic shared memory: s_state[DK*DV] + s_k[DK] + s_q[DK] + s_delta[DV]
    // = (128*128 + 128 + 128 + 128) * 4 = 67,072 bytes
    size_t smem_size = (DK * DV + DK + DK + DV) * sizeof(float);

    // H800 (SM90) supports up to 228KB dynamic SMEM per block.
    // Must opt-in for >48KB.
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(
            deltanet_recurrent_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        smem_configured = true;
    }

    deltanet_recurrent_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(state.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        state.stride(0), state.stride(1),
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        output.stride(0), output.stride(1),
        beta.stride(0), beta.stride(1),
        gate.stride(0), gate.stride(1)
    );

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &deltanet_recurrent_cuda_forward,
          "DeltaNet recurrent step CUDA forward (single-pass, dynamic SMEM)");
}
