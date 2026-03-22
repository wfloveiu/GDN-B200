/*
 * Fused DeltaNet Recurrent State Update - CUDA Warp-Cooperative Kernel
 *
 * Fully fused: raw gate parameters (A_log, a, dt_bias, b) are computed
 * inside the kernel — zero host-side PyTorch overhead.
 *
 * DeltaNet linear attention recurrence (per head, per decode step):
 *     x = a + dt_bias
 *     log_decay = -exp(A_log) * softplus(x)
 *     decay = exp(log_decay)
 *     beta = sigmoid(b)
 *
 *     S *= decay                          // gate decay
 *     residual = v - S^T @ k              // delta rule residual
 *     delta = beta * residual
 *     S += outer(delta, k)                // rank-1 state update
 *     o = S^T @ q                         // output query
 *
 * State layout: [B, HV, V, K] (k-last, K contiguous) — f32
 *
 * Warp-cooperative design:
 *   - 1 block = 1 warp = 32 threads
 *   - Each block handles BV v-rows (BV=4)
 *   - Each thread handles BV * (K/32) = 4 * 4 = 16 f32 state elements
 *   - Reductions (dot products) use __shfl_xor_sync warp shuffles
 *   - q, k vectors in shared memory (256 f32 = 1KB)
 *   - No __syncthreads needed — single warp guarantees lockstep
 *
 * Grid: (NV, B * HV) where NV = V / BV
 * Block: 32 threads (1 warp)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// Compile-time dimensions
constexpr int K_DIM = 128;    // full K dimension
constexpr int V_DIM = 128;    // full V dimension
constexpr int WARP_SIZE = 32; // threads per warp
constexpr int BV = 16;         // v-rows per block
constexpr int KPT = K_DIM / WARP_SIZE;  // K elements Per Thread = 4

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Full warp reduction: sum across 32 lanes
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void deltanet_recurrent_kernel(
    // Input vectors
    const __nv_bfloat16* __restrict__ Q,       // [B, 1, H, K]   bf16
    const __nv_bfloat16* __restrict__ K_in,    // [B, 1, H, K]   bf16
    const __nv_bfloat16* __restrict__ V_in,    // [B, 1, HV, V]  bf16
    // Raw gate parameters
    const float* __restrict__ A_log,           // [HV]           f32
    const __nv_bfloat16* __restrict__ A,       // [B, 1, HV]     bf16
    const float* __restrict__ Dt_bias,         // [HV]           f32
    const __nv_bfloat16* __restrict__ B_gate,  // [B, 1, HV]     bf16
    // State
    const float* __restrict__ S,               // [B, HV, V, K]  f32 input
    float* __restrict__ New_S,                 // [B, HV, V, K]  f32 output
    // Output
    __nv_bfloat16* __restrict__ O,             // [B, 1, HV, V]  bf16
    float scale,
    int H,
    int HV
) {
    // Grid: (NV, B * HV)
    const int i_v = blockIdx.x;                    // which V-block
    const int bh = blockIdx.y;                     // batch_id * HV + v_head_id
    const int batch_id = bh / HV;
    const int v_head_id = bh % HV;
    const int head_id = v_head_id / (HV / H);     // GQA mapping

    const int tid = threadIdx.x;  // 0..31

    // ===== In-kernel gate/beta computation (fused, all threads compute same scalar) =====
    const float a_val = __bfloat162float(A[batch_id * HV + v_head_id]);
    const float dt_bias_val = Dt_bias[v_head_id];
    const float a_log_val = A_log[v_head_id];
    const float b_val = __bfloat162float(B_gate[batch_id * HV + v_head_id]);

    const float x = a_val + dt_bias_val;
    const float sp = softplus_f(x);
    const float log_decay = -__expf(a_log_val) * sp;
    const float decay = __expf(log_decay);
    const float beta = sigmoid_f(b_val);

    // ===== Load q[K] and k[K] into shared memory (cooperative) =====
    __shared__ float s_q[K_DIM];
    __shared__ float s_k[K_DIM];

    const __nv_bfloat16* q_base = Q + batch_id * (H * K_DIM) + head_id * K_DIM;
    const __nv_bfloat16* k_base = K_in + batch_id * (H * K_DIM) + head_id * K_DIM;

    // 32 threads load 128 elements: 4 per thread, fully coalesced
    #pragma unroll
    for (int i = 0; i < KPT; i++) {
        const int idx = tid * KPT + i;
        s_q[idx] = __bfloat162float(q_base[idx]) * scale;
        s_k[idx] = __bfloat162float(k_base[idx]);
    }
    // Single warp — no __syncthreads needed

    // ===== State base pointers =====
    const int s_head_offset = batch_id * (HV * V_DIM * K_DIM) + v_head_id * (V_DIM * K_DIM);

    // ===== Process BV v-rows, each thread handles KPT=4 K-elements per row =====
    // Thread tid owns state elements at K-indices: [tid*KPT .. tid*KPT+KPT-1]
    // for each of BV v-rows

    float s_regs[BV][KPT];  // BV * KPT = 4 * 4 = 16 f32 registers per thread

    // Load BV v-values (only threads 0..BV-1 have meaningful v_val initially,
    // but we broadcast via shuffle)
    float v_vals[BV];

    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = i_v * BV + r;
        if (v_idx < V_DIM) {
            // Only one thread needs to load, then broadcast — but simpler: all threads load
            // (it's a scalar, bf16 load is cheap)
            v_vals[r] = __bfloat162float(V_in[batch_id * (HV * V_DIM) + v_head_id * V_DIM + v_idx]);
        } else {
            v_vals[r] = 0.0f;
        }
    }

    // Load state rows [BV][KPT] + apply decay
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = i_v * BV + r;
        const float* s_row_base = S + s_head_offset + v_idx * K_DIM;
        if (v_idx < V_DIM) {
            #pragma unroll
            for (int i = 0; i < KPT; i++) {
                s_regs[r][i] = s_row_base[tid * KPT + i] * decay;  // ① Gate decay
            }
        } else {
            #pragma unroll
            for (int i = 0; i < KPT; i++) {
                s_regs[r][i] = 0.0f;
            }
        }
    }

    // ===== ②③④ Delta rule for each v-row =====
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        // ② S^T @ k: partial dot product, then warp reduce
        float partial_stk = 0.0f;
        #pragma unroll
        for (int i = 0; i < KPT; i++) {
            partial_stk += s_regs[r][i] * s_k[tid * KPT + i];
        }
        float stk = warp_reduce_sum(partial_stk);  // broadcast to all lanes

        // ③ Delta rule: delta = beta * (v - stk)
        float delta = beta * (v_vals[r] - stk);

        // ④ Rank-1 update: s_row[j] += delta * k[j]
        #pragma unroll
        for (int i = 0; i < KPT; i++) {
            s_regs[r][i] += delta * s_k[tid * KPT + i];
        }
    }

    // ===== Store updated state + compute output =====
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = i_v * BV + r;
        if (v_idx < V_DIM) {
            // Store state row
            float* new_s_row = New_S + s_head_offset + v_idx * K_DIM;
            #pragma unroll
            for (int i = 0; i < KPT; i++) {
                new_s_row[tid * KPT + i] = s_regs[r][i];
            }

            // ⑤ Output: o = dot(s_row, q), partial then warp reduce
            float partial_o = 0.0f;
            #pragma unroll
            for (int i = 0; i < KPT; i++) {
                partial_o += s_regs[r][i] * s_q[tid * KPT + i];
            }
            float o_val = warp_reduce_sum(partial_o);

            // Only lane 0 writes output (all lanes have the value, but avoid duplicate writes)
            if (tid == 0) {
                O[batch_id * (HV * V_DIM) + v_head_id * V_DIM + v_idx] = __float2bfloat16(o_val);
            }
        }
    }
}


// =============================================================================
// PyTorch C++ extension interface
// =============================================================================
std::vector<torch::Tensor> deltanet_recurrent_cuda_forward(
    torch::Tensor q,          // [B, T, H, K]   bf16
    torch::Tensor k,          // [B, T, H, K]   bf16
    torch::Tensor v,          // [B, T, HV, V]  bf16
    torch::Tensor state,      // [B, HV, V, K]  f32
    torch::Tensor A_log,      // [HV]           f32
    torch::Tensor a,          // [B, T, HV]     bf16
    torch::Tensor dt_bias,    // [HV]           f32
    torch::Tensor b,          // [B, T, HV]     bf16
    float scale,
    torch::Tensor output,     // [B, T, HV, V]  bf16 (pre-allocated)
    torch::Tensor new_state   // [B, HV, V, K]  f32  (pre-allocated)
) {
    const int B = q.size(0);
    const int H = q.size(2);
    const int HV = v.size(2);
    const int V_val = v.size(3);

    const int NV = (V_val + BV - 1) / BV;

    dim3 grid(NV, B * HV);
    dim3 block(WARP_SIZE);  // 32 threads = 1 warp

    // Shared memory: s_q[128] + s_k[128] = 1024 bytes
    size_t smem_size = 2 * K_DIM * sizeof(float);

    deltanet_recurrent_kernel<<<grid, block, smem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const float*>(A_log.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const float*>(dt_bias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<const float*>(state.data_ptr()),
        reinterpret_cast<float*>(new_state.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        scale,
        H,
        HV
    );

    return {output, new_state};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &deltanet_recurrent_cuda_forward,
          "DeltaNet recurrent step CUDA warp-cooperative forward (fused gate computation)");
}
