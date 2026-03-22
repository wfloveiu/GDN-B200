/*
 * Fused DeltaNet Recurrent State Update - CUDA V2: Vectorized Memory Access
 *
 * Optimizations over V1 (vectorization only, same grid layout):
 *   1. float4 (128-bit) vectorized state load/store
 *   2. uint2 (64-bit) vectorized q/k bf16 loads → shared memory (1KB)
 *   3. uint2 (64-bit) vectorized v bf16 loads
 *
 * Same as V1:
 *   - Grid: (NV, B * HV) where NV = V / BV — multi-block per head
 *   - Block: 32 threads (1 warp)
 *   - BV = 4 v-rows per block
 *
 * State layout: [B, HV, V, K] (k-last, K contiguous) — f32
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int BV = 4;
constexpr int KPT = K_DIM / WARP_SIZE;  // 4 floats per thread = 1 float4

__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : __logf(1.0f + __expf(x));
}

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void deltanet_recurrent_v2_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_in,
    const __nv_bfloat16* __restrict__ V_in,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ Dt_bias,
    const __nv_bfloat16* __restrict__ B_gate,
    const float* __restrict__ S,
    float* __restrict__ New_S,
    __nv_bfloat16* __restrict__ O,
    float scale,
    int H,
    int HV
) {
    const int i_v = blockIdx.x;
    const int bh = blockIdx.y;
    const int batch_id = bh / HV;
    const int v_head_id = bh % HV;
    const int head_id = v_head_id / (HV / H);
    const int tid = threadIdx.x;  // 0..31

    // ===== Gate/beta computation =====
    const float a_val = __bfloat162float(A[batch_id * HV + v_head_id]);
    const float dt_bias_val = Dt_bias[v_head_id];
    const float a_log_val = A_log[v_head_id];
    const float b_val = __bfloat162float(B_gate[batch_id * HV + v_head_id]);

    const float x = a_val + dt_bias_val;
    const float sp = softplus_f(x);
    const float log_decay = -__expf(a_log_val) * sp;
    const float decay = __expf(log_decay);
    const float beta = sigmoid_f(b_val);

    // ===== Load q[K] and k[K] into shared memory =====
    __shared__ float s_q[K_DIM];
    __shared__ float s_k[K_DIM];

    const __nv_bfloat16* q_base = Q + batch_id * (H * K_DIM) + head_id * K_DIM;
    const __nv_bfloat16* k_base = K_in + batch_id * (H * K_DIM) + head_id * K_DIM;

    // 32 threads load 128 elements: 4 per thread via uint2 (64-bit) vectorized
    {
        const uint2* q_u2 = reinterpret_cast<const uint2*>(q_base);
        const uint2* k_u2 = reinterpret_cast<const uint2*>(k_base);
        uint2 q_packed = q_u2[tid];  // 64-bit coalesced load: 4 x bf16
        uint2 k_packed = k_u2[tid];  // 64-bit coalesced load: 4 x bf16
        const __nv_bfloat162* q_bf2 = reinterpret_cast<const __nv_bfloat162*>(&q_packed);
        const __nv_bfloat162* k_bf2 = reinterpret_cast<const __nv_bfloat162*>(&k_packed);
        s_q[tid * KPT + 0] = __bfloat162float(q_bf2[0].x) * scale;
        s_q[tid * KPT + 1] = __bfloat162float(q_bf2[0].y) * scale;
        s_q[tid * KPT + 2] = __bfloat162float(q_bf2[1].x) * scale;
        s_q[tid * KPT + 3] = __bfloat162float(q_bf2[1].y) * scale;
        s_k[tid * KPT + 0] = __bfloat162float(k_bf2[0].x);
        s_k[tid * KPT + 1] = __bfloat162float(k_bf2[0].y);
        s_k[tid * KPT + 2] = __bfloat162float(k_bf2[1].x);
        s_k[tid * KPT + 3] = __bfloat162float(k_bf2[1].y);
    }
    // Single warp — no __syncthreads needed

    // ===== State base pointer =====
    const int s_head_offset = batch_id * (HV * V_DIM * K_DIM) + v_head_id * (V_DIM * K_DIM);
    const int v_head_base = batch_id * (HV * V_DIM) + v_head_id * V_DIM;

    // ===== Load v via uint2 (64-bit) =====
    float v_vals[BV];
    {
        const uint2* v_u2 = reinterpret_cast<const uint2*>(V_in + v_head_base + i_v * BV);
        uint2 v_packed = v_u2[0];  // 64-bit load: 4 x bf16
        const __nv_bfloat162* v_bf2 = reinterpret_cast<const __nv_bfloat162*>(&v_packed);
        v_vals[0] = __bfloat162float(v_bf2[0].x);
        v_vals[1] = __bfloat162float(v_bf2[0].y);
        v_vals[2] = __bfloat162float(v_bf2[1].x);
        v_vals[3] = __bfloat162float(v_bf2[1].y);
    }

    // ===== Load state via float4 (128-bit) + apply decay =====
    float s_regs[BV][KPT];

    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = i_v * BV + r;
        if (v_idx < V_DIM) {
            const float4* s_row_f4 = reinterpret_cast<const float4*>(
                S + s_head_offset + v_idx * K_DIM);
            float4 tmp = s_row_f4[tid];  // 128-bit coalesced load
            s_regs[r][0] = tmp.x * decay;
            s_regs[r][1] = tmp.y * decay;
            s_regs[r][2] = tmp.z * decay;
            s_regs[r][3] = tmp.w * decay;
        } else {
            #pragma unroll
            for (int i = 0; i < KPT; i++) s_regs[r][i] = 0.0f;
        }
    }

    // ===== Delta rule =====
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        float partial_stk = 0.0f;
        #pragma unroll
        for (int i = 0; i < KPT; i++) {
            partial_stk += s_regs[r][i] * s_k[tid * KPT + i];
        }
        float stk = warp_reduce_sum(partial_stk);
        float delta = beta * (v_vals[r] - stk);
        #pragma unroll
        for (int i = 0; i < KPT; i++) {
            s_regs[r][i] += delta * s_k[tid * KPT + i];
        }
    }

    // ===== Store state via float4 (128-bit) + compute output =====
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = i_v * BV + r;
        if (v_idx < V_DIM) {
            float4* dst_f4 = reinterpret_cast<float4*>(
                New_S + s_head_offset + v_idx * K_DIM);
            float4 out_f4;
            out_f4.x = s_regs[r][0];
            out_f4.y = s_regs[r][1];
            out_f4.z = s_regs[r][2];
            out_f4.w = s_regs[r][3];
            dst_f4[tid] = out_f4;  // 128-bit coalesced store

            float partial_o = 0.0f;
            #pragma unroll
            for (int i = 0; i < KPT; i++) {
                partial_o += s_regs[r][i] * s_q[tid * KPT + i];
            }
            float o_val = warp_reduce_sum(partial_o);
            if (tid == 0) {
                O[v_head_base + v_idx] = __float2bfloat16(o_val);
            }
        }
    }
}


std::vector<torch::Tensor> deltanet_recurrent_cuda_v2_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor state,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b,
    float scale,
    torch::Tensor output,
    torch::Tensor new_state
) {
    const int B = q.size(0);
    const int H = q.size(2);
    const int HV = v.size(2);
    const int V_val = v.size(3);

    const int NV = (V_val + BV - 1) / BV;

    dim3 grid(NV, B * HV);
    dim3 block(WARP_SIZE);  // 32 threads = 1 warp

    deltanet_recurrent_v2_kernel<<<grid, block>>>(
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
    m.def("forward", &deltanet_recurrent_cuda_v2_forward,
          "DeltaNet recurrent step CUDA V2 vectorized forward (fused gate computation)");
}
