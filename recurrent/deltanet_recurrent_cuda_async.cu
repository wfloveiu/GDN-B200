/*
 * Fused DeltaNet Recurrent State Update - CUDA with cp.async prefetch
 *
 * Key difference from baseline: uses cp.async to prefetch next state rows
 * into shared memory while computing delta rule on current rows.
 * This overlaps memory latency with compute within the same warp.
 *
 * Pipeline:
 *   1. Prefetch state row[0..BV-1] via cp.async → shared memory
 *   2. For each row r:
 *      a. Wait for row[r] data in shared memory
 *      b. Start prefetch of next batch (if any) — not applicable for BV=4
 *      c. Load from shared → registers, apply decay
 *      d. Delta rule: stk = warp_reduce(s · k), s += beta*(v-stk)*k
 *      e. Store updated state via float4
 *      f. Output: o = warp_reduce(s · q)
 *
 * State layout: [B, HV, V, K] (k-last, K contiguous) — f32
 *
 * Grid: (V/(NUM_WARPS*BV), B*HV) = (8, B*HV)
 * Block: 128 threads (4 warps × 32 lanes)
 * Shared memory: 4 warps × BV rows × K_DIM floats = 4 × 4 × 128 × 4 = 8 KB
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cmath>

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int BV = 4;
constexpr int NUM_WARPS = 4;
constexpr int BV_TOTAL = NUM_WARPS * BV;  // 16 v-rows per block
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 128 threads
constexpr int KPT = K_DIM / WARP_SIZE;   // 4

// Shared memory per warp: BV * K_DIM floats = 4 * 128 = 512 floats = 2 KB
// Total: 4 warps * 2 KB = 8 KB per block
constexpr int SMEM_PER_WARP = BV * K_DIM;  // 512 floats
constexpr int SMEM_TOTAL = NUM_WARPS * SMEM_PER_WARP;  // 2048 floats = 8 KB

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

__global__ void deltanet_recurrent_async_kernel(
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
    // Shared memory for state prefetch — each warp gets its own region
    extern __shared__ float smem[];

    const int i_vb = blockIdx.x;
    const int bh = blockIdx.y;
    const int batch_id = bh / HV;
    const int v_head_id = bh % HV;
    const int head_id = v_head_id / (HV / H);

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Each warp's shared memory region
    float* warp_smem = smem + warp_id * SMEM_PER_WARP;

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

    // ===== Load q, k per-warp via uint2 (64-bit) =====
    const __nv_bfloat16* q_base = Q + batch_id * (H * K_DIM) + head_id * K_DIM;
    const __nv_bfloat16* k_base = K_in + batch_id * (H * K_DIM) + head_id * K_DIM;

    float r_q[KPT], r_k[KPT];
    {
        const uint2* q_u2 = reinterpret_cast<const uint2*>(q_base);
        const uint2* k_u2 = reinterpret_cast<const uint2*>(k_base);
        uint2 q_packed = q_u2[lane_id];
        uint2 k_packed = k_u2[lane_id];
        const __nv_bfloat162* q_bf2 = reinterpret_cast<const __nv_bfloat162*>(&q_packed);
        const __nv_bfloat162* k_bf2 = reinterpret_cast<const __nv_bfloat162*>(&k_packed);
        r_q[0] = __bfloat162float(q_bf2[0].x) * scale;
        r_q[1] = __bfloat162float(q_bf2[0].y) * scale;
        r_q[2] = __bfloat162float(q_bf2[1].x) * scale;
        r_q[3] = __bfloat162float(q_bf2[1].y) * scale;
        r_k[0] = __bfloat162float(k_bf2[0].x);
        r_k[1] = __bfloat162float(k_bf2[0].y);
        r_k[2] = __bfloat162float(k_bf2[1].x);
        r_k[3] = __bfloat162float(k_bf2[1].y);
    }

    // ===== Pointers =====
    const int s_head_offset = batch_id * (HV * V_DIM * K_DIM) + v_head_id * (V_DIM * K_DIM);
    const int v_head_base = batch_id * (HV * V_DIM) + v_head_id * V_DIM;
    const int v_base = i_vb * BV_TOTAL + warp_id * BV;

    // ===== Prefetch ALL BV state rows via cp.async → shared memory =====
    // Each lane copies 16 bytes (float4 = 4 floats) per row
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = v_base + r;
        if (v_idx < V_DIM) {
            // Source: global memory (state row)
            const void* src = reinterpret_cast<const void*>(
                S + s_head_offset + v_idx * K_DIM + lane_id * KPT);
            // Destination: shared memory
            void* dst = reinterpret_cast<void*>(
                warp_smem + r * K_DIM + lane_id * KPT);
            // Async copy 16 bytes (float4)
            __pipeline_memcpy_async(dst, src, sizeof(float4));
        }
    }
    __pipeline_commit();

    // ===== Load v via uint2 (64-bit) — tiny, no need for async =====
    float v_vals[BV];
    {
        const int vb_idx = v_head_base + v_base;
        if (v_base + BV <= V_DIM) {
            const uint2* v_u2 = reinterpret_cast<const uint2*>(V_in + vb_idx);
            uint2 v_packed = v_u2[0];
            const __nv_bfloat162* v_bf2 = reinterpret_cast<const __nv_bfloat162*>(&v_packed);
            v_vals[0] = __bfloat162float(v_bf2[0].x);
            v_vals[1] = __bfloat162float(v_bf2[0].y);
            v_vals[2] = __bfloat162float(v_bf2[1].x);
            v_vals[3] = __bfloat162float(v_bf2[1].y);
        } else {
            #pragma unroll
            for (int r = 0; r < BV; r++) {
                const int v_idx = v_base + r;
                v_vals[r] = (v_idx < V_DIM)
                    ? __bfloat162float(V_in[vb_idx + r]) : 0.0f;
            }
        }
    }

    // ===== Wait for ALL async copies to complete =====
    __pipeline_wait_prior(0);
    // No __syncthreads needed — each warp only reads its own smem region

    // ===== Load from shared memory to registers + apply decay =====
    float s_regs[BV][KPT];
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            s_regs[r][j] = warp_smem[r * K_DIM + lane_id * KPT + j] * decay;
        }
    }

    // ===== Delta rule =====
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        float partial_stk = 0.0f;
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            partial_stk += s_regs[r][j] * r_k[j];
        }
        float stk = warp_reduce_sum(partial_stk);
        float delta = beta * (v_vals[r] - stk);
        #pragma unroll
        for (int j = 0; j < KPT; j++) {
            s_regs[r][j] += delta * r_k[j];
        }
    }

    // ===== Store state via float4 + compute output =====
    #pragma unroll
    for (int r = 0; r < BV; r++) {
        const int v_idx = v_base + r;
        if (v_idx < V_DIM) {
            float4* dst_f4 = reinterpret_cast<float4*>(
                New_S + s_head_offset + v_idx * K_DIM);
            float4 out_f4;
            out_f4.x = s_regs[r][0];
            out_f4.y = s_regs[r][1];
            out_f4.z = s_regs[r][2];
            out_f4.w = s_regs[r][3];
            dst_f4[lane_id] = out_f4;

            float partial_o = 0.0f;
            #pragma unroll
            for (int j = 0; j < KPT; j++) {
                partial_o += s_regs[r][j] * r_q[j];
            }
            float o_val = warp_reduce_sum(partial_o);
            if (lane_id == 0) {
                O[v_head_base + v_idx] = __float2bfloat16(o_val);
            }
        }
    }
}


std::vector<torch::Tensor> deltanet_recurrent_cuda_async_forward(
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

    const int NV_BLOCKS = (V_val + BV_TOTAL - 1) / BV_TOTAL;
    const int smem_bytes = SMEM_TOTAL * sizeof(float);  // 8 KB

    dim3 grid(NV_BLOCKS, B * HV);
    dim3 block(BLOCK_SIZE);

    deltanet_recurrent_async_kernel<<<grid, block, smem_bytes>>>(
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
    m.def("forward", &deltanet_recurrent_cuda_async_forward,
          "DeltaNet recurrent step CUDA with cp.async forward");
}
