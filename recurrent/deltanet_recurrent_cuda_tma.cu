/*
 * Fused DeltaNet Recurrent State Update - CUDA with TMA (Tensor Memory Accelerator)
 *
 * Uses TMA to load state tiles [BV, K] directly from global memory to shared
 * memory via a dedicated hardware copy engine, bypassing the L1 cache and
 * freeing up CUDA cores for computation.
 *
 * TMA setup:
 *   - Host creates a CUtensorMap descriptor for state [B*HV*V, K] as 2D tensor
 *   - Kernel loads [BV, K] tiles via cp.async.bulk.tensor.2d
 *   - mbarrier synchronization for TMA completion
 *
 * State layout: [B, HV, V, K] (k-last, K contiguous) — f32
 * Viewed as 2D for TMA: [B*HV*V, K] = [total_rows, 128]
 *
 * Grid: (V/(NUM_WARPS*BV), B*HV) = (8, B*HV)
 * Block: 128 threads (4 warps × 32 lanes)
 * Shared memory: 4 warps × BV × K_DIM floats + 4 mbarriers = ~8 KB + barriers
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cmath>
#include <cassert>

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int WARP_SIZE = 32;
constexpr int BV = 4;
constexpr int NUM_WARPS = 4;
constexpr int BV_TOTAL = NUM_WARPS * BV;  // 16 v-rows per block
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 128 threads
constexpr int KPT = K_DIM / WARP_SIZE;   // 4

// Shared memory layout per warp: BV * K_DIM floats
constexpr int SMEM_PER_WARP = BV * K_DIM;      // 512 floats = 2048 bytes
constexpr int SMEM_FLOATS = NUM_WARPS * SMEM_PER_WARP;  // 2048 floats = 8192 bytes

// TMA tile: [BV, K_DIM] = [4, 128] float32 = 2048 bytes
constexpr int TMA_TILE_BYTES = BV * K_DIM * sizeof(float);  // 2048

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

// ===== TMA helper: initialize mbarrier =====
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t expected_count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(expected_count)
        : "memory"
    );
}

// ===== TMA helper: arrive with expected tx bytes =====
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(tx_bytes)
        : "memory"
    );
}

// ===== TMA helper: issue 2D TMA load =====
__device__ __forceinline__ void tma_load_2d(
    void* smem_ptr,
    const void* tensor_map,
    uint32_t coord_x,  // K dimension coordinate (always 0 for full-K load)
    uint32_t coord_y,  // row coordinate
    uint64_t* mbar
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(smem_ptr)),
          "l"(tensor_map),
          "r"(coord_x),
          "r"(coord_y),
          "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory"
    );
}

// ===== TMA helper: wait on mbarrier =====
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_LOOP:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@!P bra WAIT_LOOP;\n"
        "}\n"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(phase)
        : "memory"
    );
}

__global__ void deltanet_recurrent_tma_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_in,
    const __nv_bfloat16* __restrict__ V_in,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ A,
    const float* __restrict__ Dt_bias,
    const __nv_bfloat16* __restrict__ B_gate,
    const __grid_constant__ CUtensorMap S_tmap,  // TMA descriptor for state (read)
    float* __restrict__ New_S,
    __nv_bfloat16* __restrict__ O,
    float scale,
    int H,
    int HV
) {
    // Shared memory: state tiles + mbarriers
    extern __shared__ char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);

    // mbarriers at the end of shared memory (one per warp, 8-byte aligned)
    // Align to 8 bytes after the float data
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + SMEM_FLOATS);

    const int i_vb = blockIdx.x;
    const int bh = blockIdx.y;
    const int batch_id = bh / HV;
    const int v_head_id = bh % HV;
    const int head_id = v_head_id / (HV / H);

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Each warp's shared memory region
    float* warp_smem = smem + warp_id * SMEM_PER_WARP;

    // ===== Initialize mbarrier (one thread per warp) =====
    if (lane_id == 0) {
        mbarrier_init(&mbar[warp_id], 1);  // expect 1 arrive
    }
    __syncthreads();  // Ensure all mbarriers are initialized

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

    // ===== Compute state row offset for TMA =====
    // State viewed as 2D: [total_rows, K] where total_rows = B * HV * V
    const int v_base = i_vb * BV_TOTAL + warp_id * BV;
    const int row_offset = batch_id * (HV * V_DIM) + v_head_id * V_DIM + v_base;

    // ===== Issue TMA load: [BV, K] tile from global → shared memory =====
    if (lane_id == 0) {
        mbarrier_arrive_expect_tx(&mbar[warp_id], TMA_TILE_BYTES);
        tma_load_2d(
            warp_smem,          // destination in shared memory
            &S_tmap,            // tensor map descriptor
            0,                  // x coordinate (K dim, always 0)
            row_offset,         // y coordinate (row index)
            &mbar[warp_id]
        );
    }

    // ===== Load v via uint2 (overlaps with TMA in-flight) =====
    const int v_head_base = batch_id * (HV * V_DIM) + v_head_id * V_DIM;
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

    // ===== Wait for TMA load to complete =====
    mbarrier_wait(&mbar[warp_id], 0);

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
    const int s_head_offset = batch_id * (HV * V_DIM * K_DIM) + v_head_id * (V_DIM * K_DIM);
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


std::vector<torch::Tensor> deltanet_recurrent_cuda_tma_forward(
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
    const int total_rows = B * HV * V_val;

    // ===== Create TMA tensor map for state [total_rows, K_DIM] =====
    CUtensorMap tensorMap;
    {
        uint64_t globalDim[2]     = {(uint64_t)K_DIM, (uint64_t)total_rows};
        uint64_t globalStrides[1] = {(uint64_t)K_DIM * sizeof(float)};  // stride of dim1 in bytes
        uint32_t boxDim[2]        = {(uint32_t)K_DIM, (uint32_t)BV};    // [128, 4] tile
        uint32_t elemStrides[2]   = {1, 1};

        CUresult res = cuTensorMapEncodeTiled(
            &tensorMap,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,  // rank
            state.data_ptr<float>(),
            globalDim,
            globalStrides,
            boxDim,
            elemStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(res == CUDA_SUCCESS,
                     "cuTensorMapEncodeTiled failed with error ", res);
    }

    const int NV_BLOCKS = (V_val + BV_TOTAL - 1) / BV_TOTAL;
    // Shared memory: state tiles + mbarriers (8-byte aligned)
    const int smem_bytes = SMEM_FLOATS * sizeof(float) + NUM_WARPS * sizeof(uint64_t);

    dim3 grid(NV_BLOCKS, B * HV);
    dim3 block(BLOCK_SIZE);

    // Set max dynamic shared memory if needed
    cudaFuncSetAttribute(
        deltanet_recurrent_tma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );

    deltanet_recurrent_tma_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const float*>(A_log.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const float*>(dt_bias.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        tensorMap,
        reinterpret_cast<float*>(new_state.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        scale,
        H,
        HV
    );

    return {output, new_state};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &deltanet_recurrent_cuda_tma_forward,
          "DeltaNet recurrent step CUDA with TMA forward");
}
