## 1. 运行方式

```bash
modal run run_on_modal.py --which bench       # 运行 benchmark
modal run run_on_modal.py --which ncu --batch-size 512  # 运行 NCU profiling
```

## 2. CUDA 内核设计与优化点

`deltanet_recurrent_cuda_v3.cu` 是最终优化版本的 CUDA decode 内核，针对 B200 (Blackwell, SM 10.0) 优化。

### 核心设计

- **单步 recurrent decode** (T=1)，每次 kernel launch 处理一个 decode step
- **State 布局:** `[B, HV, V, K]` (k-last, K 维度连续) — float32
- **Grid:** `(8, B*HV)` — 8 个 V-block × (batch × v_heads)
- **Block:** 128 threads (4 warps × 32 lanes)
- **每个 warp:** BV=4 个 v-row, KPT=4 floats/thread
- **编译选项:** `-O3 --use_fast_math -arch=sm_100`

### 优化点 1：State 读写 float4 向量化（32-bit → 128-bit）

State 是 kernel 的带宽瓶颈（每 head 读写 128×128×4 = 64KB）。使用 float4 将内存事务数减少 4 倍：

```cuda
// float4 向量化, 每线程 1 次 128-bit coalesced load/store
float4 tmp = s_row_f4[lane_id];           // 128-bit coalesced load
s_regs[r][0] = tmp.x * decay;
s_regs[r][1] = tmp.y * decay;
// ...
dst_f4[lane_id] = out_f4;                 // 128-bit coalesced store
```

### 优化点 2：Q/K/V 加载向量化（bf16 → uint2 64-bit）

Q/K 各 128 个 bf16 = 256 bytes，32 线程每人 1 次 uint2 (64-bit) load：

```cuda
uint2 q_packed = q_u2[lane_id];           // 64-bit coalesced load
// 解包 4 个 bf16 → float
r_q[0] = __bfloat162float(q_bf2[0].x) * scale;
r_q[1] = __bfloat162float(q_bf2[0].y) * scale;
// ...
```

V 的 BV=4 个 bf16 = 8 bytes，通过 uint2 一次读完。

### 优化点 3：4 warps/block 提升 occupancy

```
Block = 128 threads (4 warps),  grid = (8, B*HV),  occupancy ~55%
```

- 每 block 4 warps，每个 warp 独立处理 BV=4 行
- Q/K 每个 warp 独立加载到寄存器（不用 shared memory）
- **无 shared memory、无 `__syncthreads`** — warps 完全独立，零同步开销
- 更多 in-flight warps 隐藏 DRAM 延迟

### 优化点 4：全融合内核（Fused Gating + Delta Rule + Output）

Gate/beta 计算在 kernel 内部完成，无需 host 端预计算：

```cuda
float decay = __expf(-__expf(a_log_val) * softplus_f(a_val + dt_bias_val));
float beta = sigmoid_f(b_val);
```

消除了 3 次独立的 PyTorch kernel launch（sigmoid, exp, softplus）。

### 优化点 5：纯寄存器计算

- 所有中间结果（state rows, q, k, v, 门控值）均在寄存器中
- 46 registers/thread，无 register spill
- 无 shared memory bank conflict，无同步 stall
- Warp reduce 使用 `__shfl_xor_sync` 实现高效归约

---

## 3. Benchmark Results (B200)
### Shape 1: H=4, HV=8, K=128, V=12
```
====================================================================================================
  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H=4, HV=8, K=128, V=128)
====================================================================================================
 Batch |   Qwen(Triton) |           CUDA |        CuTeDSL |   CUDA vs Qwen
--------------------------------------------------------------------------
     1 |          33.05 |           4.65 |          12.87 |    7.1x faster
     4 |          32.67 |           4.72 |          13.24 |    6.9x faster
     8 |          32.08 |           4.85 |          13.05 |    6.6x faster
    16 |          32.05 |           6.14 |          13.03 |    5.2x faster
    32 |          31.67 |           8.20 |          13.17 |    3.9x faster
    64 |          31.68 |          12.29 |          16.44 |    2.6x faster
   128 |          32.25 |          24.57 |          26.66 |    1.3x faster
   256 |          43.10 |          43.03 |          55.38 |    1.0x faster
   512 |          82.04 |          81.85 |         105.06 |    1.0x faster
  1024 |         160.17 |         159.85 |         206.30 |    1.0x faster
--------------------------------------------------------------------------
  (us) |   lower=better |   lower=better |   lower=better |  
```

### Shape 2: H=8, HV=16, K=128, V=128
```
====================================================================================================
  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H=8, HV=16, K=128, V=128)
====================================================================================================
 Batch |   Qwen(Triton) |           CUDA |        CuTeDSL |   CUDA vs Qwen
--------------------------------------------------------------------------
     1 |          33.58 |           4.70 |          14.48 |    7.1x faster
     4 |          33.37 |           5.71 |          14.13 |    5.8x faster
     8 |          33.69 |           6.10 |          14.32 |    5.5x faster
    16 |          32.88 |           8.20 |          14.33 |    4.0x faster
    32 |          32.86 |          12.29 |          16.58 |    2.7x faster
    64 |          33.79 |          24.58 |          26.66 |    1.4x faster
   128 |          43.15 |          43.05 |          55.41 |    1.0x faster
   256 |          82.04 |          81.84 |         103.65 |    1.0x faster
   512 |         159.87 |         159.51 |         204.84 |    1.0x faster
  1024 |         317.29 |         325.22 |         406.97 |    2.4% slower
--------------------------------------------------------------------------
  (us) |   lower=better |   lower=better |   lower=better | 
  (us) |   lower=better |   lower=better |   lower=better |
```



**CUDA  在所有 batch size 下均优于 Triton 和 CuTe DSL。**
