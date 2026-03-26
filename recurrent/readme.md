## 1. 运行方式

```bash
modal run run_on_modal.py --which bench       # 运行 benchmark
modal run run_on_modal.py --which ncu --batch-size 512  # 运行 NCU profiling
```

## 2. CUDA V3 内核设计与优化点

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

## 3. Benchmark Results (B200, H=8, HV=16, K=128, V=128)

```
====================================================================================================
  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H=8, HV=16, K=128, V=128)
====================================================================================================
 Batch |   Qwen(Triton) |        CUDA_V3 |        CuTeDSL |     V3 vs Qwen
------------------------------------------------------------------------
     1 |          28.24 |           4.36 |          11.35 |    6.5x faster
     4 |          26.61 |           4.32 |          11.33 |    6.2x faster
     8 |          27.28 |           6.15 |          11.34 |    4.4x faster
    16 |          26.76 |           8.20 |          11.40 |    3.3x faster
    32 |          26.87 |          12.29 |          16.43 |    2.2x faster
    64 |          27.02 |          24.58 |          26.66 |    1.1x faster
   128 |          43.43 |          43.10 |          55.41 |    1.0x faster
   256 |          82.74 |          81.96 |         103.68 |    1.0x faster
   512 |         161.00 |         159.04 |         198.63 |    1.0x faster
  1024 |         317.49 |         313.23 |         387.51 |    1.0x faster
------------------------------------------------------------------------
  (us) |   lower=better |   lower=better |   lower=better |
```

**CUDA V3 在所有 batch size 下均优于 Triton 和 CuTe DSL。**
