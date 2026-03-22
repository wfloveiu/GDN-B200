## 1. modal 中运行 B200
```
modal run run_on_modal.py --which bench  运行bench  
modal run run_on_modal.py --which ncu --batch-size 512 # 运行ncu
```
`Triton_recurrent.py`是基于FLA 改的

3个cuda文件是基于triton做的优化
## 2. V2 相较于 V1 的优化点

### 优化点 1：State 读写 float4 向量化（32-bit → 128-bit）

```cuda
// V1: 逐个 float 标量读写, 每线程 4 次 32-bit 事务
s_regs[r][i] = s_row_base[tid * KPT + i] * decay;
new_s_row[tid * KPT + i] = s_regs[r][i];

// V2: float4 向量化, 每线程 1 次 128-bit 事务
float4 tmp = s_row_f4[tid];                   // 128-bit coalesced load
s_regs[r][0] = tmp.x * decay;
s_regs[r][1] = tmp.y * decay;
...
dst_f4[tid] = out_f4;                          // 128-bit coalesced store
```

State 是 kernel 的带宽瓶颈（每 head 读写 64KB），float4 将内存事务数减少 4 倍。

### 优化点 2：Q/K 加载 uint2 向量化（16-bit → 64-bit）

```cuda
// V1: 逐个 bf16 标量加载到 shared memory, 每线程 4 次 16-bit
s_q[idx] = __bfloat162float(q_base[idx]);

// V2: uint2 一次读 4 个 bf16, 再解包到 shared memory
uint2 q_packed = q_u2[tid];                    // 64-bit coalesced load
s_q[tid * KPT + 0] = __bfloat162float(q_bf2[0].x) * scale;
s_q[tid * KPT + 1] = __bfloat162float(q_bf2[0].y) * scale;
...
```

Q/K 各 128 个 bf16 = 256 bytes，32 线程每人 1 次 64-bit load，warp 合并为 2 条 cache line。

### 优化点 3：V 加载 uint2 向量化（16-bit → 64-bit）

```cuda
// V1: 逐个 bf16 标量 + 边界检查
v_vals[r] = __bfloat162float(V_in[...+ v_idx]);

// V2: BV=4 个 bf16 = 8 bytes, 一次 uint2 读完
uint2 v_packed = v_u2[0];                      // 64-bit load
v_vals[0] = __bfloat162float(v_bf2[0].x);
...
```

去除了逐元素的 `if (v_idx < V_DIM)` 边界检查。

---

## 3. V3 相较于 V2 的优化点

### 优化点：4 warps / block 提升 occupancy（1 warp → 4 warps）

```
V2: block = 32 threads (1 warp),  grid = (32, B*8),  occupancy ~50%
V3: block = 128 threads (4 warps), grid = (8, B*8),  occupancy ~100%
```

- 每 block 从 1 warp 扩展到 4 warps，每个 warp 独立处理 BV=4 行
- Q/K 每个 warp 独立加载到寄存器（不用 shared memory）
- **无 shared memory、无 `__syncthreads`** — warps 完全独立，零同步开销
- Block 总数减少 4 倍，降低 GPU block 调度压力
- SM occupancy 翻倍，更多 in-flight warps 隐藏 DRAM 延迟

---

## 4. Benchmark Results

### Shape 1: H=4, HV=8, K=128, V=128

```
====================================================================================================
  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H=4, HV=8, K=128, V=128)
====================================================================================================
 Batch |   Qwen(Triton) |        CUDA_V1 |        CUDA_V2 |        CUDA_V3 |     V3 vs Qwen
-------------------------------------------------------------------------------------------
     1 |          32.66 |           6.15 |           4.74 |           4.71 |    6.9x faster
     4 |          32.34 |           6.15 |           4.72 |           4.63 |    7.0x faster
     8 |          32.29 |           6.15 |           4.76 |           4.61 |    7.0x faster
    16 |          31.86 |           7.29 |           6.15 |           6.14 |    5.2x faster
    32 |          32.09 |          10.25 |           8.20 |           8.19 |    3.9x faster
    64 |          32.33 |          18.43 |          13.26 |          10.57 |    3.1x faster
   128 |          32.06 |          34.83 |          24.67 |          24.70 |    1.3x faster
   256 |          43.48 |          63.32 |          46.56 |          45.15 |    3.7% slower
   512 |          83.22 |         115.07 |          88.00 |          85.88 |    3.1% slower
  1024 |         160.88 |         219.20 |         171.59 |         165.98 |    3.1% slower
-------------------------------------------------------------------------------------------
  (us) |   lower=better |   lower=better |   lower=better |   lower=better |
```

### Shape 2: H=8, HV=16, K=128, V=128

```
====================================================================================================
  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H=8, HV=16, K=128, V=128)
====================================================================================================
 Batch |   Qwen(Triton) |        CUDA_V1 |        CUDA_V2 |        CUDA_V3 |     V3 vs Qwen
-------------------------------------------------------------------------------------------
     1 |          32.15 |           6.15 |           4.76 |           4.71 |    6.8x faster
     4 |          32.05 |           6.15 |           4.85 |           4.60 |    7.0x faster
     8 |          31.89 |           8.16 |           6.16 |           6.15 |    5.2x faster
    16 |          32.32 |          10.25 |           8.20 |           8.19 |    3.9x faster
    32 |          32.09 |          18.43 |          12.90 |          10.56 |    3.0x faster
    64 |          32.37 |          34.84 |          25.21 |          24.73 |    1.3x faster
   128 |          43.23 |          62.98 |          45.98 |          45.05 |    4.1% slower
   256 |          82.00 |         114.87 |          87.16 |          85.93 |    4.6% slower
   512 |         161.72 |         219.53 |         171.36 |         168.96 |    4.3% slower
  1024 |         323.07 |         424.92 |         347.93 |         334.68 |    3.5% slower
-------------------------------------------------------------------------------------------
  (us) |   lower=better |   lower=better |   lower=better |   lower=better |
```
