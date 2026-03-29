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


## 3. Benchmark Results (B200)
### Shape 1: H=4, HV=8, K=128, V=12
```
========================================================================================================================
  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H=4, HV=8, K=128, V=128)
========================================================================================================================
 Batch |  Data(MB) |       Triton |         CUDA |      CuTeDSL |    CUDA BW |  Util% |  CUDA/Triton
----------------------------------------------------------------------------------------------------
     1 |     1.05 |        44.87 |        13.08 |        20.41 |       81 GB/s |   1.0% |  3.4x faster
     4 |     4.22 |        44.56 |        12.45 |        20.17 |      339 GB/s |   4.2% |  3.6x faster
     8 |     8.44 |        45.29 |        13.23 |        20.65 |      638 GB/s |   8.0% |  3.4x faster
    16 |    16.88 |        45.51 |        12.75 |        21.83 |     1324 GB/s |  16.6% |  3.6x faster
    32 |    33.75 |        45.97 |        13.04 |        26.96 |     2589 GB/s |  32.4% |  3.5x faster
    64 |    67.50 |        48.69 |        18.70 |        32.87 |     3609 GB/s |  45.1% |  2.6x faster
   128 |   135.01 |        59.57 |        29.27 |        46.50 |     4612 GB/s |  57.6% |  2.0x faster
   256 |   270.02 |        78.74 |        48.33 |        69.52 |     5586 GB/s |  69.8% |  1.6x faster
   512 |   540.03 |       118.41 |        86.70 |       117.19 |     6229 GB/s |  77.9% |  1.4x faster
  1024 |  1080.07 |       196.17 |       163.54 |       213.21 |     6604 GB/s |  82.6% |  1.2x faster
----------------------------------------------------------------------------------------------------
  (us) |           | lower=better | lower=better | lower=better |            |        | 
```

### Shape 2: H=8, HV=16, K=128, V=128
```
========================================================================================================================
  DeltaNet Recurrent Kernel Benchmark (B200, T=1, H=8, HV=16, K=128, V=128)
========================================================================================================================
 Batch |  Data(MB) |       Triton |         CUDA |      CuTeDSL |    CUDA BW |  Util% |  CUDA/Triton
----------------------------------------------------------------------------------------------------
     1 |     2.11 |        42.30 |        12.12 |        18.75 |      174 GB/s |   2.2% |  3.5x faster
     4 |     8.44 |        41.01 |        11.68 |        19.47 |      723 GB/s |   9.0% |  3.5x faster
     8 |    16.88 |        41.49 |        11.63 |        22.00 |     1451 GB/s |  18.1% |  3.6x faster
    16 |    33.75 |        42.38 |        12.71 |        26.23 |     2657 GB/s |  33.2% |  3.3x faster
    32 |    67.50 |        45.68 |        18.35 |        32.19 |     3678 GB/s |  46.0% |  2.5x faster
    64 |   135.01 |        56.20 |        28.72 |        45.66 |     4701 GB/s |  58.8% |  2.0x faster
   128 |   270.02 |        76.56 |        48.23 |        72.15 |     5598 GB/s |  70.0% |  1.6x faster
   256 |   540.03 |       127.25 |        86.68 |       116.29 |     6230 GB/s |  77.9% |  1.5x faster
   512 |  1080.07 |       194.72 |       163.34 |       211.46 |     6612 GB/s |  82.7% |  1.2x faster
  1024 |  2160.13 |       353.08 |       317.35 |       410.41 |     6807 GB/s |  85.1% |  1.1x faster
----------------------------------------------------------------------------------------------------
  (us) |           | lower=better | lower=better | lower=better |            |        |             
```

