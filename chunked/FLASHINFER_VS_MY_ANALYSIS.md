# FlashInfer vs MY: Long Sequence Performance Analysis

## Benchmark Data (B200, Cold Cache)

| Config | T | MY (ms) | FI (ms) | FI/MY | MY µs/tok | FI µs/tok |
|--------|------|---------|---------|-------|-----------|-----------|
| 1×6 | 6 | 0.612 | 0.158 | 0.26× | 102.0 | 26.3 |
| 32×256 | 8192 | 0.709 | 0.216 | 0.30× | 0.087 | 0.026 |
| 1×8192 | 8192 | 0.780 | 0.869 | 1.11× | 0.095 | 0.106 |
| 4×8192 | 32768 | 0.981 | 0.926 | 0.94× | 0.030 | 0.028 |
| 1×65536 | 65536 | 2.809 | 6.116 | 2.18× | 0.043 | 0.093 |
| 4×8192 (16h) | 32768 | 1.172 | 0.910 | 0.78× | 0.036 | 0.028 |
| 1×65536 (16h) | 65536 | 3.110 | 6.217 | 2.00× | 0.047 | 0.095 |

**交叉点：** T ≈ 8192 时两者持平。T < 8192 FlashInfer 快，T > 8192 MY 快。

## 架构差异

| | MY (Triton) | FlashInfer (CuTe-DSL) |
|---|---|---|
| chunk_size | 64 | **128** |
| kernel 数量 | 5 个独立 kernel | **1 个融合 kernel** |
| 硬件特性 | Triton 通用编译 | Blackwell TMA + warp specialization |
| state 递推 | 串行 chunk loop | 串行 chunk loop（融合在 kernel 内） |
| inter-chunk 通信 | 经 HBM 传递 h/state | 寄存器/SMEM 直传 |

## 为什么 FlashInfer 在短序列快

### 1. 单 kernel launch vs 5 kernel launches

MY 的 pipeline 需要 5 次 kernel launch：
```
fused_gdn_gating → cumsum → fused_kkt_solve+wy → chunk_h → chunk_o
```

每次 kernel launch 有 ~3-5µs 开销。5 次 = ~15-25µs 固定成本。

FlashInfer 只有 **1 次 kernel launch**：所有计算（gate、cumsum、intra-chunk attention、inter-chunk state、output）融合在一个 CuTe-DSL kernel 里。

**T=6 时：** MY 的 15-25µs launch 开销占总时间 612µs 的 3-4%。FlashInfer 只有 1 次 launch = ~5µs，总时间 158µs 中 launch 占比更小。

### 2. 零 HBM 中间 round-trip

MY 的 5 个 kernel 之间通过 HBM 传递中间结果（g_cumsum、A、w、u、h、v_new），每个中间 tensor 都要写入 HBM 再读回。

FlashInfer 的单 kernel 在 SMEM/寄存器中直接传递，**完全消除了中间结果的 HBM 读写**。

T=6 时中间结果很小（~几 KB），但 HBM 事务的最小粒度是 32-128 bytes，小数据量的 HBM 访问效率很低。

### 3. Blackwell TMA 硬件加速

FlashInfer 使用 B200 的 Tensor Memory Accelerator (TMA) 进行异步数据搬运：
- `cp.async.bulk.tensor`：硬件级的全局→共享内存拷贝，不占用计算资源
- warp specialization：producer warps 搬数据，consumer warps 做计算，真正的流水线重叠

MY 的 Triton kernel 使用 `tl.make_block_ptr` + `tl.load`，由 Triton 编译器决定是否使用 TMA，无法精细控制 warp 分工。

## 为什么 FlashInfer 在长序列慢

### 1. chunk_size=128 的 intra-chunk 计算是 O(CS²)

GDN chunked 算法中，intra-chunk attention 的计算量为 O(CS² × K)：
- CS=64 (MY)：每 chunk 计算 64² × 128 = 524K FLOPs
- CS=128 (FI)：每 chunk 计算 128² × 128 = **2097K FLOPs（4×）**

虽然 chunk 数量减半（T/128 vs T/64），但 intra-chunk 计算量翻 4 倍，净增 2 倍。

### 2. solve_tril 的 O(CS³) 复杂度

solve_tril（(I+A)^{-1} 的前向替代）复杂度约 O(CS²) 到 O(CS³)：
- CS=64：forward substitution 在 4 个 16×16 sub-block 上，merge 需要 10 次 16×16 matmul
- CS=128：forward substitution 在 8 个 16×16 sub-block 上，merge 需要 **36 次** 16×16 matmul

这是超线性增长，CS 翻倍但 solve 计算量增长 >3×。

### 3. 寄存器压力导致 occupancy 下降

CS=128 的 intra-chunk attention 需要同时持有 128×128 的 A 矩阵 tile（或其分块），这对寄存器和 shared memory 的需求远超 CS=64。

从 MY 的 chunk_size 实验数据（ITERATIONS.md）可以佐证：
```
CS=64:  chunk_o = 161µs (基准)
CS=128: chunk_o = 245µs (+52%)
CS=64:  kkt = 53µs
CS=128: kkt = 122µs (+130%)
```

FlashInfer 将所有计算融合在一个 kernel 里，寄存器压力是**所有阶段的叠加**，occupancy 受限更严重。

### 4. 单 kernel 无法分阶段 autotune

MY 的 5 个 kernel 各自独立 autotune：
- kkt_solve：BK=32, num_warps=1（寄存器优先）
- wy_fast：BK=128, num_warps=4（吞吐优先）
- chunk_o：BK=128, BV=128, num_warps=8（最大并行）

FlashInfer 的单 kernel 只能用一套配置，所有阶段共享相同的 warp 数量和寄存器分配。不同阶段的最优配置冲突时，只能取折中。

### 5. 串行 chunk loop 的总迭代次数

inter-chunk state 递推是串行的（chunk 间有数据依赖），不能并行化：
- MY (CS=64)：T=65536 → 1024 个 chunk → 串行循环 1024 次
- FI (CS=128)：T=65536 → 512 个 chunk → 串行循环 512 次

FlashInfer 循环次数减半，但**每次循环的 intra-chunk 计算量翻 4 倍**。

MY 的 chunk_h kernel 在 CS=64 时占 360µs，CS=128 时降到 263µs（-27%），但其他 kernel 的 regression 吞噬了收益（详见 ITERATIONS.md Iter 22）。FlashInfer 融合后无法分离优化，只能承受 CS=128 的全部代价。

### 6. per-token cost 的关键证据

从 benchmark 数据：

| | MY µs/tok | FI µs/tok | 趋势 |
|---|---|---|---|
| T=8192, 1 seq | 0.095 | 0.106 | 接近 |
| T=32768, 4 seqs | 0.030 | 0.028 | 接近 |
| T=65536, 1 seq | 0.043 | **0.093** | FI 2× 慢 |

关键对比：
- **4×8192 vs 1×65536**：总 token 数相同（32768 vs 65536 差 2×），但 4×8192 有 4 个独立 seq 可以**并行处理不同 seq 的 chunk**
- FlashInfer 在 4×8192 时 0.028 µs/tok，在 1×65536 时 0.093 µs/tok（3.3× 差异）
- MY 在 4×8192 时 0.030 µs/tok，在 1×65536 时 0.043 µs/tok（1.4× 差异）

**FlashInfer 对 seq 并行度更敏感**：单序列长链（1×65536）时，chunk 间串行依赖无法被并行掩盖。多序列（4×8192）时，不同 seq 的 chunk 可以并行执行，延迟被隐藏。

MY 的多 kernel 方案天然支持这种并行——每个 kernel 的 grid 包含所有 (batch × head × chunk) 维度，GPU 可以同时调度不同 chunk 的 block。

## 总结

| 场景 | 胜者 | 原因 |
|------|------|------|
| T ≤ 8192 | **FlashInfer** | 单 kernel launch + TMA + 零 HBM round-trip |
| T > 8192, 多 seq | **接近** | 两者都能并行不同 seq |
| T > 8192, 单 seq | **MY** | CS=64 的 O(CS²) intra-chunk 更轻量 + 独立 autotune |
| T = 65536, 单 seq | **MY 2×** | FlashInfer 的 CS=128 intra-chunk 计算量翻 4× 成为瓶颈 |
