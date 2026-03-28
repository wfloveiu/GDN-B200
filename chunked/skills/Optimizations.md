# MY Optimizations vs FLA Baseline

## 优化清单

### 1. GQA 支持（Hk != H）

**问题：** 官方 FLA kernel 假设 `k.shape[2] == H`，不支持 `num_k_heads=4, num_v_heads=8` 的 GQA 配置。调用前需要 `k.repeat_interleave()` 扩展到 H 头，浪费显存和带宽。

**修改：** 在 `chunk_fwd_intra.py`、`chunk_delta_h.py`、`chunk_o.py`、`wy_fast.py` 中引入 `Hk` 参数，kernel 内部通过 `i_hk = i_h // (H // Hk)` 映射 v-head 到 k-head 索引，用 `Hk*K` 作为 k 的 stride。



### 2. 融合 kernel 的 off-diagonal mask 修复

**问题：** 原始 FLA 融合 kernel 中，off-diagonal blocks（b_A10, b_A20 等）没有 mask padding 区域。当 T < 64 时，padding 位置的 beta=0（boundary_check 填零），但 gate `exp(g_pad - g_valid)` 可能产生巨大值。`0 * huge` 在 bf16 下可能不精确。

**修改：** 在 beta scaling 前，对 off-diagonal blocks 添加显式 `tl.where` mask：
```python
# 原始
b_A10 = b_A10 * b_b1[:, None]
# 修改后
b_A10 = tl.where(m_tc1[:, None] & m_tc0[None, :], b_A10, 0.) * b_b1[:, None]
```

### 3. A 矩阵初始化 torch.zeros

**问题：** 原始使用 `torch.empty` 分配 A 矩阵，padding 区域包含垃圾值。当 T < 64 时 kernel 不会写入所有位置，残留垃圾值导致后续 wy_fast kernel 读到非零 padding。

**修改：** `chunk_fwd_intra.py` 中 `A = torch.zeros(...)` 替代 `torch.empty(...)`。


### 4. recompute_w_u BK=BV=128

**问题：** 原始 FLA 使用 BK=BV=64，当 K=V=128 时需要 2 次内循环迭代。

**修改：** `wy_fast.py` 中改为 `BK=min(K,128), BV=min(V,128)`，K=V=128 时消除内循环。

### 5. Add USE_EXP2

**问题：** 原始 FLA 有 USE_EXP2参数

**修改：** 为MY triton kernel也增加

**结果：**
```
配置	MY exp (ms)	MY exp2 (ms)	exp2/exp	结论
1×6	0.555	0.615	0.90×	exp 更快
32×256	0.705	0.685	1.03×	持平
1×8192	0.720	0.718	1.00×	持平
4×8192	1.227	1.081	1.14×	exp2 更快
1×65536	3.043	3.017	1.01×	持平
1×8192 QK8V16	0.753	0.799	0.94×	exp 更快
4×8192 QK8V16	1.168	1.117	1.05×	微幅
1×65536 QK8V16	3.026	2.992	1.01×	持平
```
但是瓶颈不在这里