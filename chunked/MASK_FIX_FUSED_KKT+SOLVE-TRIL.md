# Padding Fix Analysis: chunk_fwd_intra.py

## Background

When `T < chunk_size(64)`, the fused kernel processes a 64-token chunk but only `T` tokens are valid. The remaining `64 - T` positions are "padding". Two fixes are required to prevent padding from corrupting computation results.

## Overall Layout: A 64x64 matrix composed of 4x4 sub-blocks (BC=16)

```
         col 0-15      col 16-31     col 32-47     col 48-63
        (sub-chunk0)  (sub-chunk1)  (sub-chunk2)  (sub-chunk3)
       +-------------+-------------+-------------+-------------+
row    |             |             |             |             |
0-15   |   b_A00     |      0      |      0      |      0      |
(sc0)  |             |             |             |             |
       +-------------+-------------+-------------+-------------+
row    |             |             |             |             |
16-31  |   b_A10     |   b_A11     |      0      |      0      |
(sc1)  |             |             |             |             |
       +-------------+-------------+-------------+-------------+
row    |             |             |             |             |
32-47  |   b_A20     |   b_A21     |   b_A22     |      0      |
(sc2)  |             |             |             |             |
       +-------------+-------------+-------------+-------------+
row    |             |             |             |             |
48-63  |   b_A30     |   b_A31     |   b_A32     |   b_A33     |
(sc3)  |             |             |             |             |
       +-------------+-------------+-------------+-------------+
```

## When T=6: valid vs padding positions

```
         col 0-15                    col 16-31     col 32-47     col 48-63
       +------+--------+           +-------------+-------------+-------------+
row 0  |@@@@@@|        |           |             |             |             |
row 1  |@@@@@@|        |           |             |             |             |
row 2  |@@@@@@|        |           |  all        |  all        |  all        |
row 3  |@@@@@@|        |           |  padding    |  padding    |  padding    |
row 4  |@@@@@@|        |           |             |             |             |
row 5  |@@@@@@| padding|           |             |             |             |
row 6  |      |        |           |             |             |             |
  ...  |      |        |           |             |             |             |
row 15 |      |        |           |             |             |             |
       +------+--------+           +-------------+-------------+-------------+
row    |               |           |             |             |             |
16-63  |  all padding  |           |  all padding across all sub-chunks      |
       +---------------+           +-------------+-------------+-------------+

@@@@@@ = valid data (6 rows x 6 cols)
blank  = padding
```

---

## Fix 4: `A = torch.zeros(...)` -- inter-kernel padding propagation

### Problem

A has shape `[B, T, H, BT=64]`. When T=6, the fused kernel only writes the first 6 rows (via `boundary_check`). **Rows 7-63 are never written.**

```
  Fused kernel writes to A          wy_fast kernel reads from A
  --------------------------         ----------------------------

  A[T=64, BT=64]                    b_A = tl.load(A[0:64, 0:64])
  +------+--------+                       |
  |correct| kernel |                    b_w = dot(b_A, b_kb)
  |result | writes |
  +------+  zeros  |                 <-- reads rows 6~63
  |??????| (via    |                   torch.empty -> garbage values
  |??????| bound-  |                   torch.zeros -> all 0, safe
  |??????| ary_    |
  |??????| check)  |
  +------+--------+
```

- `torch.empty`: rows 6-63 = VRAM garbage -> `dot(garbage, k) = garbage` -> NaN
- `torch.zeros`: rows 6-63 = 0 -> `dot(0, k) = 0` -> safe

### Why doesn't `boundary_check` protect wy_fast?

wy_fast loads A with `tl.make_block_ptr(A, (T, BT), ...)`. The second dimension is BT=64, which is the full block size -- **there is no out-of-bounds access on the BT dimension**, so `boundary_check` never triggers. The garbage rows 6-63 are read as-is.

### Fix

```python
# Before (line 243)
A = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)

# After
A = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
```

---

## Fix 3: off-diagonal `tl.where` mask -- intra-kernel exp overflow

### Problem

Even with `A = torch.zeros`, the fused kernel has an internal numerical issue. Consider `b_A10` (sub-chunk1 rows x sub-chunk0 cols):

**Step 1: K @ K^T**

```
b_k0 (load sub-chunk0's k, position 0~15):
+------+--------+
|real k |   0    |  <-- boundary_check: pos 6~15 filled with 0
|(0-5)  | (6-15) |
+------+--------+

b_k1 (load sub-chunk1's k, position 16~31):
+-----------------+
|    all zeros    |  <-- boundary_check: pos 16~31 all OOB, filled with 0
+-----------------+

b_A10 = dot(b_k1, b_k0^T)
      = dot(all_zero, anything) = 0    <-- theoretically all zero

BUT with TF32 precision:
      ~ 0.0000001                       <-- may have tiny nonzero values!
```

**Step 2: Gate scaling -- where the problem occurs**

```
g_cumsum values:
b_g0 = [-2.7, -5.1, -3.2, -8.0, -1.5, -9.9,  0,  0,  0, ...,  0]
        |--- valid (negative) -------------|  |--- padding (=0) ---|

b_g1 = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
        |-------------- all padding (=0) ------------------------------|

exp(b_g1[:, None] - b_g0[None, :]):

            b_g0 col:  -2.7   -5.1   -3.2   -8.0   -1.5   -9.9    0     0
b_g1 row: +----------------------------------------------------------------+
    0     |  e^2.7  e^5.1  e^3.2  e^8.0  e^1.5  e^9.9  e^0   e^0   |
    0     | =14.9  =164   =24.5  =2981  =4.5  =19930   =1    =1    |
    0     |  14.9   164    24.5   2981   4.5   19930    1     1     |
   ...    |  ...    ...    ...    ...    ...    ...     ...   ...    |
    0     |  14.9   164    24.5   2981   4.5   19930    1     1     |
          +----------------------------------------------------------------+
                                 ^^^^^
                          huge exp values!
```

**Then multiply by b_A10:**

```
WITHOUT mask:
  b_A10 (theoretically=0, actually~1e-7) x exp(diff)

  1e-7 x 19930 = 0.002          <-- small diff, still OK
  1e-7 x exp(30) = 1e6          <-- if cumsum more extreme, explodes
  0    x inf     = NaN           <-- worst case

WITH mask (Fix 3):
  tl.where(m_tc1[:, None] & m_tc0[None, :], b_A10, 0.)

  m_tc1 = [F, F, F, ..., F]     (pos 16~31 all >= T=6)
  -> entire b_A10 forced to exact 0.0
  -> 0.0 x 19930 = 0.0          <-- exact zero, safe
```

### Fix

```python
# Before (off-diagonal blocks only had beta scaling)
b_A10 = b_A10 * b_b1[:, None]
b_A20 = b_A20 * b_b2[:, None]
b_A21 = b_A21 * b_b2[:, None]
b_A30 = b_A30 * b_b3[:, None]
b_A31 = b_A31 * b_b3[:, None]
b_A32 = b_A32 * b_b3[:, None]

# After (explicit mask before beta scaling)
b_A10 = tl.where(m_tc1[:, None] & m_tc0[None, :], b_A10, 0.) * b_b1[:, None]
b_A20 = tl.where(m_tc2[:, None] & m_tc0[None, :], b_A20, 0.) * b_b2[:, None]
b_A21 = tl.where(m_tc2[:, None] & m_tc1[None, :], b_A21, 0.) * b_b2[:, None]
b_A30 = tl.where(m_tc3[:, None] & m_tc0[None, :], b_A30, 0.) * b_b3[:, None]
b_A31 = tl.where(m_tc3[:, None] & m_tc1[None, :], b_A31, 0.) * b_b3[:, None]
b_A32 = tl.where(m_tc3[:, None] & m_tc2[None, :], b_A32, 0.) * b_b3[:, None]
```

Note: diagonal blocks (b_A00, b_A11, etc.) already had the combined mask `m_d & (m_tcX[:, None] & m_tcX[None, :])` and are unaffected.

---

## Relationship between the two fixes

```
  Inside fused kernel                    Fused kernel -> wy_fast
  ----------------------                 -----------------------

  Fix 3 acts here                        Fix 4 acts here
         |                                       |
         v                                       v
  +-------------+     store      +-------------+     load      +----------+
  | kkt -> gate |------------->  |  A matrix   |-------------> | wy_fast  |
  | -> solve    |                | (HBM)       |               | kernel   |
  |             |                |             |               |          |
  | Fix 3:      |                | Fix 4:      |               | reads    |
  | tl.where    |                | zeros init  |               | clean 0s |
  | prevents    |                | prevents    |               |          |
  | exp overflow|                | garbage     |               |          |
  +-------------+                +-------------+               +----------+
```

- **Without Fix 3**: fused kernel internally computes `tiny_value x exp(huge)` -> writes corrupted values to A
- **Without Fix 4**: even if fused kernel correctly writes first 6 rows, rows 7-63 remain VRAM garbage -> wy_fast reads garbage

**Both fixes are necessary.** Fix 3 prevents corruption during computation; Fix 4 prevents corruption from uninitialized memory.

---

## Why the unfused (separate) kernels DON'T have these problems

### No exp overflow (Fix 3 not needed)

The separate `chunk_scaled_dot_kkt` kernel processes the **entire BT x BT matrix** at once, not in sub-blocks. It applies a unified mask at the very end:

```python
# chunk_scaled_dot_kkt.py line 68-73
b_g_diff = b_g[:, None] - b_g[None, :]    # full 64x64
b_A *= exp(b_g_diff)                        # multiply by exp (may overflow)
b_A *= b_b[:, None]                         # multiply by beta

m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
b_A = tl.where(m_A, b_A, 0)                # FINAL unified mask zeros everything
```

`m_t = o_t < T`, so `m_A` requires **both row AND col < T**. All padding positions are zeroed by `tl.where` in the last step, including any positions where `exp()` produced huge values.

The key difference: the unfused kernel masks **after** all computation in one shot. The fused kernel processes sub-blocks separately and passes intermediate results to solve_tril steps before any unified mask can be applied.

### No garbage from A (Fix 4 not needed)

The unfused pipeline:

```
kkt kernel:      A = torch.empty(...)    <-- padding rows may be garbage
                 kernel writes masked b_A, but only boundary_check range
                 -> A rows 6~63 may contain garbage

solve_tril:      Ai = torch.zeros(...)   <-- OUTPUT initialized with zeros
                 kernel only writes valid range
                 -> Ai padding rows = 0

wy_fast:         reads Ai (NOT A) -> padding rows = 0 -> safe
```

The separate `solve_tril` function (line 474) initializes its output with `Ai = torch.zeros_like(A)`. wy_fast reads `Ai`, not `A`. So the garbage in `A` never reaches wy_fast.
