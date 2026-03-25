# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch
import triton
import triton.language as tl

from utils import prepare_chunk_indices, make_tensor_descriptor, IS_TMA_SUPPORTED, autotune_cache_kwargs, input_guard

FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'ieee')
assert FLA_TRIL_PRECISION in ['ieee', 'tf32', 'tf32x3'], \
    f"FLA_TRIL_PRECISION must be one of 'ieee', 'tf32', or 'tf32x3', but got {FLA_TRIL_PRECISION}"
DOT_PRECISION_AUTOTUNE_LIST = ["ieee"] if not IS_TMA_SUPPORTED else list({"ieee", FLA_TRIL_PRECISION})


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': 'ieee'}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_16x16_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    A = A + (bos*H + i_h) * BT
    Ai = Ai + (bos*H + i_h) * 16

    offset = (i_t * 16) % BT
    if not USE_TMA:
        p_A = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * 16, offset), (16, 16), (1, 0))
        b_A = tl.load(p_A, boundary_check=(0, 1)).to(tl.float32)
        b_A = tl.where(m_A, b_A, 0)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, 16], [H*16, 1], [16, 16])
        b_A = desc.load([i_t * 16, offset]).to(tl.float32)
        b_A = tl.where(m_A, b_A, 0)
    b_A = -b_A

    for i in range(2, min(16, T - i_t * 16)):
        b_a = -tl.load(A + (i_t * 16 + i) * H*BT + o_i + offset)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)
        b_A = tl.where((o_i == i)[:, None], b_a, b_A)
    b_A += m_I
    if not USE_TMA:
        p_Ai = tl.make_block_ptr(Ai, (T, 16), (H*16, 1), (i_t * 16, 0), (16, 16), (1, 0))
        tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * 16, 0], b_A.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a_11 = -tl.load(A + (i_t * BT + i) * H*BT + o_i)
        b_a_11 += tl.sum(b_a_11[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a_11, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a_22 = -tl.load(A + (i_t * BT + i) * H*BT + o_i + 16)
        b_a_22 += tl.sum(b_a_22[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a_22, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)

    if not USE_TMA:
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        tl.store(p_Ai_11, b_Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, b_Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_21, b_Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def solve_tril_64x64_phase1_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Phase 1: solve the two independent 32×32 diagonal blocks within a 64×64 tile.

    Computes the inverse of the top-left and bottom-right 32×32 sub-blocks,
    stores them into Ai. The off-diagonal cross-block is handled by phase 2.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, 16)
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]
    A_base = A + (bos * H + i_h) * BT
    Ai_base = Ai + (bos * H + i_h) * BT

    # ---- Top-left 32×32 block (rows 0-31, cols 0-31) ----
    if not USE_TMA:
        p_A_11 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_A_22 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_A_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_A_22, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc = make_tensor_descriptor(A_base, [T, BT], [H*BT, 1], [16, 16])
        desc_o = make_tensor_descriptor(Ai_base, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_22 = desc.load([i_t * BT + 16, 16]).to(tl.float32)

    b_Ai_11 = -tl.where(m_A, b_Ai_11, 0)
    b_Ai_22 = -tl.where(m_A, b_Ai_22, 0)

    for i in range(2, min(16, T - i_t * BT)):
        b_a = -tl.load(A_base + (i_t * BT + i) * H*BT + o_i)
        b_a = tl.where(o_i < i, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_11, 0)
        b_Ai_11 = tl.where((o_i == i)[:, None], b_a, b_Ai_11)
    for i in range(16 + 2, min(32, T - i_t * BT)):
        b_a = -tl.load(A_base + (i_t * BT + i) * H*BT + o_i + 16)
        b_a = tl.where(o_i < i - 16, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_22, 0)
        b_Ai_22 = tl.where((o_i == i - 16)[:, None], b_a, b_Ai_22)

    b_Ai_11 += m_I
    b_Ai_22 += m_I

    if not USE_TMA:
        p_A_21 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        b_A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_21 = desc.load([i_t * BT + 16, 0]).to(tl.float32)

    b_Ai_21 = -tl.dot(tl.dot(b_Ai_22, b_A_21, input_precision=DOT_PRECISION), b_Ai_11, input_precision=DOT_PRECISION)

    # Store top-left 32×32 inverse
    if not USE_TMA:
        p_o_11 = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_o_21 = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_o_22 = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        tl.store(p_o_11, b_Ai_11.to(p_o_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_21, b_Ai_21.to(p_o_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_22, b_Ai_22.to(p_o_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 0, 0], b_Ai_11.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 0], b_Ai_21.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 16, 16], b_Ai_22.to(desc_o.dtype, fp_downcast_rounding="rtne"))

    # ---- Bottom-right 32×32 block (rows 32-63, cols 32-63) ----
    if not USE_TMA:
        p_A_33 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_A_44 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        b_Ai_33 = tl.load(p_A_33, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_44 = tl.load(p_A_44, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_Ai_33 = desc.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_44 = desc.load([i_t * BT + 48, 48]).to(tl.float32)

    b_Ai_33 = -tl.where(m_A, b_Ai_33, 0)
    b_Ai_44 = -tl.where(m_A, b_Ai_44, 0)

    for i in range(32 + 2, min(48, T - i_t * BT)):
        b_a = -tl.load(A_base + (i_t * BT + i) * H*BT + o_i + 32)
        b_a = tl.where(o_i < i - 32, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_33, 0)
        b_Ai_33 = tl.where((o_i == i - 32)[:, None], b_a, b_Ai_33)
    for i in range(48 + 2, min(64, T - i_t * BT)):
        b_a = -tl.load(A_base + (i_t * BT + i) * H*BT + o_i + 48)
        b_a = tl.where(o_i < i - 48, b_a, 0.)
        b_a += tl.sum(b_a[:, None] * b_Ai_44, 0)
        b_Ai_44 = tl.where((o_i == i - 48)[:, None], b_a, b_Ai_44)

    b_Ai_33 += m_I
    b_Ai_44 += m_I

    if not USE_TMA:
        p_A_43 = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        b_A_43 = tl.load(p_A_43, boundary_check=(0, 1)).to(tl.float32)
    else:
        b_A_43 = desc.load([i_t * BT + 48, 32]).to(tl.float32)

    b_Ai_43 = -tl.dot(tl.dot(b_Ai_44, b_A_43, input_precision=DOT_PRECISION), b_Ai_33, input_precision=DOT_PRECISION)

    # Store bottom-right 32×32 inverse
    if not USE_TMA:
        p_o_33 = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_o_43 = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        p_o_44 = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        tl.store(p_o_33, b_Ai_33.to(p_o_33.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_43, b_Ai_43.to(p_o_43.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_44, b_Ai_44.to(p_o_44.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o.store([i_t * BT + 32, 32], b_Ai_33.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 32], b_Ai_43.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 48], b_Ai_44.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4, 5]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_32x32_to_64x64_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Phase 2: merge two 32×32 inverses (already in Ai) into 64×64.

    Ai already contains the 32×32 block inverses from merge_16x16_to_32x32.
    We read the diagonal blocks from Ai and the off-diagonal A blocks from A,
    then compute the off-diagonal inverse blocks.

    This kernel only holds ~6 live 16×16 matrices instead of 10, reducing
    register pressure from 255 to ~160.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * BT
    Ai += (bos * H + i_h) * BT

    # Read the 32×32 block inverses (already computed by phase 1)
    if not USE_TMA:
        # Top-left 32x32 inverse: blocks (11, 21)
        p_Ai_11 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT, 0), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0))
        # Bottom-right 32x32 inverse: blocks (33, 43)
        p_Ai_33 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0))
        p_Ai_43 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0))
        p_Ai_44 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0))
        b_Ai_11 = tl.load(p_Ai_11, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_21 = tl.load(p_Ai_21, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_22 = tl.load(p_Ai_22, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_33 = tl.load(p_Ai_33, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_43 = tl.load(p_Ai_43, boundary_check=(0, 1)).to(tl.float32)
        b_Ai_44 = tl.load(p_Ai_44, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc_ai = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        b_Ai_11 = desc_ai.load([i_t * BT + 0, 0]).to(tl.float32)
        b_Ai_21 = desc_ai.load([i_t * BT + 16, 0]).to(tl.float32)
        b_Ai_22 = desc_ai.load([i_t * BT + 16, 16]).to(tl.float32)
        b_Ai_33 = desc_ai.load([i_t * BT + 32, 32]).to(tl.float32)
        b_Ai_43 = desc_ai.load([i_t * BT + 48, 32]).to(tl.float32)
        b_Ai_44 = desc_ai.load([i_t * BT + 48, 48]).to(tl.float32)

    # Read the off-diagonal A blocks (original, between top-32 and bottom-32)
    # A_21_big = [[A_31, A_32], [A_41, A_42]]  (in 16×16 sub-blocks)
    if not USE_TMA:
        p_A_31 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_A_32 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_A_41 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_A_42 = tl.make_block_ptr(A, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
        b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
        b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
        b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
    else:
        desc_a = make_tensor_descriptor(A, [T, BT], [H*BT, 1], [16, 16])
        b_A_31 = desc_a.load([i_t * BT + 32, 0]).to(tl.float32)
        b_A_32 = desc_a.load([i_t * BT + 32, 16]).to(tl.float32)
        b_A_41 = desc_a.load([i_t * BT + 48, 0]).to(tl.float32)
        b_A_42 = desc_a.load([i_t * BT + 48, 16]).to(tl.float32)

    # Compute off-diagonal blocks: Ai_bottom_left = -Ai_bottom * A_off * Ai_top
    # Row 3 (32-47): Ai_31 = -(Ai_33*(A_31*Ai_11 + A_32*Ai_21) + 0)
    #                Ai_32 = -(Ai_33*(A_31*0     + A_32*Ai_22) + 0)  -- but A_31*0 is zero for column mapping
    # More precisely, using the 2x2 block formula:
    # [Ai_top,       0   ]   [I,               0   ]^-1
    # [0,       Ai_bottom] * [A_off * Ai_top,  I   ]
    # = [Ai_top, 0; -Ai_bottom * A_off * Ai_top, Ai_bottom]

    # Compute Ai_33 * A_3x (row 3 of bottom-right inverse × off-diag blocks)
    b_t31 = tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION)
    b_t32 = tl.dot(b_A_32, b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_31_new = -tl.dot(b_Ai_33, b_t31, input_precision=DOT_PRECISION)
    b_Ai_32_new = -tl.dot(b_Ai_33, b_t32, input_precision=DOT_PRECISION)

    # Compute Ai_4x: involves both Ai_43 and Ai_44
    b_t41 = tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
    b_t42 = tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
    b_Ai_41_new = -(tl.dot(b_Ai_43, b_t31, input_precision=DOT_PRECISION) + tl.dot(b_Ai_44, b_t41, input_precision=DOT_PRECISION))
    b_Ai_42_new = -(tl.dot(b_Ai_43, b_t32, input_precision=DOT_PRECISION) + tl.dot(b_Ai_44, b_t42, input_precision=DOT_PRECISION))

    # Store the new off-diagonal blocks
    if not USE_TMA:
        p_o_31 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0))
        p_o_32 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0))
        p_o_41 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0))
        p_o_42 = tl.make_block_ptr(Ai, (T, BT), (H*BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0))
        tl.store(p_o_31, b_Ai_31_new.to(p_o_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_32, b_Ai_32_new.to(p_o_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_41, b_Ai_41_new.to(p_o_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_42, b_Ai_42_new.to(p_o_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
    else:
        desc_o = make_tensor_descriptor(Ai, [T, BT], [H*BT, 1], [16, 16])
        desc_o.store([i_t * BT + 32, 0], b_Ai_31_new.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 32, 16], b_Ai_32_new.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 0], b_Ai_41_new.to(desc_o.dtype, fp_downcast_rounding="rtne"))
        desc_o.store([i_t * BT + 48, 16], b_Ai_42_new.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@input_guard
def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_32x32_to_64x64_inverse_kernel

    if BT == 64:
        # Two-phase approach to reduce register pressure (255 → ~160 regs)
        # Phase 1: solve two independent 32×32 blocks along the diagonal
        #   Uses the existing 32×32 kernel but with BT=64 stride awareness
        #   We call the 64×64 phase-1 kernel that inlines the 16×16 solves
        solve_tril_64x64_phase1_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
        # Phase 2: merge the two 32×32 inverses into 64×64 off-diagonal
        merge_32x32_to_64x64_inverse_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
    else:
        merge_fn[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
    return Ai
