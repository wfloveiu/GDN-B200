# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import torch
import triton
import triton.language as tl

from utils import prepare_chunk_indices, make_tensor_descriptor, IS_TMA_SUPPORTED, autotune_cache_kwargs, input_guard

FLA_TRIL_PRECISION = os.environ.get('FLA_TRIL_PRECISION', 'tf32')
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
def solve_tril_128x128_diag_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,  # 128
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Solve all four 32×32 diagonal blocks within a 128×128 tile.

    Processes diagonal blocks at row offsets 0, 32, 64, 96.
    Each 32×32 is solved as two 16×16 blocks + cross-block merge.
    This is a generalization of solve_tril_64x64_phase1_kernel for BT=128.
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
    base_t = i_t * BT

    # Process 4 diagonal 32×32 blocks at offsets 0, 32, 64, 96
    for blk in range(4):
        row_off = blk * 32

        # Load two 16×16 diagonal sub-blocks
        if not USE_TMA:
            p_A_aa = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (base_t + row_off, row_off), (16, 16), (1, 0))
            p_A_bb = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (base_t + row_off + 16, row_off + 16), (16, 16), (1, 0))
            b_Ai_aa = tl.load(p_A_aa, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_bb = tl.load(p_A_bb, boundary_check=(0, 1)).to(tl.float32)
        else:
            desc = make_tensor_descriptor(A_base, [T, BT], [H*BT, 1], [16, 16])
            desc_o = make_tensor_descriptor(Ai_base, [T, BT], [H*BT, 1], [16, 16])
            b_Ai_aa = desc.load([base_t + row_off, row_off]).to(tl.float32)
            b_Ai_bb = desc.load([base_t + row_off + 16, row_off + 16]).to(tl.float32)

        b_Ai_aa = -tl.where(m_A, b_Ai_aa, 0)
        b_Ai_bb = -tl.where(m_A, b_Ai_bb, 0)

        # Sequential solve for top 16×16
        for i in range(2, min(16, T - base_t - row_off)):
            b_a = -tl.load(A_base + (base_t + row_off + i) * H*BT + o_i + row_off)
            b_a = tl.where(o_i < i, b_a, 0.)
            b_a += tl.sum(b_a[:, None] * b_Ai_aa, 0)
            b_Ai_aa = tl.where((o_i == i)[:, None], b_a, b_Ai_aa)
        # Sequential solve for bottom 16×16
        for i in range(16 + 2, min(32, T - base_t - row_off)):
            b_a = -tl.load(A_base + (base_t + row_off + i) * H*BT + o_i + row_off + 16)
            b_a = tl.where(o_i < i - 16, b_a, 0.)
            b_a += tl.sum(b_a[:, None] * b_Ai_bb, 0)
            b_Ai_bb = tl.where((o_i == i - 16)[:, None], b_a, b_Ai_bb)

        b_Ai_aa += m_I
        b_Ai_bb += m_I

        # Cross-block merge
        if not USE_TMA:
            p_A_ba = tl.make_block_ptr(A_base, (T, BT), (H*BT, 1), (base_t + row_off + 16, row_off), (16, 16), (1, 0))
            b_A_ba = tl.load(p_A_ba, boundary_check=(0, 1)).to(tl.float32)
        else:
            b_A_ba = desc.load([base_t + row_off + 16, row_off]).to(tl.float32)

        b_Ai_ba = -tl.dot(tl.dot(b_Ai_bb, b_A_ba, input_precision=DOT_PRECISION), b_Ai_aa, input_precision=DOT_PRECISION)

        # Store 32×32 inverse
        if not USE_TMA:
            p_o_aa = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (base_t + row_off, row_off), (16, 16), (1, 0))
            p_o_ba = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (base_t + row_off + 16, row_off), (16, 16), (1, 0))
            p_o_bb = tl.make_block_ptr(Ai_base, (T, BT), (H*BT, 1), (base_t + row_off + 16, row_off + 16), (16, 16), (1, 0))
            tl.store(p_o_aa, b_Ai_aa.to(p_o_aa.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            tl.store(p_o_ba, b_Ai_ba.to(p_o_ba.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            tl.store(p_o_bb, b_Ai_bb.to(p_o_bb.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        else:
            desc_o.store([base_t + row_off, row_off], b_Ai_aa.to(desc_o.dtype, fp_downcast_rounding="rtne"))
            desc_o.store([base_t + row_off + 16, row_off], b_Ai_ba.to(desc_o.dtype, fp_downcast_rounding="rtne"))
            desc_o.store([base_t + row_off + 16, row_off + 16], b_Ai_bb.to(desc_o.dtype, fp_downcast_rounding="rtne"))


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
def merge_32x32_to_64x64_within_128_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,  # 128
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Merge 32×32 inverses into 64×64 for both diagonal blocks within a 128×128 tile.

    Processes the top-left 64×64 (rows 0-63) and bottom-right 64×64 (rows 64-127).
    For each: reads the two 32×32 diagonal inverses from Ai, the off-diagonal from A,
    and computes the off-diagonal inverse blocks.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A_ptr = A + (bos * H + i_h) * BT
    Ai_ptr = Ai + (bos * H + i_h) * BT
    base_t = i_t * BT
    stride_row = H * BT

    # Process two 64×64 diagonal blocks
    for blk64 in range(2):
        off64 = blk64 * 64  # 0 or 64

        # Read the two 32×32 inverses (already in Ai)
        # Top-left 32×32: sub-blocks (0,0), (1,0), (1,1)
        if not USE_TMA:
            p_11 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64, off64), (16, 16), (1, 0))
            p_21 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 16, off64), (16, 16), (1, 0))
            p_22 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 16, off64 + 16), (16, 16), (1, 0))
            p_33 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 32, off64 + 32), (16, 16), (1, 0))
            p_43 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 48, off64 + 32), (16, 16), (1, 0))
            p_44 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 48, off64 + 48), (16, 16), (1, 0))
            b_Ai_11 = tl.load(p_11, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_21 = tl.load(p_21, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_22 = tl.load(p_22, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_33 = tl.load(p_33, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_43 = tl.load(p_43, boundary_check=(0, 1)).to(tl.float32)
            b_Ai_44 = tl.load(p_44, boundary_check=(0, 1)).to(tl.float32)
        else:
            desc_ai = make_tensor_descriptor(Ai_ptr, [T, BT], [stride_row, 1], [16, 16])
            b_Ai_11 = desc_ai.load([base_t + off64, off64]).to(tl.float32)
            b_Ai_21 = desc_ai.load([base_t + off64 + 16, off64]).to(tl.float32)
            b_Ai_22 = desc_ai.load([base_t + off64 + 16, off64 + 16]).to(tl.float32)
            b_Ai_33 = desc_ai.load([base_t + off64 + 32, off64 + 32]).to(tl.float32)
            b_Ai_43 = desc_ai.load([base_t + off64 + 48, off64 + 32]).to(tl.float32)
            b_Ai_44 = desc_ai.load([base_t + off64 + 48, off64 + 48]).to(tl.float32)

        # Read off-diagonal A blocks (rows 32-63, cols 0-31 within this 64×64)
        if not USE_TMA:
            p_A_31 = tl.make_block_ptr(A_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 32, off64), (16, 16), (1, 0))
            p_A_32 = tl.make_block_ptr(A_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 32, off64 + 16), (16, 16), (1, 0))
            p_A_41 = tl.make_block_ptr(A_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 48, off64), (16, 16), (1, 0))
            p_A_42 = tl.make_block_ptr(A_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 48, off64 + 16), (16, 16), (1, 0))
            b_A_31 = tl.load(p_A_31, boundary_check=(0, 1)).to(tl.float32)
            b_A_32 = tl.load(p_A_32, boundary_check=(0, 1)).to(tl.float32)
            b_A_41 = tl.load(p_A_41, boundary_check=(0, 1)).to(tl.float32)
            b_A_42 = tl.load(p_A_42, boundary_check=(0, 1)).to(tl.float32)
        else:
            desc_a = make_tensor_descriptor(A_ptr, [T, BT], [stride_row, 1], [16, 16])
            b_A_31 = desc_a.load([base_t + off64 + 32, off64]).to(tl.float32)
            b_A_32 = desc_a.load([base_t + off64 + 32, off64 + 16]).to(tl.float32)
            b_A_41 = desc_a.load([base_t + off64 + 48, off64]).to(tl.float32)
            b_A_42 = desc_a.load([base_t + off64 + 48, off64 + 16]).to(tl.float32)

        # Compute off-diagonal: same formula as merge_32x32_to_64x64_inverse_kernel
        b_t31 = tl.dot(b_A_31, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_32, b_Ai_21, input_precision=DOT_PRECISION)
        b_t32 = tl.dot(b_A_32, b_Ai_22, input_precision=DOT_PRECISION)
        b_Ai_31 = -tl.dot(b_Ai_33, b_t31, input_precision=DOT_PRECISION)
        b_Ai_32 = -tl.dot(b_Ai_33, b_t32, input_precision=DOT_PRECISION)

        b_t41 = tl.dot(b_A_41, b_Ai_11, input_precision=DOT_PRECISION) + tl.dot(b_A_42, b_Ai_21, input_precision=DOT_PRECISION)
        b_t42 = tl.dot(b_A_42, b_Ai_22, input_precision=DOT_PRECISION)
        b_Ai_41 = -(tl.dot(b_Ai_43, b_t31, input_precision=DOT_PRECISION) + tl.dot(b_Ai_44, b_t41, input_precision=DOT_PRECISION))
        b_Ai_42 = -(tl.dot(b_Ai_43, b_t32, input_precision=DOT_PRECISION) + tl.dot(b_Ai_44, b_t42, input_precision=DOT_PRECISION))

        # Store off-diagonal blocks
        if not USE_TMA:
            p_o_31 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 32, off64), (16, 16), (1, 0))
            p_o_32 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 32, off64 + 16), (16, 16), (1, 0))
            p_o_41 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 48, off64), (16, 16), (1, 0))
            p_o_42 = tl.make_block_ptr(Ai_ptr, (T, BT), (stride_row, 1), (base_t + off64 + 48, off64 + 16), (16, 16), (1, 0))
            tl.store(p_o_31, b_Ai_31.to(p_o_31.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            tl.store(p_o_32, b_Ai_32.to(p_o_32.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            tl.store(p_o_41, b_Ai_41.to(p_o_41.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
            tl.store(p_o_42, b_Ai_42.to(p_o_42.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        else:
            desc_o = make_tensor_descriptor(Ai_ptr, [T, BT], [stride_row, 1], [16, 16])
            desc_o.store([base_t + off64 + 32, off64], b_Ai_31.to(desc_o.dtype, fp_downcast_rounding="rtne"))
            desc_o.store([base_t + off64 + 32, off64 + 16], b_Ai_32.to(desc_o.dtype, fp_downcast_rounding="rtne"))
            desc_o.store([base_t + off64 + 48, off64], b_Ai_41.to(desc_o.dtype, fp_downcast_rounding="rtne"))
            desc_o.store([base_t + off64 + 48, off64 + 16], b_Ai_42.to(desc_o.dtype, fp_downcast_rounding="rtne"))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
        for DOT_PRECISION in DOT_PRECISION_AUTOTUNE_LIST
    ],
    key=['H', 'BT', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def merge_64x64_to_128x128_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,  # 128
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Merge two 64×64 inverses (in Ai) into a 128×128 inverse.

    The 128×128 tile is split into 2×2 of 64×64 blocks:
      Ai_TL (rows 0-63, cols 0-63)   — already solved
      Ai_BR (rows 64-127, cols 64-127) — already solved
      A_BL  (rows 64-127, cols 0-63)  — from original A
    We compute:
      Ai_BL = -Ai_BR @ A_BL @ Ai_TL

    We work in 32×32 sub-blocks. The 64×64 blocks are 2×2 grids of 32×32.
    Ai_TL = [[Ai_TL_00, 0       ], [Ai_TL_10, Ai_TL_11]]
    Ai_BR = [[Ai_BR_00, 0       ], [Ai_BR_10, Ai_BR_11]]
    A_BL  = [[A_BL_00,  A_BL_01 ], [A_BL_10,  A_BL_11 ]]

    But 32×32 is too big for direct Triton block ptrs with 16×16 TMA.
    So we use 16×16 sub-blocks throughout, iterating with loops to reduce
    register pressure. We process one 16×16 output sub-block at a time.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A_base = A + (bos * H + i_h) * BT
    Ai_base = Ai + (bos * H + i_h) * BT
    stride_row = H * BT  # stride between rows

    # We need: Ai_BL[r, c] = -sum_k( Ai_BR[r, k] * sum_j(A_BL[k, j] * Ai_TL[j, c]) )
    # where r in [64..127], c in [0..63], k in [64..127], j in [0..63]
    #
    # In 16×16 sub-blocks (indices 0..7 for 128 rows):
    #   Ai_TL sub-blocks: (i,j) for i,j in [0..3]  — lower triangular
    #   Ai_BR sub-blocks: (i,j) for i,j in [4..7]  — lower triangular
    #   A_BL  sub-blocks: (i,j) for i in [4..7], j in [0..3] — dense
    #   Ai_BL sub-blocks: (i,j) for i in [4..7], j in [0..3] — output
    #
    # For each output sub-block Ai_BL(i_r, i_c) where i_r in {4,5,6,7}, i_c in {0,1,2,3}:
    #   Ai_BL(i_r, i_c) = -sum_{k=4}^{7} sum_{j=0}^{3} Ai_BR(i_r, k) * A_BL(k, j) * Ai_TL(j, i_c)
    # But Ai_BR and Ai_TL are lower triangular, so many sub-blocks are zero.
    #
    # First compute T_BL = A_BL @ Ai_TL (temp 64×64, 4×4 sub-blocks)
    # Then Ai_BL = -Ai_BR @ T_BL

    # Process output row by row (i_r = 4,5,6,7) to limit register pressure
    # For each output row i_r, compute all 4 columns at once using 4 accumulators

    base_t = i_t * BT

    for i_r in range(4, 8):
        # Accumulators for 4 output columns [16×16 each]
        b_out_0 = tl.zeros([16, 16], dtype=tl.float32)
        b_out_1 = tl.zeros([16, 16], dtype=tl.float32)
        b_out_2 = tl.zeros([16, 16], dtype=tl.float32)
        b_out_3 = tl.zeros([16, 16], dtype=tl.float32)

        # For each k in the middle sum (over Ai_BR columns = A_BL rows)
        for i_k in range(4, 8):
            # Load Ai_BR(i_r, i_k) — only nonzero if i_r >= i_k (lower triangular)
            if i_r >= i_k:
                p_ai_br = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
                    (base_t + i_r * 16, i_k * 16), (16, 16), (1, 0))
                b_ai_br = tl.load(p_ai_br, boundary_check=(0, 1)).to(tl.float32)

                # For each j in the inner sum (over A_BL columns = Ai_TL rows)
                for i_j in range(4):
                    # Load A_BL(i_k, i_j) — dense, always nonzero
                    p_a_bl = tl.make_block_ptr(A_base, (T, BT), (stride_row, 1),
                        (base_t + i_k * 16, i_j * 16), (16, 16), (1, 0))
                    b_a_bl = tl.load(p_a_bl, boundary_check=(0, 1)).to(tl.float32)

                    # temp = Ai_BR(i_r, i_k) @ A_BL(i_k, i_j)  [16×16]
                    b_temp = tl.dot(b_ai_br, b_a_bl, input_precision=DOT_PRECISION)

                    # Accumulate: for each output column i_c, add temp @ Ai_TL(i_j, i_c)
                    # Ai_TL is lower triangular: nonzero only if i_j >= i_c
                    if i_j >= 0:  # i_c = 0
                        p_ai_tl = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
                            (base_t + i_j * 16, 0 * 16), (16, 16), (1, 0))
                        b_ai_tl = tl.load(p_ai_tl, boundary_check=(0, 1)).to(tl.float32)
                        b_out_0 += tl.dot(b_temp, b_ai_tl, input_precision=DOT_PRECISION)

                    if i_j >= 1:  # i_c = 1
                        p_ai_tl = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
                            (base_t + i_j * 16, 1 * 16), (16, 16), (1, 0))
                        b_ai_tl = tl.load(p_ai_tl, boundary_check=(0, 1)).to(tl.float32)
                        b_out_1 += tl.dot(b_temp, b_ai_tl, input_precision=DOT_PRECISION)

                    if i_j >= 2:  # i_c = 2
                        p_ai_tl = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
                            (base_t + i_j * 16, 2 * 16), (16, 16), (1, 0))
                        b_ai_tl = tl.load(p_ai_tl, boundary_check=(0, 1)).to(tl.float32)
                        b_out_2 += tl.dot(b_temp, b_ai_tl, input_precision=DOT_PRECISION)

                    if i_j >= 3:  # i_c = 3
                        p_ai_tl = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
                            (base_t + i_j * 16, 3 * 16), (16, 16), (1, 0))
                        b_ai_tl = tl.load(p_ai_tl, boundary_check=(0, 1)).to(tl.float32)
                        b_out_3 += tl.dot(b_temp, b_ai_tl, input_precision=DOT_PRECISION)

        # Store -Ai_BL(i_r, 0..3)
        p_o_0 = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
            (base_t + i_r * 16, 0 * 16), (16, 16), (1, 0))
        p_o_1 = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
            (base_t + i_r * 16, 1 * 16), (16, 16), (1, 0))
        p_o_2 = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
            (base_t + i_r * 16, 2 * 16), (16, 16), (1, 0))
        p_o_3 = tl.make_block_ptr(Ai_base, (T, BT), (stride_row, 1),
            (base_t + i_r * 16, 3 * 16), (16, 16), (1, 0))
        tl.store(p_o_0, (-b_out_0).to(p_o_0.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_1, (-b_out_1).to(p_o_1.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_2, (-b_out_2).to(p_o_2.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_o_3, (-b_out_3).to(p_o_3.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


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
    assert A.shape[-1] in [16, 32, 64, 128]
    output_dtype = A.dtype if output_dtype is None else output_dtype

    B, T, H, BT = A.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)

    if BT == 16:
        solve_tril_16x16_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
    elif BT == 32:
        merge_16x16_to_32x32_inverse_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
    elif BT == 64:
        # Two-phase approach to reduce register pressure
        solve_tril_64x64_phase1_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
        merge_32x32_to_64x64_inverse_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
    elif BT == 128:
        # Three-phase approach for 128×128:
        # Phase 1: solve all four 32×32 diagonal blocks
        solve_tril_128x128_diag_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
        # Phase 2: merge 32→64 for both diagonal 64×64 blocks
        merge_32x32_to_64x64_within_128_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
        # Phase 3: merge two 64×64 inverses into 128×128 off-diagonal
        merge_64x64_to_128x128_inverse_kernel[NT, B * H](
            A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            T=T, H=H, BT=BT, USE_TMA=IS_TMA_SUPPORTED,
        )
    return Ai
