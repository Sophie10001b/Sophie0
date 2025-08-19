import os
import sys

os.environ['TRITON_INTERPRET'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import random
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl
import pdb

from einops import rearrange

# def get_autotune_configs():
#     return [
#         triton.Config(kwargs={'SEQ_CHUNK_SIZE': 64}, num_warps=2, num_stages=2),
#         triton.Config(kwargs={'SEQ_CHUNK_SIZE': 128}, num_warps=4, num_stages=2),
#         triton.Config(kwargs={'SEQ_CHUNK_SIZE': 128}, num_warps=4, num_stages=3),
#         triton.Config(kwargs={'SEQ_CHUNK_SIZE': 128}, num_warps=8, num_stages=3),
#         triton.Config(kwargs={'SEQ_CHUNK_SIZE': 256}, num_warps=8, num_stages=3),
#         triton.Config(kwargs={'SEQ_CHUNK_SIZE': 512}, num_warps=8, num_stages=3),
#     ]

def get_autotune_configs():
    return [
        triton.Config(kwargs={'SEQ_CHUNK_SIZE': 64}, num_warps=2, num_stages=2),
        triton.Config(kwargs={'SEQ_CHUNK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config(kwargs={'SEQ_CHUNK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config(kwargs={'SEQ_CHUNK_SIZE': 128}, num_warps=8, num_stages=3),
        triton.Config(kwargs={'SEQ_CHUNK_SIZE': 256}, num_warps=8, num_stages=3),
        triton.Config(kwargs={'SEQ_CHUNK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config(kwargs={'SEQ_CHUNK_SIZE': 1024}, num_warps=8, num_stages=3),
    ]

@triton.autotune(
    configs=get_autotune_configs(),
    key=['B', 'nQH', 'dH']
)
@triton.jit
def decode_attention_varlen_triton_onepass(
    tQ: tl.tensor, # [total_seqs, heads_num, heads_dim]
    tPrefill: tl.tensor, # [2, total_nnz, heads_num, heads_dim]
    tDecode: tl.tensor, # [2, total_seqs, max_len, heads_num, heads_dim]
    tO: tl.tensor, # [total_seqs, heads_num, heads_dim]
    tCu: tl.tensor, # [total_seqs + 1]
    step: tl.constexpr,
    total_nnz: tl.constexpr,
    B: tl.constexpr,
    nQH: tl.constexpr,
    nKH: tl.constexpr,
    dH: tl.constexpr,
    maxL: tl.constexpr,
    scaled: tl.constexpr,
    SEQ_CHUNK_SIZE: tl.constexpr
):
    # Grid: [total_seqs, num_heads]
    batch_id, head_id = tl.program_id(1), tl.program_id(0)
    nnz = tl.load(tCu + batch_id)
    next_nnz = tl.load(tCu + batch_id + 1)
    seqlen = next_nnz - nnz
    qk_head_map = nQH // nKH

    tQ_ptr = tl.make_block_ptr(
        base=tQ,
        shape=(B, nQH, dH),
        strides=(nQH * dH, dH, 1),
        offsets=(batch_id, head_id, 0),
        block_shape=(1, 1, dH),
        order=(2, 1, 0)
    )
    tKPrefill_ptr = tl.make_block_ptr(
        base=tPrefill,
        shape=(2, total_nnz, nKH, dH),
        strides=(total_nnz * nKH * dH, nKH * dH, dH, 1),
        offsets=(0, nnz.to(tl.int32), head_id // qk_head_map, 0),
        block_shape=(1, SEQ_CHUNK_SIZE, 1, dH),
        order=(3, 2, 1, 0)
    )
    tVPrefill_ptr = tl.advance(tKPrefill_ptr, (1, 0, 0, 0))

    tKDecode_ptr = tl.make_block_ptr(
        base=tDecode,
        shape=(2, B, maxL, nKH, dH),
        strides=(B * maxL * nKH * dH, maxL * nKH * dH, nKH * dH, dH, 1),
        offsets=(0, batch_id, 0, head_id // qk_head_map, 0),
        block_shape=(1, 1, SEQ_CHUNK_SIZE, 1, dH),
        order=(4, 3, 2, 1, 0)
    )
    tVDecode_ptr = tl.advance(tKDecode_ptr, (1, 0, 0, 0, 0))
    
    # load Q
    q = tl.load(tQ_ptr)
    q = tl.reshape(q * scaled, (1, dH))
    pad_q = tl.full((16, dH), 0, dtype=q.dtype)
    q = tl.where(tl.arange(0, 16)[:, None] < q.shape[0], q, pad_q) # [16, dH]

    pad_lse = tl.full((16, SEQ_CHUNK_SIZE), 0, dtype=tl.float32)

    M = tl.full((1,), -float('inf'), dtype=tl.float32)
    LSE = tl.full((1,), 0, dtype=tl.float32)
    O = tl.zeros((1, dH), dtype=tl.float32)

    # get prefill cache
    for i in tl.range(0, seqlen, SEQ_CHUNK_SIZE):
        current_k_ptr = tl.advance(tKPrefill_ptr, (0, i.to(tl.int32), 0, 0))
        current_v_ptr = tl.advance(tVPrefill_ptr, (0, i.to(tl.int32), 0, 0))

        k_prefill = tl.load(current_k_ptr, boundary_check=(1,), padding_option="zero")
        v_prefill = tl.load(current_v_ptr, boundary_check=(1,), padding_option="zero")

        k_prefill = tl.reshape(k_prefill, (SEQ_CHUNK_SIZE, dH))
        v_prefill = tl.reshape(v_prefill, (SEQ_CHUNK_SIZE, dH))

        qk = tl.dot(q.to(tl.bfloat16), tl.trans(k_prefill, 1, 0).to(tl.bfloat16)) # [16, SEQ_CHUNK_SIZE]
        qk = tl.sum(qk, axis=0, keep_dims=True) # [1, SEQ_CHUNK_SIZE]
        qk = tl.where(tl.arange(0, SEQ_CHUNK_SIZE)[None, :] < (seqlen - i), qk, -float('inf'))

        new_M = tl.maximum(tl.max(qk, axis=-1), M)

        exp = tl.exp(M - new_M)
        lse = tl.exp(qk - new_M) # Pij
        new_LSE = LSE * exp + tl.sum(lse, axis=-1) # Li_new

        lse = tl.where(tl.arange(0, 16)[:, None] < lse.shape[0], lse, pad_lse) # [16, SEQ_CHUNK_SIZE]
        
        qkv = tl.dot(lse.to(tl.bfloat16), v_prefill.to(tl.bfloat16))
        qkv = tl.sum(qkv, axis=0, keep_dims=True)
        O = O * exp + qkv

        M = new_M
        LSE = new_LSE

    # get decode cache
    for i in tl.range(0, step, SEQ_CHUNK_SIZE):
        current_k_ptr = tl.advance(tKDecode_ptr, (0, 0, i.to(tl.int32), 0, 0))
        current_v_ptr = tl.advance(tVDecode_ptr, (0, 0, i.to(tl.int32), 0, 0))

        k_decode = tl.load(current_k_ptr, boundary_check=(2,), padding_option="zero")
        v_decode = tl.load(current_v_ptr, boundary_check=(2,), padding_option="zero")

        k_decode = tl.reshape(k_decode, (SEQ_CHUNK_SIZE, dH))
        v_decode = tl.reshape(v_decode, (SEQ_CHUNK_SIZE, dH))
        
        qk = tl.dot(q.to(tl.bfloat16), tl.trans(k_decode, 1, 0).to(tl.bfloat16)) # [16, SEQ_CHUNK_SIZE]
        qk = tl.sum(qk, axis=0, keep_dims=True) # [1, SEQ_CHUNK_SIZE]
        qk = tl.where(tl.arange(0, SEQ_CHUNK_SIZE)[None, :] < (step - i), qk, -float('inf'))

        new_M = tl.maximum(tl.max(qk, axis=-1), M)

        exp = tl.exp(M - new_M)
        lse = tl.exp(qk - new_M)
        new_LSE = LSE * exp + tl.sum(lse, axis=-1)

        lse = tl.where(tl.arange(0, 16)[:, None] < lse.shape[0], lse, pad_lse) # [16, SEQ_CHUNK_SIZE]
        
        qkv = tl.dot(lse.to(tl.bfloat16), v_decode.to(tl.bfloat16))
        qkv = tl.sum(qkv, axis=0, keep_dims=True)
        O = O * exp + qkv

        M = new_M
        LSE = new_LSE

    O = O / LSE
    O = tl.expand_dims(O, axis=0)

    tO_ptr = tl.make_block_ptr(
        base=tO,
        shape=(B, nQH, dH),
        strides=(nQH * dH, dH, 1),
        offsets=(batch_id, head_id, 0),
        block_shape=(1, 1, dH),
        order=(2, 1, 0)
    )

    tl.store(tO_ptr, O.to(tl.bfloat16))

@torch.autocast(device_type='cuda')
def flash_decoding_varlen(
    Q: torch.Tensor,
    prefill_cache: torch.Tensor,
    decode_cache: torch.Tensor,
    prefill_cu_seqlens: torch.Tensor,
    step: int,
    force_no_split: bool = True
) -> torch.Tensor:
    """
    Varlen decoding for inference, with separate prefill cache and decode cache

    Args:
        Q (`[total_seqs, num_q_heads, head_size]`):
            The input query for current step
        prefill_cache (`[2, total_nnz, num_kv_heads, head_size]`)
            The static prefill cache for inference
        decode_cache (`[2, total_seqs, max_len, num_kv_heads, head_size]`)
            The dynamic decode cache for current step
        prefill_cu_seqlens (`[total_seqs + 1]`):
            The cu_seqlens for prefill cache
        step (`int`):
            The current step, must > 0
    """
    
    B, nQH, dH = Q.size()
    total_nnz = prefill_cache.size(1)
    maxL = decode_cache.size(2)
    nKH = decode_cache.size(3)
    scaled = 1 / math.sqrt(dH)

    # max_prefill_len = prefill_cu_seqlens.max().item()
    # num_stages = math.ceil((max_prefill_len + step) / CHUNK_SIZE)

    grid = (nQH, B, 1)
    new_Q = torch.empty_like(Q)
    decode_attention_varlen_triton_onepass[grid](
        Q,
        prefill_cache,
        decode_cache,
        new_Q,
        prefill_cu_seqlens,
        step,
        total_nnz,
        B,
        nQH,
        nKH,
        dH,
        maxL,
        scaled
    )
    return new_Q

# @triton.jit
# def decode_attention_varlen_triton_split_kv(
#     tQ: tl.tensor, # [total_seqs, heads_num, heads_dim]
#     tPrefill: tl.tensor, # [2, total_nnz, heads_num, heads_dim]
#     tDecode: tl.tensor, # [2, total_seqs, max_len, heads_num, heads_dim]
#     tCu: tl.tensor, # [total_seqs + 1]
#     tMAX: tl.tensor, # [total_seqs, heads_num, num_chunks]
#     tLSE: tl.tensor, # [total_seqs, heads_num, num_chunks]
#     tO: tl.tensor, # [total_seqs, heads_num, num_chunks, heads_dim]
#     step: tl.constexpr,
#     total_nnz: tl.constexpr,
#     B: tl.constexpr,
#     nQH: tl.constexpr,
#     nKH: tl.constexpr,
#     dH: tl.constexpr,
#     maxL: tl.constexpr,
#     NUM_CHUNK: tl.constexpr,
#     CHUNK_SIZE: tl.constexpr,
#     SEQ_CHUNK_SIZE: tl.constexpr,
#     scaled: tl.constexpr,
# ):
#     # Grid: [total_seqs, num_heads, num_chunks]
#     chunk_id, head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     nnz = tl.load(tCu + batch_id)
#     next_nnz = tl.load(tCu + batch_id + 1)
#     seqlen = next_nnz - nnz
#     qk_head_map = nQH // nKH
#     seqlen_offset = chunk_id * CHUNK_SIZE
#     total_seqlen = seqlen + step

#     tQ_ptr = tl.make_block_ptr(
#         base=tQ,
#         shape=(B, nQH, dH),
#         strides=(nQH * dH, dH, 1),
#         offsets=(batch_id, head_id, 0),
#         block_shape=(1, 1, dH),
#         order=(2, 1, 0)
#     )
#     tKPrefill_ptr = tl.make_block_ptr(
#         base=tPrefill,
#         shape=(2, total_nnz, nKH, dH),
#         strides=(total_nnz * nKH * dH, nKH * dH, dH, 1),
#         offsets=(0, nnz.to(tl.int32) + seqlen_offset.to(tl.int32), head_id // qk_head_map, 0),
#         block_shape=(1, SEQ_CHUNK_SIZE, 1, dH),
#         order=(3, 2, 1, 0)
#     )
#     tVPrefill_ptr = tl.advance(tKPrefill_ptr, (1, 0, 0, 0))

#     tKDecode_ptr = tl.make_block_ptr(
#         base=tDecode,
#         shape=(2, B, maxL, nKH, dH),
#         strides=(B * maxL * nKH * dH, maxL * nKH * dH, nKH * dH, dH, 1),
#         offsets=(0, batch_id, tl.maximum(seqlen_offset - seqlen, 0).to(tl.int32), head_id // qk_head_map, 0),
#         block_shape=(1, 1, SEQ_CHUNK_SIZE, 1, dH),
#         order=(4, 3, 2, 1, 0)
#     )
#     tVDecode_ptr = tl.advance(tKDecode_ptr, (1, 0, 0, 0, 0))

#     # load Q
#     q = tl.load(tQ_ptr)
#     q = tl.reshape(q * scaled, (1, dH))
#     pad_q = tl.full((16, q.shape[1]), 0, dtype=q.dtype)
#     q = tl.where(tl.arange(0, 16)[:, None] < q.shape[0], q, pad_q) # [16, dH]

#     M = tl.full((1,), -float('inf'), dtype=tl.float32)
#     LSE = tl.full((1,), 0, dtype=tl.float32)
#     O = tl.zeros((1, dH), dtype=tl.float32)

#     # get prefill cache
#     for i in tl.range(seqlen_offset, tl.minimum(seqlen_offset + SEQ_CHUNK_SIZE, seqlen), SEQ_CHUNK_SIZE):
#         current_k_ptr = tl.advance(tKPrefill_ptr, (0, i.to(tl.int32), 0, 0))
#         current_v_ptr = tl.advance(tVPrefill_ptr, (0, i.to(tl.int32), 0, 0))

#         k_prefill = tl.load(current_k_ptr, boundary_check=(1,), padding_option="zero")
#         v_prefill = tl.load(current_v_ptr, boundary_check=(1,), padding_option="zero")

#         k_prefill = tl.reshape(k_prefill, (SEQ_CHUNK_SIZE, dH))
#         v_prefill = tl.reshape(v_prefill, (SEQ_CHUNK_SIZE, dH))

#         qk = tl.dot(q.to(tl.bfloat16), tl.trans(k_prefill, 1, 0).to(tl.bfloat16)) # [16, SEQ_CHUNK_SIZE]
#         qk = tl.sum(qk, axis=0, keep_dims=True) # [1, SEQ_CHUNK_SIZE]
#         qk = tl.where(tl.arange(0, SEQ_CHUNK_SIZE)[None, :] < (seqlen - i), qk, -float('inf'))

#         new_M = tl.maximum(tl.max(qk, axis=-1), M)

#         exp = tl.exp(M - new_M)
#         lse = tl.exp(qk - new_M) # Pij
#         lse = tl.where(tl.arange(0, SEQ_CHUNK_SIZE)[None, :] < seqlen - i, lse, 0)
#         new_LSE = LSE * exp + tl.sum(lse, axis=-1) # Li_new

#         pad_lse = tl.full((16, lse.shape[1]), 0, dtype=lse.dtype)
#         lse = tl.where(tl.arange(0, 16)[:, None] < lse.shape[0], lse, pad_lse) # [16, SEQ_BLOCK]
        
#         qkv = tl.dot(lse.to(tl.bfloat16), v_prefill.to(tl.bfloat16))
#         qkv = tl.sum(qkv, axis=0, keep_dims=True)
#         O = O * exp + qkv

#         M = new_M
#         LSE = new_LSE

#     # get decode cache
#     decode_start = tl.maximum(seqlen_offset - seqlen, 0)
#     for i in tl.range(decode_start, seqlen_offset + SEQ_CHUNK_SIZE - seqlen, SEQ_CHUNK_SIZE):
#         current_k_ptr = tl.advance(tKDecode_ptr, (0, 0, i.to(tl.int32), 0, 0))
#         current_v_ptr = tl.advance(tVDecode_ptr, (0, 0, i.to(tl.int32), 0, 0))

#         k_decode = tl.load(current_k_ptr, boundary_check=(2,), padding_option="zero")
#         v_decode = tl.load(current_v_ptr, boundary_check=(2,), padding_option="zero")

#         k_decode = tl.reshape(k_decode, (SEQ_CHUNK_SIZE, dH))
#         v_decode = tl.reshape(v_decode, (SEQ_CHUNK_SIZE, dH))
        
#         qk = tl.dot(q.to(tl.bfloat16), tl.trans(k_decode, 1, 0).to(tl.bfloat16)) # [16, SEQ_CHUNK_SIZE]
#         qk = tl.sum(qk, axis=0, keep_dims=True) # [1, SEQ_CHUNK_SIZE]
#         qk = tl.where(tl.arange(0, SEQ_CHUNK_SIZE)[None, :] < (step - i), qk, -float('inf'))

#         new_M = tl.maximum(tl.max(qk, axis=-1), M)

#         exp = tl.exp(M - new_M)
#         lse = tl.exp(qk - new_M)
#         lse = tl.where(tl.arange(0, SEQ_CHUNK_SIZE)[None, :] < step - i, lse, 0)
#         new_LSE = LSE * exp + tl.sum(lse, axis=-1)

#         pad_lse = tl.full((16, lse.shape[1]), 0, dtype=lse.dtype)
#         lse = tl.where(tl.arange(0, 16)[:, None] < lse.shape[0], lse, pad_lse) # [16, SEQ_BLOCK]
        
#         qkv = tl.dot(lse.to(tl.bfloat16), v_decode.to(tl.bfloat16))
#         qkv = tl.sum(qkv, axis=0, keep_dims=True)
#         O = O * exp + qkv

#         M = new_M
#         LSE = new_LSE
    
#     tMAX_ptr = tl.make_block_ptr(
#         base=tMAX,
#         shape=(B, nQH, NUM_CHUNK),
#         strides=(nQH * NUM_CHUNK, NUM_CHUNK, 1),
#         offsets=(batch_id, head_id, chunk_id),
#         block_shape=(1, 1, 1),
#         order=(2, 1, 0)
#     )
#     tLSE_ptr = tl.make_block_ptr(
#         base=tLSE,
#         shape=(B, nQH, NUM_CHUNK),
#         strides=(nQH * NUM_CHUNK, NUM_CHUNK, 1),
#         offsets=(batch_id, head_id, chunk_id),
#         block_shape=(1, 1, 1),
#         order=(2, 1, 0)
#     )
#     tO_ptr = tl.make_block_ptr(
#         base=tO,
#         shape=(B, nQH, NUM_CHUNK, dH),
#         strides=(nQH * NUM_CHUNK * dH, NUM_CHUNK * dH, dH, 1),
#         offsets=(batch_id, head_id, chunk_id, 0),
#         block_shape=(1, 1, 1, dH),
#         order=(3, 2, 1, 0)
#     )

#     tl.store(tMAX_ptr, tl.reshape(M, (1, 1, 1)))
#     tl.store(tLSE_ptr, tl.reshape(LSE, (1, 1, 1)))
#     tl.store(tO_ptr, tl.reshape(O, (1, 1, 1, dH)).to(tl.bfloat16))

# @triton.jit
# def decode_attention_varlen_triton_reduce_kv(
#     tQ: tl.tensor, # [total_seqs, heads_num, heads_dim]
#     tMAX: tl.tensor, # [total_seqs, heads_num, num_chunks]
#     tLSE: tl.tensor, # [total_seqs, heads_num, num_chunks]
#     tO: tl.tensor, # [total_seqs, heads_num, num_chunks, heads_dim]
#     B: tl.constexpr,
#     nQH: tl.constexpr,
#     dH: tl.constexpr,
#     NUM_CHUNK: tl.constexpr
# ):
#     # Grid: [total_seqs, num_heads]
#     head_id, batch_id = tl.program_id(0), tl.program_id(1)
#     tQ_ptr = tl.make_block_ptr(
#         base=tQ,
#         shape=(B, nQH, dH),
#         strides=(nQH * dH, dH, 1),
#         offsets=(batch_id, head_id, 0),
#         block_shape=(1, 1, dH),
#         order=(2, 1, 0)
#     )
#     tMAX_ptr = tl.make_block_ptr(
#         base=tMAX,
#         shape=(B, nQH, NUM_CHUNK),
#         strides=(nQH * NUM_CHUNK, NUM_CHUNK, 1),
#         offsets=(batch_id, head_id, 0),
#         block_shape=(1, 1, 1),
#         order=(2, 1, 0)
#     )
#     tLSE_ptr = tl.make_block_ptr(
#         base=tLSE,
#         shape=(B, nQH, NUM_CHUNK),
#         strides=(nQH * NUM_CHUNK, NUM_CHUNK, 1),
#         offsets=(batch_id, head_id, 0),
#         block_shape=(1, 1, 1),
#         order=(2, 1, 0)
#     )
#     tO_ptr = tl.make_block_ptr(
#         base=tO,
#         shape=(B, nQH, NUM_CHUNK, dH),
#         strides=(nQH * NUM_CHUNK * dH, NUM_CHUNK * dH, dH, 1),
#         offsets=(batch_id, head_id, 0, 0),
#         block_shape=(1, 1, 1, dH),
#         order=(3, 2, 1, 0)
#     )

#     M = tl.full((1,), -float('inf'), dtype=tl.float32)
#     LSE = tl.full((1,), 0, dtype=tl.float32)
#     O = tl.zeros((dH,), dtype=tl.float32)

#     for i in tl.range(0, NUM_CHUNK):
#         current_MAX_ptr = tl.advance(tMAX_ptr, (0, 0, i))
#         current_LSE_ptr = tl.advance(tLSE_ptr, (0, 0, i))
#         current_O_ptr = tl.advance(tO_ptr, (0, 0, i, 0))

#         chunk_M = tl.load(current_MAX_ptr).reshape((1,))
#         chunk_LSE = tl.load(current_LSE_ptr).reshape((1,))
#         chunk_O = tl.load(current_O_ptr).reshape((dH,))

#         scale = tl.exp(M - chunk_M)

#         LSE = LSE * scale + chunk_LSE
#         O = O * scale + chunk_O
#         M = chunk_M
    
#     O = O / LSE
#     tl.store(tQ_ptr, tl.reshape(O, (1, 1, dH)).to(tl.bfloat16))
