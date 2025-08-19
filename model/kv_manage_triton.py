import os
import sys

os.environ['TRITON_INTERPRET'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import random
import math
import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl
import pdb

from functools import partial
from lightning import seed_everything
from triton.testing import do_bench
from einops import rearrange
from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any

from flash_attn import flash_attn_varlen_kvpacked_func

# @triton.jit
# def varlen_kv_manage_fwd(
#     past_kv: tl.tensor,
#     past_kv_cu_seqlens: tl.tensor,
#     new_kv: tl.tensor,
#     new_kv_cu_seqlens: tl.tensor,
#     final_kv: tl.tensor,
#     batch_id_list: tl.tensor,
#     chunk_id_list: tl.tensor,
#     pNNZ: tl.constexpr,
#     nNNZ: tl.constexpr,
#     nH: tl.constexpr,
#     dH: tl.constexpr,
#     B: tl.constexpr,
#     CHUNK_SIZE: tl.constexpr
# ):
#     chunk_id, head_id, kv_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
#     batch_id = tl.load(batch_id_list + chunk_id)
#     inner_chunk_id = tl.load(chunk_id_list + chunk_id)
#     chunk_start = inner_chunk_id * CHUNK_SIZE

#     past_kv_nnz, new_kv_nnz = tl.load(past_kv_cu_seqlens + batch_id), tl.load(new_kv_cu_seqlens + batch_id)
#     next_past_kv_nnz, next_new_kv_nzz = tl.load(past_kv_cu_seqlens + batch_id + 1), tl.load(new_kv_cu_seqlens + batch_id + 1)
#     past_len, new_len = next_past_kv_nnz - past_kv_nnz, next_new_kv_nzz - new_kv_nnz

#     past_kv_start = tl.minimum(chunk_start, past_len)
#     past_kv_total = tl.minimum(CHUNK_SIZE, past_len - past_kv_start)
#     new_kv_total = tl.minimum(CHUNK_SIZE - past_kv_total, new_len)
#     new_kv_start = (chunk_start - past_len).to(tl.int32) if chunk_start > past_len else 0

#     zeros = tl.zeros((CHUNK_SIZE, dH), dtype=tl.int32)

#     if past_kv_total > 0:
#         past_kv_ptr = past_kv + kv_id * pNNZ * nH * dH + (past_kv_nnz + chunk_start) * nH * dH + head_id * dH
#         final_kv_ptr = final_kv + kv_id * (pNNZ + nNNZ) * nH * dH + (past_kv_nnz + new_kv_nnz + chunk_start) * nH * dH + head_id * dH

#         past_kv_ptr += tl.arange(0, CHUNK_SIZE)[:, None] * nH * dH + tl.arange(0, dH)[None, :]
#         final_kv_ptr += tl.arange(0, CHUNK_SIZE)[:, None] * nH * dH + tl.arange(0, dH)[None, :]

#         past_kv_data = tl.load(past_kv_ptr, mask=(tl.arange(0, CHUNK_SIZE) + chunk_start < past_len)[:, None], other=zeros)
#         tl.store(final_kv_ptr, past_kv_data, mask=(tl.arange(0, CHUNK_SIZE) + chunk_start < past_len)[:, None])
    
#     if new_kv_total > 0:
#         new_kv_ptr = new_kv + kv_id * nNNZ * nH * dH + (new_kv_nnz + new_kv_start) * nH * dH + head_id * dH
#         final_kv_ptr = final_kv + kv_id * (pNNZ + nNNZ) * nH * dH + (past_kv_nnz + new_kv_nnz + past_len + new_kv_start) * nH * dH + head_id * dH

#         new_kv_ptr += tl.arange(0, CHUNK_SIZE)[:, None] * nH * dH + tl.arange(0, dH)[None, :]
#         final_kv_ptr += tl.arange(0, CHUNK_SIZE)[:, None] * nH * dH + tl.arange(0, dH)[None, :]

#         new_kv_data = tl.load(new_kv_ptr, mask=(tl.arange(0, CHUNK_SIZE) + new_kv_start < new_len)[:, None], other=zeros)
#         tl.store(final_kv_ptr, new_kv_data, mask=(tl.arange(0, CHUNK_SIZE) + new_kv_start < new_len)[:, None])

# @torch.autocast(device_type='cuda')
# def varlen_kv_manage(
#     past_kv: torch.Tensor,
#     past_cu_seqlens: torch.Tensor,
#     new_kv: torch.Tensor,
#     new_cu_seqlens: torch.Tensor
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Varlen decoding for inference, with separate prefill cache and decode cache

#     Args:
#         past_kv (`[2, total_nnz, num_kv_heads, head_size]`)
#         past_cu_seqlens (`[total_seqs + 1]`):
#         new_kv (`[2, total_nnz, num_kv_heads, head_size]`)
#         new_cu_seqlens (`[total_seqs + 1]`):
#     """

#     final_cu_seqlens = past_cu_seqlens + new_cu_seqlens
#     final_kv = torch.empty((2, final_cu_seqlens[-1], *past_kv.shape[2:]), device=past_kv.device, dtype=past_kv.dtype)

#     B, nH, dH = past_cu_seqlens.shape[0] - 1, past_kv.shape[2], past_kv.shape[3]
#     pNNZ, nNNZ = past_kv.shape[1], new_kv.shape[1]

#     CHUNK_SIZE = 1024
#     final_seqlen = final_cu_seqlens[1:] - final_cu_seqlens[:-1]
#     chunk_list = (final_seqlen + CHUNK_SIZE - 1) // CHUNK_SIZE
#     chunk_list_cpu = chunk_list.cpu().numpy()
#     total_chunk = chunk_list.sum().item()

#     batch_id_list = torch.arange(batch_size, device=device, dtype=torch.int64).repeat_interleave(chunk_list)
#     chunk_id_list = np.concat([np.arange(chunk_list_cpu[i]) for i in range(batch_size)], axis=0)
#     chunk_id_list = torch.tensor(chunk_id_list, dtype=torch.int64, device=device)

#     grid = (total_chunk, nH, 2)
#     varlen_kv_manage_fwd[grid](
#         past_kv,
#         past_cu_seqlens,
#         new_kv,
#         new_cu_seqlens,
#         final_kv,
#         batch_id_list,
#         chunk_id_list,
#         pNNZ,
#         nNNZ,
#         nH,
#         dH,
#         B,
#         CHUNK_SIZE
#     )

#     return final_kv, final_cu_seqlens

def get_autotune_configs():
    return [
        triton.Config(kwargs={'PAST_CHUNK_SIZE': 4}, num_warps=4, num_stages=2),
        triton.Config(kwargs={'PAST_CHUNK_SIZE': 8}, num_warps=4, num_stages=2),
        triton.Config(kwargs={'PAST_CHUNK_SIZE': 16}, num_warps=4, num_stages=2),
        triton.Config(kwargs={'PAST_CHUNK_SIZE': 32}, num_warps=4, num_stages=2),
        triton.Config(kwargs={'PAST_CHUNK_SIZE': 64}, num_warps=4, num_stages=2),
    ]

@triton.autotune(
    configs=get_autotune_configs(),
    key=['B', 'nH']
)
@triton.jit
def varlen_kv_manage_fwd(
    past_kv: tl.tensor,
    past_kv_cu_seqlens: tl.tensor,
    new_kv: tl.tensor,
    new_kv_cu_seqlens: tl.tensor,
    final_kv: tl.tensor,
    pNNZ: tl.constexpr,
    nNNZ: tl.constexpr,
    nH: tl.constexpr,
    dH: tl.constexpr,
    B: tl.constexpr,
    NEW_CHUNK_SIZE: tl.constexpr,
    PAST_CHUNK_SIZE: tl.constexpr
):
    head_id, batch_id, kv_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    past_kv_nnz, new_kv_nnz = tl.load(past_kv_cu_seqlens + batch_id), tl.load(new_kv_cu_seqlens + batch_id)
    next_past_kv_nnz, next_new_kv_nzz = tl.load(past_kv_cu_seqlens + batch_id + 1), tl.load(new_kv_cu_seqlens + batch_id + 1)
    past_len, new_len = next_past_kv_nnz - past_kv_nnz, next_new_kv_nzz - new_kv_nnz

    past_kv_nnz = past_kv_nnz.to(tl.int32)
    new_kv_nnz = new_kv_nnz.to(tl.int32)

    past_kv_ptr = tl.make_block_ptr(
        base=past_kv,
        shape=(2, pNNZ, nH, dH),
        strides=(pNNZ * nH * dH, nH * dH, dH, 1),
        offsets=(kv_id, past_kv_nnz, head_id, 0),
        block_shape=(1, PAST_CHUNK_SIZE, 1, dH),
        order=(3, 2, 1, 0)
    )
    new_kv_ptr = tl.make_block_ptr(
        base=new_kv,
        shape=(2, nNNZ, nH, dH),
        strides=(nNNZ * nH * dH, nH * dH, dH, 1),
        offsets=(kv_id, new_kv_nnz, head_id, 0),
        block_shape=(1, NEW_CHUNK_SIZE, 1, dH),
        order=(3, 2, 1, 0)
    )

    final_kv_ptr = final_kv + kv_id * (pNNZ + nNNZ) * nH * dH + (past_kv_nnz + new_kv_nnz) * nH * dH + head_id * dH
    for i in tl.range(0, past_len, PAST_CHUNK_SIZE):
        current_kv_ptr = tl.advance(past_kv_ptr, (0, i.to(tl.int32), 0, 0))
        target_kv_ptr = final_kv_ptr + (i + tl.arange(0, PAST_CHUNK_SIZE))[:, None] * nH * dH + tl.arange(0, dH)[None, :]

        current_kv = tl.load(current_kv_ptr, boundary_check=(1,), padding_option="zero")
        tl.store(target_kv_ptr, current_kv.reshape(PAST_CHUNK_SIZE, dH), mask=(tl.arange(0, PAST_CHUNK_SIZE) + i < past_len)[:, None])
    
    final_kv_ptr = final_kv_ptr + past_len.to(tl.int32) * nH * dH
    for i in tl.range(0, new_len, NEW_CHUNK_SIZE):
        current_kv_ptr = tl.advance(new_kv_ptr, (0, i.to(tl.int32), 0, 0))
        target_kv_ptr = final_kv_ptr + (i + tl.arange(0, NEW_CHUNK_SIZE))[:, None] * nH * dH + tl.arange(0, dH)[None, :]

        current_kv = tl.load(current_kv_ptr, boundary_check=(1,), padding_option="zero")
        tl.store(target_kv_ptr, current_kv.reshape(NEW_CHUNK_SIZE, dH), mask=(tl.arange(0, NEW_CHUNK_SIZE) + i < new_len)[:, None])

@torch.autocast(device_type='cuda')
def varlen_kv_manage(
    past_kv: torch.Tensor,
    past_cu_seqlens: torch.Tensor,
    new_kv: torch.Tensor,
    new_cu_seqlens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Varlen decoding for inference, with separate prefill cache and decode cache

    Args:
        past_kv (`[2, total_nnz, num_kv_heads, head_size]`)
        past_cu_seqlens (`[total_seqs + 1]`):
        new_kv (`[2, total_nnz, num_kv_heads, head_size]`)
        new_cu_seqlens (`[total_seqs + 1]`):
    """

    final_cu_seqlens = past_cu_seqlens + new_cu_seqlens
    final_kv = torch.empty((2, final_cu_seqlens[-1], *past_kv.shape[2:]), device=past_kv.device, dtype=past_kv.dtype)

    B, nH, dH = past_cu_seqlens.shape[0] - 1, past_kv.shape[2], past_kv.shape[3]
    pNNZ, nNNZ = past_kv.shape[1], new_kv.shape[1]
    NEW_CHUNK_SIZE = 1

    grid = (nH, B, 2)
    varlen_kv_manage_fwd[grid](
        past_kv,
        past_cu_seqlens,
        new_kv,
        new_cu_seqlens,
        final_kv,
        pNNZ,
        nNNZ,
        nH,
        dH,
        B,
        NEW_CHUNK_SIZE
    )

    return final_kv, final_cu_seqlens

def naive_concat(
    past_kv: torch.Tensor,
    past_cu_seqlens: torch.Tensor,
    new_kv: torch.Tensor,
    new_cu_seqlens: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    past_seqlens = past_cu_seqlens[1:] - past_cu_seqlens[:-1]
    new_seqlens = new_cu_seqlens[1:] - new_cu_seqlens[:-1]

    splited_past_kv, splited_new_kv = list(map(lambda x, y: torch.split(x, y.cpu().tolist(), dim=1), [past_kv, new_kv], [past_seqlens, new_seqlens]))

    splited_final_kv = list(map(lambda x, y: torch.cat([x, y], dim=1), splited_past_kv, splited_new_kv))
    final_kv = torch.cat(splited_final_kv, dim=1)
    final_cu_seqlens = past_cu_seqlens + new_cu_seqlens
    return final_kv, final_cu_seqlens

def naive_test(
    q: torch.Tensor,
    q_cu_seqlens: torch.Tensor,
    past_kv: torch.Tensor,
    past_cu_seqlens: torch.Tensor,
    new_kv: torch.Tensor,
    new_cu_seqlens: torch.Tensor,
    step: int,
    manage_type: str
):
    for i in range(step):
        if manage_type == "naive":
            final_kv, final_kv_cu_seqlens = naive_concat(past_kv, past_cu_seqlens, new_kv, new_cu_seqlens)
        elif manage_type == "triton":
            final_kv, final_kv_cu_seqlens = varlen_kv_manage(past_kv, past_cu_seqlens, new_kv, new_cu_seqlens)

        res = flash_attn_varlen_kvpacked_func(q, final_kv, q_cu_seqlens, final_kv_cu_seqlens, (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item(), (final_kv_cu_seqlens[1:] - final_kv_cu_seqlens[:-1]).max().item(), causal=True)
    

if __name__ == "__main__":
    seed_everything(17)

    device = "cuda:0"
    dtype = torch.bfloat16

    head_dim = 64
    num_heads = 8
    batch_size = 1

    # for num_heads in [1, 2, 4, 8]:
    #     for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    #         past_seqlen = torch.randint(16, 1024, (batch_size,), device=device, dtype=torch.int64)

    #         past_cu_seqlens = torch.tensor([0] + past_seqlen.cpu().tolist(), dtype=torch.int32, device=device).cumsum(0)
    #         dummy_input = torch.randn((2, past_cu_seqlens[-1].item(), num_heads, head_dim), device=device, dtype=dtype)

    #         dummy_new_input = torch.randn((2, batch_size, num_heads, head_dim), device=device, dtype=dtype)
    #         new_cu_seqlens = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)

    #         naive_kv, naive_cu_seqlens = naive_concat(dummy_input, past_cu_seqlens, dummy_new_input, new_cu_seqlens)
    #         triton_kv, triton_cu_seqlens = varlen_kv_manage(dummy_input, past_cu_seqlens, dummy_new_input, new_cu_seqlens)

    #         assert torch.allclose(naive_kv, triton_kv, atol=1e-5)

    #         naive_fwd = partial(lambda: naive_concat(dummy_input, past_cu_seqlens, dummy_new_input, new_cu_seqlens))
    #         triton_fwd = partial(lambda: varlen_kv_manage(dummy_input, past_cu_seqlens, dummy_new_input, new_cu_seqlens))

    #         naive_bench = do_bench(naive_fwd)
    #         triton_bench = do_bench(triton_fwd)
    #         print(f"Batch Size: {batch_size}, Num Heads: {num_heads}, Head Dim: {head_dim} --> Naive: {naive_bench:.4f}, Triton: {triton_bench:.4f}, Speedup: {naive_bench / triton_bench:.4f}")

    past_seqlen = torch.randint(16, 1024, (batch_size,), device=device, dtype=torch.int64)

    past_cu_seqlens = torch.tensor([0] + past_seqlen.cpu().tolist(), dtype=torch.int32, device=device).cumsum(0)
    dummy_input = torch.randn((2, past_cu_seqlens[-1].item(), num_heads, head_dim), device=device, dtype=dtype)

    dummy_new_input = torch.randn((2, batch_size, num_heads, head_dim), device=device, dtype=dtype)
    new_cu_seqlens = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)

    q = torch.randn((batch_size, num_heads, head_dim), dtype=dtype, device=device)
    q_cu_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    naive_fwd = partial(lambda: naive_test(q, q_cu_seqlens, dummy_input, past_cu_seqlens, dummy_new_input, new_cu_seqlens, 1024, "vanilla"))
    triton_fwd = partial(lambda: naive_test(q, q_cu_seqlens, dummy_input, past_cu_seqlens, dummy_new_input, new_cu_seqlens, 1024, "triton"))

    naive_bench = do_bench(naive_fwd)
    triton_bench = do_bench(triton_fwd)
    print(f"Batch Size: {batch_size}, Num Heads: {num_heads}, Head Dim: {head_dim} --> Naive: {naive_bench:.4f}, Triton: {triton_bench:.4f}, Speedup: {naive_bench / triton_bench:.4f}")