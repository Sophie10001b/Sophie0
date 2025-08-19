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

from lightning import seed_everything
from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache
)
from flash_decoding_triton import flash_decoding_varlen

def decode_attention_varlen_fa(
    Q: torch.Tensor,
    prefill_cache: torch.Tensor,
    decode_cache: torch.Tensor,
    prefill_cu_seqlens: torch.Tensor,
    step: int
) -> torch.Tensor:
    split_lens = (prefill_cu_seqlens[1:] - prefill_cu_seqlens[:-1]).cpu().tolist()
    prefill_chunked = list(torch.split(prefill_cache, split_lens, 1))
    prefill_chunked = list(map(lambda x, y: torch.cat([x, y.squeeze(1)], dim=1), prefill_chunked, decode_cache[:, :, :step].split(1, dim=1)))
    prefill_chunked = torch.cat(prefill_chunked, dim=1)

    new_cu_seqlens = prefill_cu_seqlens + torch.arange(Q.size(0)+1, device=device, dtype=torch.int32) * step
    new_cu_seqlens = new_cu_seqlens.to(torch.int32)
    max_seqlen = torch.max(new_cu_seqlens[1:] - new_cu_seqlens[:-1]).item()

    q_cu_seqlens = torch.arange(0, Q.size(0)+1, dtype=torch.int32, device=Q.device)
    q_max_seqlens = torch.max(q_cu_seqlens[1:] - q_cu_seqlens[:-1]).item()

    return flash_attn_varlen_func(Q, prefill_chunked[0], prefill_chunked[1], q_cu_seqlens, new_cu_seqlens, q_max_seqlens, max_seqlen)

def decode_attention_padding(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor
):
    return flash_attn_with_kvcache(Q, K, V)

def decode_attention_vanilla(
    Q: torch.Tensor,
    prefill_cache: torch.Tensor,
    decode_cache: torch.Tensor,
    prefill_cu_seqlens: torch.Tensor,
    step: int
):  
    q_scale = 1 / math.sqrt(Q.size(-1))
    attn_fun = lambda q, k, v: ((q * q_scale) @ k.transpose(-1, -2)).softmax(dim=-1) @ v
    head_map = Q.size(-2) // prefill_cache.size(-2)
    Q_new = Q.clone().detach()

    for batch_id in range(Q.size(0)):
        tQ = Q[batch_id].unsqueeze(0).transpose(0, 1)
        nnz, next_nnz = prefill_cu_seqlens[batch_id].item(), prefill_cu_seqlens[batch_id + 1].item()

        tK = torch.cat([prefill_cache[0, nnz:next_nnz], decode_cache[0, batch_id, :step]], dim=0)
        tV = torch.cat([prefill_cache[1, nnz:next_nnz], decode_cache[1, batch_id, :step]], dim=0)

        tK = rearrange(tK, "n h d -> h n d").repeat_interleave(head_map, dim=0)
        tV = rearrange(tV, "n h d -> h n d").repeat_interleave(head_map, dim=0)

        res = attn_fun(tQ, tK, tV)
        Q_new[batch_id] = res.transpose(0, 1).squeeze(0)
    
    return Q_new

def generate_bench_data(
    batch_size: int=1,
    prefill_maxlen: int=1024,
    prefill_sparse: float=0.0,
    decode_maxlen: int=1024,
    num_q_head: int=8,
    num_kv_head: int=2,
    head_dim: int=64,
    device: str="cuda:0",
    dtype: torch.dtype=torch.bfloat16
):
    total_nnz = int(batch_size * prefill_maxlen * (1 - prefill_sparse))
    if prefill_sparse > 0:
        assert batch_size > 1
        assert prefill_maxlen < batch_size * prefill_maxlen * (1 - prefill_sparse)

        seqlens = torch.randint(low=1, high=prefill_maxlen, size=(batch_size,), dtype=torch.int64).tolist()
        seqlens[0] = prefill_maxlen
        current_nnz = sum(seqlens[1:])

        while current_nnz < total_nnz - prefill_maxlen:
            row = random.randint(1, batch_size-1)
            seqlens[row] += 1
            current_nnz += 1
        
        while current_nnz > total_nnz - prefill_maxlen:
            row = random.randint(1, batch_size-1)
            seqlens[row] -= 1
            current_nnz -= 1
        
        seqlens = torch.tensor(seqlens, dtype=torch.int64, device=device)

    else:
        seqlens = torch.full((batch_size,), prefill_maxlen, dtype=torch.int64, device=device)
    
    Q = torch.randn((batch_size, num_q_head, head_dim), dtype=dtype, device=device)
    K = torch.randn((batch_size, prefill_maxlen + decode_maxlen, num_kv_head, head_dim), dtype=dtype, device=device)
    V = torch.randn((batch_size, prefill_maxlen + decode_maxlen, num_kv_head, head_dim), dtype=dtype, device=device)

    KV_prefill = torch.randn((2, total_nnz, num_kv_head, head_dim), dtype=dtype, device=device)
    KV_decode = torch.randn((2, batch_size, decode_maxlen, num_kv_head, head_dim), dtype=dtype, device=device)
    cu_seqlens = torch.tensor([0] + seqlens.cpu().tolist(), dtype=torch.int32, device=device).cumsum(0)

    return dict(
        Q=Q,
        K=K,
        V=V,
        KV_prefill=KV_prefill,
        KV_decode=KV_decode,
        cu_seqlens=cu_seqlens,
        seqlens=seqlens
    )


if __name__ == "__main__":
    seed_everything(17)
    device = "cuda:0"
    dtype = torch.bfloat16
    base_path = os.path.dirname(os.path.abspath(__file__))

    nqh = 8
    nkh = 2
    dh = 64

    bsz = 16
    seqlens = [random.choice([512, 1024, 2048]) for _ in range(bsz)]
    

    total_nnz = sum(seqlens)
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32, device=device).cumsum(0)
    max_seqlen = max(seqlens)
    max_steps = 2048

    Q = torch.randn((bsz, nqh, dh), dtype=dtype, device=device)
    Q_fa = Q.clone().detach()
    Q_vanilla = Q.clone().detach()
    Q_triton = Q.clone().detach()

    prefill_cache = torch.randn((2, total_nnz, nkh, dh), dtype=dtype, device=device)
    decode_cache = torch.randn((2, bsz, max_steps, nkh, dh), dtype=dtype, device=device)
    step = 1024

    # consist check
    fa_res = decode_attention_varlen_fa(Q_fa, prefill_cache, decode_cache, cu_seqlens, step)
    tl_res = flash_decoding_varlen(Q_triton, prefill_cache, decode_cache, cu_seqlens, step)
    vanilla_res = decode_attention_vanilla(Q_vanilla, prefill_cache, decode_cache, cu_seqlens, step)

    print(f"tl_vs_vanilla: {(tl_res - vanilla_res).abs().mean()}")
    print(f"tl_vs_fa: {(tl_res - fa_res).abs().mean()}")
    pass

    # benchmark
    from functools import partial

    # batch size
    import matplotlib.pyplot as plt

    BSZ = [1, 2, 4, 8, 16, 32, 64]
    fa_res = []
    tl_res = []
    va_res = []
    for B in BSZ:
        dummy: dict = generate_bench_data(
            batch_size=B,
            prefill_maxlen=1024,
            prefill_sparse=0,
            decode_maxlen=1024,
            num_q_head=8,
            num_kv_head=2,
            head_dim=64,
            device=device,
            dtype=dtype
        )

        fa_fwd = partial(lambda : decode_attention_padding(dummy['Q'].unsqueeze(1), dummy['K'], dummy['V']))
        tl_fwd = partial(lambda : flash_decoding_varlen(dummy['Q'], dummy['KV_prefill'], dummy['KV_decode'], dummy['cu_seqlens'], 1024))
        va_fwd = partial(lambda : decode_attention_varlen_fa(dummy['Q'], dummy['KV_prefill'], dummy['KV_decode'], dummy['cu_seqlens'], 1024))
        fa_bench = triton.testing.do_bench(fa_fwd)
        tl_bench = triton.testing.do_bench(tl_fwd)
        va_bench = triton.testing.do_bench(va_fwd)
        print(f"B={B}, FA: {fa_bench:.2f}, TL: {tl_bench:.2f}, VA: {va_bench:.2f}")
        fa_res.append(fa_bench)
        tl_res.append(tl_bench)
        va_res.append(va_bench)

    x_row = list(range(len(BSZ)))
    plt.plot(x_row, fa_res, label="flash_attn_padding", linewidth=2, color='r')
    plt.plot(x_row, tl_res, label="triton_flatten", linewidth=2, color='b')
    plt.plot(x_row, va_res, label="flash_attn_varlen", linewidth=2, color='g')

    plt.legend(loc='upper right')
    plt.xticks(x_row, [str(_) for _ in BSZ])
    plt.plot()
    plt.savefig(os.path.join(base_path, "benchmark_bsz.pdf"), dpi=600, format="pdf")
    plt.clf()
    
    # seqlen
    SEQ = [512, 1024, 2048, 4096, 8192]
    fa_res = []
    tl_res = []
    va_res = []
    for S in SEQ:
        dummy: dict = generate_bench_data(
            batch_size=8,
            prefill_maxlen=S,
            prefill_sparse=0,
            decode_maxlen=1024,
            num_q_head=8,
            num_kv_head=2,
            head_dim=64,
            device=device,
            dtype=dtype
        )

        fa_fwd = partial(lambda : decode_attention_padding(dummy['Q'].unsqueeze(1), dummy['K'], dummy['V']))
        tl_fwd = partial(lambda : flash_decoding_varlen(dummy['Q'], dummy['KV_prefill'], dummy['KV_decode'], dummy['cu_seqlens'], 1024))
        va_fwd = partial(lambda : decode_attention_varlen_fa(dummy['Q'], dummy['KV_prefill'], dummy['KV_decode'], dummy['cu_seqlens'], 1024))
        fa_bench = triton.testing.do_bench(fa_fwd)
        tl_bench = triton.testing.do_bench(tl_fwd)
        va_bench = triton.testing.do_bench(va_fwd)
        print(f"S={S}, FA: {fa_bench:.2f}, TL: {tl_bench:.2f}, VA: {va_bench:.2f}")
        fa_res.append(fa_bench)
        tl_res.append(tl_bench)
        va_res.append(va_bench)

    x_row = list(range(len(SEQ)))
    plt.plot(x_row, fa_res, label="flash_attn_padding", linewidth=2, color='r')
    plt.plot(x_row, tl_res, label="triton_flatten", linewidth=2, color='b')
    plt.plot(x_row, va_res, label="flash_attn_varlen", linewidth=2, color='g')

    plt.legend(loc='upper right')
    plt.xticks(x_row, [str(_) for _ in SEQ])
    plt.plot()
    plt.savefig(os.path.join(base_path, "benchmark_seqlen.pdf"), dpi=600, format="pdf")
    plt.clf()


    # sparse prefill
    SP = [0.1, 0.2, 0.3, 0.4, 0.5]
    fa_res = []
    tl_res = []
    va_res = []
    for S in SP:
        dummy: dict = generate_bench_data(
            batch_size=8,
            prefill_maxlen=4096,
            prefill_sparse=S,
            decode_maxlen=1024,
            num_q_head=8,
            num_kv_head=2,
            head_dim=64,
            device=device,
            dtype=dtype
        )

        fa_fwd = partial(lambda : decode_attention_padding(dummy['Q'].unsqueeze(1), dummy['K'], dummy['V']))
        tl_fwd = partial(lambda : flash_decoding_varlen(dummy['Q'], dummy['KV_prefill'], dummy['KV_decode'], dummy['cu_seqlens'], 1024))
        va_fwd = partial(lambda : decode_attention_varlen_fa(dummy['Q'], dummy['KV_prefill'], dummy['KV_decode'], dummy['cu_seqlens'], 1024))
        fa_bench = triton.testing.do_bench(fa_fwd)
        tl_bench = triton.testing.do_bench(tl_fwd)
        va_bench = triton.testing.do_bench(va_fwd)
        print(f"SP={S}, FA: {fa_bench:.2f}, TL: {tl_bench:.2f}, VA: {va_bench:.2f}")
        fa_res.append(fa_bench)
        tl_res.append(tl_bench)
        va_res.append(va_bench)

    x_row = list(range(len(SP)))
    plt.plot(x_row, fa_res, label="flash_attn_padding", linewidth=2, color='r')
    plt.plot(x_row, tl_res, label="triton_flatten", linewidth=2, color='b')
    plt.plot(x_row, va_res, label="flash_attn_varlen", linewidth=2, color='g')

    plt.legend(loc='upper right')
    plt.xticks(x_row, [str(_) for _ in SP])
    plt.plot()
    plt.savefig(os.path.join(base_path, "benchmark_sparse_4096.pdf"), dpi=600, format="pdf")
    plt.clf()