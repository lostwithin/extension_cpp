#!/usr/bin/env python3
"""
Qwen3-0.6B Flash Attention benchmark — 三方对比:
  1. custom : 本项目的 CuTe flash attention kernel
  2. SDPA   : torch F.scaled_dot_product_attention (FLASH 后端 = FlashAttention-2,已 profiler 确认)
  3. naive  : 非-flash 基础版,显式物化 softmax(QKᵀ)·V (无分块 / 无 online-softmax,O(N²) 显存)

Qwen3-0.6B: n_heads=16, n_kv_heads=8 (GQA 2:1), head_dim=128。
本 kernel 为 MHA,基准里把 KV 头 repeat 到 16 (三方同样处理,公平对比)。
  · prefill      : N_QO == N_KV = seqlen
  · decode-chunk : N_QO=64 (推测解码/chunk), N_KV = KV-cache 长度 (真·N_QO=1 需 flash-decoding kernel)
"""
import math, torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from extension_cpp.ops import flash_attn

WARMUP, ITERS = 15, 100
HQ, HKV, D = 16, 8, 128          # Qwen3-0.6B

def bench(fn):
    for _ in range(WARMUP): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(ITERS):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(torch.tensor(ts).median())

def expand_kv(t, rep):   # GQA -> MHA
    return t.repeat_interleave(rep, dim=1)

def sdpa_flash(q, k, v):
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(q, k, v)

def naive_attention(q, k, v):
    # 非-flash 基础版:显式物化完整注意力矩阵
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale     # [B,H,Nq,Nkv]  <-- O(N²) 物化
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

# (label, B, N_QO, N_KV)
PREFILL = [
    ("prefill  512  b=1", 1,   512,   512),
    ("prefill 1024  b=1", 1,  1024,  1024),
    ("prefill 2048  b=1", 1,  2048,  2048),
    ("prefill 4096  b=1", 1,  4096,  4096),
    ("prefill 2048  b=4", 4,  2048,  2048),
    ("prefill 4096  b=2", 2,  4096,  4096),
]
DECODE = [
    ("dec-chunk64 KV=1024 ", 1, 64, 1024),
    ("dec-chunk64 KV=2048 ", 1, 64, 2048),
    ("dec-chunk64 KV=4096 ", 1, 64, 4096),
    ("dec-chunk64 KV=8192 ", 1, 64, 8192),
]

def run_group(title, shapes):
    print(f"\n{'='*100}\n  {title}\n{'='*100}")
    print(f"  {'shape':<22s} │ {'B':>2s} {'Nq':>5s} {'Nkv':>5s} │ "
          f"{'custom':>8s} │ {'SDPA':>8s} │ {'naive':>9s} │ {'vs SDPA':>8s} │ {'vs naive':>8s}")
    print("-"*100)
    sp_sdpa, sp_naive = [], []
    for label, B, Nq, Nkv in shapes:
        q  = torch.randn(B, HQ,  Nq,  D, dtype=torch.float16, device="cuda")
        k  = expand_kv(torch.randn(B, HKV, Nkv, D, dtype=torch.float16, device="cuda"), HQ // HKV).contiguous()
        v  = expand_kv(torch.randn(B, HKV, Nkv, D, dtype=torch.float16, device="cuda"), HQ // HKV).contiguous()

        try:
            out = flash_attn(q, k, v)
        except Exception as ex:
            print(f"  {label:<22s} │ {B:>2d} {Nq:>5d} {Nkv:>5d} │ SKIP: {str(ex)[:40]}")
            continue

        # 正确性 (custom vs SDPA)
        ref = sdpa_flash(q, k, v)
        cos = F.cosine_similarity(out.float().reshape(-1), ref.float().reshape(-1), dim=0).item()
        chk = "OK" if cos > 0.999 else f"!! cos={cos:.5f}"

        ms_c = bench(lambda: flash_attn(q, k, v))
        ms_s = bench(lambda: sdpa_flash(q, k, v))

        # naive (可能 OOM)
        try:
            _ = naive_attention(q, k, v)
            torch.cuda.synchronize()
            ms_n = bench(lambda: naive_attention(q, k, v))
            naive_str = f"{ms_n:>7.3f}ms"
            spn = ms_n / ms_c; sp_naive.append(spn); spn_str = f"{spn:>6.1f}x"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ms_n = None; naive_str = f"{'OOM':>9s}"; spn_str = f"{'OOM':>8s}"

        sps = ms_s / ms_c; sp_sdpa.append(sps)
        m = ">" if sps > 1.02 else ("<" if sps < 0.98 else "=")
        print(f"  {label:<22s} │ {B:>2d} {Nq:>5d} {Nkv:>5d} │ "
              f"{ms_c:>6.3f}ms │ {ms_s:>6.3f}ms │ {naive_str} │ {m}{sps:>6.2f}x │ {spn_str}  [{chk}]")
        del q, k, v, out, ref
        torch.cuda.empty_cache()
    print("-"*100)
    if sp_sdpa:
        g = math.exp(sum(math.log(x) for x in sp_sdpa)/len(sp_sdpa))
        line = f"  Geomean vs SDPA: {g:.3f}x  (>1 = custom faster)"
        if sp_naive:
            gn = math.exp(sum(math.log(x) for x in sp_naive)/len(sp_naive))
            line += f"   |   vs naive: {gn:.1f}x faster"
        print(line)

if __name__ == "__main__":
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Config : Qwen3-0.6B  Hq={HQ} Hkv={HKV} (GQA→MHA expand) D={D}")
    print(f"Baselines: SDPA=FLASH backend (FA2) | naive=显式物化 softmax(QKᵀ)V (非-flash)")
    print(f"Warmup={WARMUP} Bench={ITERS}")
    run_group("PREFILL  (N_QO = N_KV)", PREFILL)
    run_group("DECODE-CHUNK  (N_QO=64;  真·N_QO=1 需 flash-decoding kernel)", DECODE)
