#!/usr/bin/env python3
"""
Benchmark: 大模型常用 Flash Attention shapes
LLaMA / GPT / Qwen / Mistral 典型配置
"""
import math, torch, sys

WARMUP = 10
ITERS = 100

def bench(fn, *args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(torch.tensor(times).median())

from extension_cpp.ops import flash_attn
try:
    from flash_attn import flash_attn_func as tri_dao_fa
    HAS_TRI_DAO = True
except ImportError:
    HAS_TRI_DAO = False

print(f"Device : {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}  |  CUDA {torch.version.cuda}")
print(f"Tri Dao: {'YES' if HAS_TRI_DAO else 'NO'}")
print(f"Warmup={WARMUP}  Bench={ITERS} iters\n")

# ══════════════════════════════════════════════════════════════════════════════
# 大模型典型 Shape: (name, B, H_qo, H_kv, N_qo, D)
# ══════════════════════════════════════════════════════════════════════════════

SHAPES = [
    # ── Prefill (训练前向 / 推理首token) ──
    ("LLaMA-7B    pref 4K   b=4   ", 4,  32, 32,  4096, 128),
    ("LLaMA-7B    pref 8K   b=2   ", 2,  32, 32,  8192, 128),
    ("LLaMA-13B   pref 4K   b=2   ", 2,  40, 40,  4096, 128),
    ("LLaMA-70B   pref 4K   b=1   ", 1,  64, 64,  4096, 128),
    ("Qwen-14B    pref 8K   b=1   ", 1,  40, 40,  8192, 128),
    # ── Batched Decode (多请求并发,N_QO=64) ──
    ("LLaMA-7B    dec KV=8K  b=1   ", 1,  32, 32,    64, 128),  # N_KV=8192
    ("LLaMA-7B    dec KV=32K b=1   ", 1,  32, 32,    64, 128),  # N_KV=32768
    ("LLaMA-70B   dec KV=8K  b=1   ", 1,  64, 64,    64, 128),  # N_KV=8192
    # ── Training microbatch ──
    ("LLaMA-7B    train 2K  b=8   ", 8,  32, 32,  2048, 128),
    ("LLaMA-7B    train 8K  b=4   ", 4,  32, 32,  8192, 128),
    # ── Long context prefill ──
    ("LLaMA-7B    pref 32K  b=1   ", 1,  32, 32, 32768, 128),
]

# Determine N_KV for decode shapes
def get_nkv(name, n_qo):
    if "KV=8K" in name: return 8192
    if "KV=32K" in name: return 32768
    return n_qo

col_hdr = f"  {'Shape':<32s} │ {'B':>2s} {'Hq':>3s} {'Hkv':>3s} │ {'Nq':>6s} {'Nkv':>7s} │ {'custom':>7s}"
if HAS_TRI_DAO: col_hdr += " │ {'TriDao':>7s}"
col_hdr += " │ {'SDPA':>7s} │ {'vs_SDPA':>8s}"
print(col_hdr)
print("-" * 120)

vs_sdpa = []

for name, B, Hq, Hkv, N_qo, D in SHAPES:
    N_kv = get_nkv(name, N_qo)

    q = torch.randn(B, Hq, N_qo, D, dtype=torch.float16, device="cuda").contiguous()
    k = torch.randn(B, Hkv, N_kv, D, dtype=torch.float16, device="cuda").contiguous()
    v = torch.randn(B, Hkv, N_kv, D, dtype=torch.float16, device="cuda").contiguous()

    try:
        _ = flash_attn(q, k, v)  # verify
    except Exception as e:
        print(f"  {name:<32s} │ SKIP: {str(e)[:50]}")
        continue

    torch.cuda.synchronize()
    ms_custom = bench(lambda: flash_attn(q, k, v))

    ms_tri = 0
    if HAS_TRI_DAO:
        ms_tri = bench(lambda: tri_dao_fa(q.transpose(1,2), k.transpose(1,2),
                                            v.transpose(1,2), causal=False).transpose(1,2))

    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                                          torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
        for _ in range(WARMUP):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        ms_sdpa = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))

    sp = ms_sdpa / ms_custom
    vs_sdpa.append(sp)
    m = ">" if sp > 1.02 else ("<" if sp < 0.98 else "=")

    tri_s = f" │ {ms_tri:>5.2f}ms" if HAS_TRI_DAO else ""
    print(f"  {name:<32s} │ {B:>2d} {Hq:>3d} {Hkv:>3d} │ {N_qo:>6d} {N_kv:>7d} │ {ms_custom:>5.2f}ms{tri_s} │ {ms_sdpa:>5.2f}ms │ {m}{sp:>6.2f}x")

print("-" * 120)
geo = math.exp(sum(math.log(x) for x in vs_sdpa) / len(vs_sdpa))
print(f"  Geomean vs torch SDPA: {geo:.3f}x  (>1 = custom faster)")
print()
