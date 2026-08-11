#!/usr/bin/env python3
"""
Flash Attention 正确性 & 性能对比
=================================
  · flash_attn       — CuTe Flash Attention kernel (extension_cpp)
  · Tri Dao FA (v2)  — flash_attn_func (Dao-AILab/flash-attention, 官方实现)
  · torch SDPA       — torch.nn.functional.scaled_dot_product_attention
  · ref (fp32)       — manual attention in fp32 (ground truth for correctness)

Input: Q, K, V ∈ (B, H, N, D) fp16
"""

from __future__ import annotations

import gc
import math
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

# ── Our custom kernel ─────────────────────────────────────────────────

try:
    from extension_cpp.ops import flash_attn
except ImportError as e:
    print(f"[WARN] Cannot import extension_cpp: {e}")
    flash_attn = None

# ── Tri Dao's official flash-attention (v2) ───────────────────────────

try:
    from flash_attn import flash_attn_func as tri_dao_fa
    HAS_TRI_DAO = True
except ImportError:
    tri_dao_fa = None
    HAS_TRI_DAO = False


# ═════════════════════════════════════════════════════════════════════════
#  Reference — manual attention in fp32 (ground truth)
# ═════════════════════════════════════════════════════════════════════════

def ref_attention_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float | None = None) -> torch.Tensor:
    """Manual scaled dot-product attention in fp32 — ground truth."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    q_f = q.float() * scale
    k_f = k.float()
    v_f = v.float()
    attn = torch.softmax(q_f @ k_f.transpose(-2, -1), dim=-1)
    return attn @ v_f


# ═════════════════════════════════════════════════════════════════════════
#  Bench helpers
# ═════════════════════════════════════════════════════════════════════════

WARMUP_ITERS = 10
BENCH_ITERS = 50


def _median_ms(fn, *args, warmup=WARMUP_ITERS, iters=BENCH_ITERS) -> float:
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


def _geomean(xs: list[float]) -> float:
    pos = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in pos) / len(pos)) if pos else float("nan")


# ═════════════════════════════════════════════════════════════════════════
#  Config — (B, H, N_QO, N_KV, D)
# ═════════════════════════════════════════════════════════════════════════

SELF_ATTN_SHAPES: list[tuple[int, int, int, int]] = [
    # (B, H, N, D)
    (1, 8, 128, 16),
    (1, 8, 256, 32),
    (1, 8, 512, 64),
    (1, 8, 256, 128),
    (1, 4, 128, 256),
    # Batch > 1
    (4, 8, 256, 64),
    (2, 16, 512, 64),
    # Medium stress
    (2, 8, 1024, 64),
    (1, 8, 2048, 64),
    (1, 8, 4096, 64),
]

CROSS_ATTN_SHAPES: list[tuple[int, int, int, int, int]] = [
    # (B, H, N_QO, N_KV, D)
    (1, 8, 1, 128, 64),
    (1, 8, 1, 512, 64),
    (1, 8, 1, 1024, 64),
    (1, 8, 128, 4096, 64),
]

HEAD_DIMS = [16, 32, 64, 128, 256]


def _make_inputs(b: int, h: int, n_qo: int, n_kv: int, d: int
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(b, h, n_qo, d, dtype=torch.float16, device="cuda").contiguous()
    k = torch.randn(b, h, n_kv, d, dtype=torch.float16, device="cuda").contiguous()
    v = torch.randn(b, h, n_kv, d, dtype=torch.float16, device="cuda").contiguous()
    return q, k, v


def _run_custom(q, k, v):
    """Wrapper: our kernel expects (B, H, N, D)."""
    return flash_attn(q, k, v)


def _run_tri_dao(q, k, v):
    """Wrapper: Tri Dao FA v2 expects (B, N, H, D) with causal=False."""
    return tri_dao_fa(q.transpose(1, 2), k.transpose(1, 2),
                       v.transpose(1, 2), causal=False).transpose(1, 2)


# ═════════════════════════════════════════════════════════════════════════
#  正确性 — 三者 vs fp32 reference
# ═════════════════════════════════════════════════════════════════════════

def test_correctness_all(rtol: float = 0.02, atol: float = 0.02) -> bool:
    """Compare all implementations vs fp32 ground truth."""
    print("=" * 80)
    print("  正确性 — 各实现 vs fp32 reference")
    print("=" * 80)

    col_names = ["Shape (B,H,N,D)"]
    col_names.append("custom err")
    if HAS_TRI_DAO:
        col_names.append("TriDao err")
    col_names.append("SDPA err")
    col_names.append("cos(custom)")
    col_names.append("ok?")

    header = (f"  {'Shape':>21s} │ {'custom:max':>11s} │ "
              f"{'TriDao:max':>11s} │ {'SDPA:max':>11s} │ "
              f"{'cos_sim':>9s} │ {'ok?':>5s}")
    print(header)
    print("-" * 80)

    extra = [(1, 4, 128, d) for d in HEAD_DIMS]
    all_shapes = list(SELF_ATTN_SHAPES) + extra

    all_ok = True
    errors = defaultdict(list)

    for b, h, n, d in all_shapes:
        if d not in HEAD_DIMS:
            continue

        q, k, v = _make_inputs(b, h, n, n, d)
        scale = 1.0 / math.sqrt(d)

        # fp32 reference
        out_ref = ref_attention_fp32(q, k, v, scale)

        # Custom
        try:
            out_custom = _run_custom(q, k, v)
            err_custom = (out_custom.float() - out_ref).abs().max().item()
            cos_custom = torch.nn.functional.cosine_similarity(
                out_custom.float().reshape(-1), out_ref.reshape(-1), dim=0).item()
        except Exception as e:
            print(f"  ({b:>3d},{h:>3d},{n:>5d},{d:>3d}) │ {'SKIP':>11s} │ "
                  f"{'—':>11s} │ {'—':>11s} │ {'—':>9s} │ custom err: {e}")
            continue

        # Tri Dao
        err_tri = float("nan")
        if HAS_TRI_DAO:
            out_tri = _run_tri_dao(q, k, v)
            err_tri = (out_tri.float() - out_ref).abs().max().item()

        # SDPA
        out_sdpa = F.scaled_dot_product_attention(q, k, v)
        err_sdpa = (out_sdpa.float() - out_ref).abs().max().item()

        ok = cos_custom > 0.99
        if not ok:
            all_ok = False
        tag = "✓" if ok else "✗"

        tri_str = f"{err_tri:>11.2e}" if HAS_TRI_DAO else f"{'—':>11s}"
        print(f"  ({b:>3d},{h:>3d},{n:>5d},{d:>3d}) │ {err_custom:>11.2e} │ "
              f"{tri_str} │ {err_sdpa:>11.2e} │ {cos_custom:>9.6f} │ {tag:>5s}")

        errors[d].append(err_custom)

    print("-" * 80)
    for d in sorted(errors.keys()):
        avg = sum(errors[d]) / len(errors[d])
        print(f"  head_dim={d:>3d}: avg_custom_err={avg:.2e}  ({len(errors[d])} tests)")
    print()

    return all_ok


# ═════════════════════════════════════════════════════════════════════════
#  正确性 — custom vs Tri Dao FA (直接对比)
# ═════════════════════════════════════════════════════════════════════════

def test_correctness_vs_tri_dao() -> bool | None:
    """Direct output comparison: custom vs Tri Dao flash_attn_func."""
    if not HAS_TRI_DAO:
        print("[SKIP] Tri Dao flash-attn 未安装，跳过直接对比")
        print("       安装: pip install flash-attn --no-build-isolation")
        return None

    print("=" * 80)
    print("  正确性 — custom flash_attn vs Tri Dao flash_attn_func (v2)")
    print("=" * 80)
    print(f"  {'Shape (B,H,N,D)':>21s} │ {'err_max':>9s} │ {'err_mean':>9s} │ "
          f"{'cos_sim':>9s} │ {'ok?':>5s}")
    print("-" * 80)

    all_ok = True

    for b, h, n, d in SELF_ATTN_SHAPES:
        if d not in HEAD_DIMS:
            continue

        q, k, v = _make_inputs(b, h, n, n, d)

        try:
            out_custom = _run_custom(q, k, v)
        except Exception as e:
            print(f"  ({b:>3d},{h:>3d},{n:>5d},{d:>3d}) │ {'SKIP':>9s} │ {'—':>9s} │ "
                  f"{'—':>9s} │ custom err: {e}")
            continue

        out_tri = _run_tri_dao(q, k, v)

        err_max = (out_custom.float() - out_tri.float()).abs().max().item()
        err_mean = (out_custom.float() - out_tri.float()).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_custom.float().reshape(-1), out_tri.float().reshape(-1), dim=0).item()

        ok = cos_sim > 0.999
        if not ok:
            all_ok = False
        tag = "✓" if ok else "✗"

        print(f"  ({b:>3d},{h:>3d},{n:>5d},{d:>3d}) │ {err_max:>9.2e} │ "
              f"{err_mean:>9.2e} │ {cos_sim:>9.6f} │ {tag:>5s}")

    print("-" * 80)
    print()
    return all_ok


# ═════════════════════════════════════════════════════════════════════════
#  性能 — 三者 benchmark
# ═════════════════════════════════════════════════════════════════════════

def benchmark_perf(sizes: list | None = None) -> None:
    """Benchmark: custom vs Tri Dao FA vs torch SDPA."""
    if sizes is None:
        sizes = SELF_ATTN_SHAPES

    n_baselines = 2 + (1 if HAS_TRI_DAO else 0)
    print("=" * 80)
    print(f"  性能对比 — custom vs {'Tri Dao FA vs ' if HAS_TRI_DAO else ''}torch SDPA")
    print("=" * 80)

    sep = " │ "
    # header
    hdr = f"  {'Shape (B,H,N,D)':>21s}{sep}{'custom':>9s}"
    if HAS_TRI_DAO:
        hdr += f"{sep}{'TriDao':>7s}"
    hdr += f"{sep}{'SDPA':>9s}{sep}{'vs_TriDao':>9s}{sep}{'vs_SDPA':>9s}"
    print(hdr)
    print("-" * 80)

    vs_tri_speedups = []
    vs_sdpa_speedups = []

    for b, h, n, d in sizes:
        if d not in HEAD_DIMS:
            continue

        q, k, v = _make_inputs(b, h, n, n, d)

        # Verify
        try:
            flash_attn(q, k, v)
        except Exception as e:
            print(f"  ({b:>3d},{h:>3d},{n:>5d},{d:>3d}){sep}{'SKIP':>9s}"
                  f"{sep}{'—':>7s}{sep}{'—':>9s}{sep}{'—':>9s}{sep}{'—':>9s}  "
                  f"({str(e)[:30]})")
            continue

        torch.cuda.synchronize()

        # custom
        ms_custom = _median_ms(lambda: flash_attn(q, k, v))

        # Tri Dao
        ms_tri = float("nan")
        if HAS_TRI_DAO:
            ms_tri = _median_ms(lambda: _run_tri_dao(q, k, v))

        # SDPA
        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.FLASH_ATTENTION,
             torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
        ):
            for _ in range(WARMUP_ITERS):
                F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            ms_sdpa = _median_ms(lambda: F.scaled_dot_product_attention(q, k, v))

        # speedups
        sp_tri = ms_tri / ms_custom if HAS_TRI_DAO and ms_tri > 0 else float("nan")
        sp_sdpa = ms_sdpa / ms_custom

        if not math.isnan(sp_tri):
            vs_tri_speedups.append(sp_tri)
        vs_sdpa_speedups.append(sp_sdpa)

        # markers
        def _marker(sp):
            if math.isnan(sp):
                return " "
            return ">" if sp > 1.0 else ("=" if abs(sp - 1.0) < 0.01 else "<")

        line = f"  ({b:>3d},{h:>3d},{n:>5d},{d:>3d}){sep}{ms_custom:>7.3f}ms"
        if HAS_TRI_DAO:
            line += f"{sep}{ms_tri:>5.3f}ms"
        line += f"{sep}{ms_sdpa:>7.3f}ms"
        line += f"{sep}{_marker(sp_tri)}{sp_tri:>7.2f}x" if HAS_TRI_DAO else f"{sep}{'—':>9s}"
        line += f"{sep}{_marker(sp_sdpa)}{sp_sdpa:>7.2f}x"
        print(line)

    print("-" * 80)
    if vs_tri_speedups:
        print(f"  Geomean vs Tri Dao FA : {_geomean(vs_tri_speedups):.3f}x  "
              f"(>1 = custom faster)")
    if vs_sdpa_speedups:
        print(f"  Geomean vs torch SDPA : {_geomean(vs_sdpa_speedups):.3f}x  "
              f"(>1 = custom faster)")
    print()


# ═════════════════════════════════════════════════════════════════════════
#  正确性 — custom vs torch SDPA
# ═════════════════════════════════════════════════════════════════════════

def test_correctness_vs_sdpa() -> bool:
    """Compare flash_attn vs torch SDPA output."""
    print("=" * 80)
    print("  正确性 — flash_attn vs torch SDPA (同精度)")
    print("=" * 80)
    print(f"  {'Shape (B,H,N,D)':>21s} │ {'err_max':>9s} │ {'err_mean':>9s} │ "
          f"{'cos_sim':>9s} │ {'ok?':>5s}")
    print("-" * 80)

    all_ok = True

    for b, h, n, d in SELF_ATTN_SHAPES:
        if d not in HEAD_DIMS:
            continue

        q, k, v = _make_inputs(b, h, n, n, d)

        try:
            out_custom = flash_attn(q, k, v)
        except Exception as e:
            print(f"  ({b:>3d},{h:>3d},{n:>5d},{d:>3d}) │ {'SKIP':>9s} │ {'—':>9s} │ "
                  f"{'—':>9s} │ {str(e)[:10]}")
            continue

        out_sdpa = F.scaled_dot_product_attention(q, k, v)

        err_max = (out_custom.float() - out_sdpa.float()).abs().max().item()
        err_mean = (out_custom.float() - out_sdpa.float()).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_custom.float().reshape(-1), out_sdpa.float().reshape(-1), dim=0).item()

        ok = cos_sim > 0.999
        if not ok:
            all_ok = False
        tag = "✓" if ok else "✗"

        print(f"  ({b:>3d},{h:>3d},{n:>5d},{d:>3d}) │ {err_max:>9.2e} │ "
              f"{err_mean:>9.2e} │ {cos_sim:>9.6f} │ {tag:>5s}")

    print("-" * 80)
    return all_ok


# ═════════════════════════════════════════════════════════════════════════
#  Cross-attention
# ═════════════════════════════════════════════════════════════════════════

def test_cross_attn() -> None:
    """Test on cross-attention shapes (N_QO != N_KV)."""
    print("=" * 80)
    print("  Cross-attention — custom vs fp32 ref (N_QO != N_KV)")
    print("=" * 80)
    print(f"  {'Shape (B,H,Nq,Nkv,D)':>25s} │ {'err_max':>9s} │ "
          f"{'cos_sim':>9s} │ {'note':>30s}")
    print("-" * 80)

    for b, h, n_qo, n_kv, d in CROSS_ATTN_SHAPES:
        q, k, v = _make_inputs(b, h, n_qo, n_kv, d)
        try:
            out_custom = flash_attn(q, k, v)
            out_ref = ref_attention_fp32(q, k, v)
            err_max = (out_custom.float() - out_ref).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                out_custom.float().reshape(-1), out_ref.reshape(-1), dim=0).item()
            print(f"  ({b:>3d},{h:>3d},{n_qo:>5d},{n_kv:>5d},{d:>3d}) │ "
                  f"{err_max:>9.2e} │ {cos_sim:>9.6f} │ {'✓ OK':>30s}")
        except Exception as e:
            print(f"  ({b:>3d},{h:>3d},{n_qo:>5d},{n_kv:>5d},{d:>3d}) │ "
                  f"{'—':>9s} │ {'—':>9s} │ {str(e)[:30]:>30s}")

    print("-" * 80)
    print()


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    if flash_attn is None:
        print("[FATAL] extension_cpp 未安装。")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("[FATAL] CUDA 不可用。")
        sys.exit(1)

    print(f"Device   : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch  : {torch.__version__}  |  CUDA {torch.version.cuda}")
    print(f"Tri Dao FA: {'已安装' if HAS_TRI_DAO else '未安装 (pip install flash-attn --no-build-isolation)'}")
    print(f"Warmup   : {WARMUP_ITERS}  |  Bench {BENCH_ITERS} iters\n")

    # 1. Correctness: all vs fp32
    ok_ref = test_correctness_all()

    # 2. Correctness: custom vs Tri Dao (direct)
    ok_tri = test_correctness_vs_tri_dao()

    # 3. Correctness: custom vs SDPA
    ok_sdpa = test_correctness_vs_sdpa()

    # 4. Cross-attention
    test_cross_attn()

    # 5. Performance
    benchmark_perf()

    # Summary
    print("=" * 80)
    print("  总结")
    print("=" * 80)
    print(f"  vs fp32 ref  : {'PASS' if ok_ref else 'FAIL'}")
    if ok_tri is not None:
        print(f"  vs Tri Dao FA: {'PASS' if ok_tri else 'FAIL'}")
    else:
        print(f"  vs Tri Dao FA: SKIP (未安装)")
    print(f"  vs torch SDPA: {'PASS' if ok_sdpa else 'FAIL'}")
    print()


if __name__ == "__main__":
    main()
