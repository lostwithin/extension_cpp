#!/usr/bin/env python3
"""
GEMM 正确性 & 性能对比 — cuBLAS COMPUTE_16F 作为公平 Baseline
==============================================================
双方都用 fp16 累加，比的是 kernel 实现质量。

  · mygemm  — CuTe HGEMM, SM80_16x8x16_F16F16F16F16_TN
  · cuBLAS  — cublasGemmEx + CUBLAS_COMPUTE_16F

TN layout: A(M,K) × B^T(N,K) → C(M,N)
"""

from __future__ import annotations

import ctypes
import gc
import math
import struct
import sys

import torch

try:
    from extension_cpp.ops import mygemm
except ImportError as e:
    print(f"[WARN] Cannot import extension_cpp: {e}")
    mygemm = None


# ═════════════════════════════════════════════════════════════════════════
#  cuBLAS — CUBLAS_COMPUTE_16F (fp16 累加)
# ═════════════════════════════════════════════════════════════════════════

CUBLAS_OP_N, CUBLAS_OP_T = 0, 1
CUDA_R_16F  = 2
CUBLAS_COMPUTE_16F            = 64
CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99
CUBLAS_TENSOR_OP_MATH         = 1

# alpha/beta 必须是 half 类型 — CUBLAS_COMPUTE_16F 的要求
_alpha = ctypes.c_uint16(struct.unpack('<H', struct.pack('<e', 1.0))[0])
_beta  = ctypes.c_uint16(0)

_cublas_lib = _cublas_handle = None


def _init_cublas() -> None:
    global _cublas_lib, _cublas_handle
    if _cublas_lib is not None:
        return

    torch.zeros(1, device="cuda")  # 触发 CUDA 上下文
    _cublas_lib = ctypes.CDLL(
        "/usr/local/cuda/targets/x86_64-linux/lib/libcublas.so"
    )

    _cublas_lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cublas_handle = ctypes.c_void_p()
    _cublas_lib.cublasCreate_v2(ctypes.byref(_cublas_handle))

    _cublas_lib.cublasSetMathMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _cublas_lib.cublasSetMathMode(_cublas_handle, CUBLAS_TENSOR_OP_MATH)

    _cublas_lib.cublasGemmEx.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]

    # Warm up: 首次调用触发内部 JIT
    a = torch.randn(128, 64, dtype=torch.float16, device="cuda")
    b = torch.randn(128, 64, dtype=torch.float16, device="cuda")
    c = torch.empty(128, 128, dtype=torch.float16, device="cuda")
    _cublas_lib.cublasGemmEx(
        _cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N, 128, 128, 64,
        ctypes.byref(_alpha),
        b.data_ptr(), CUDA_R_16F, 64,
        a.data_ptr(), CUDA_R_16F, 64,
        ctypes.byref(_beta),
        c.data_ptr(), CUDA_R_16F, 128,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
    torch.cuda.synchronize()


def cublas_hgemm(a: torch.Tensor, b: torch.Tensor,
                 out: torch.Tensor | None = None) -> torch.Tensor:
    """cuBLAS CUBLAS_COMPUTE_16F — fp16 累加, 公平 baseline."""
    _init_cublas()
    M, K = a.shape; N = b.shape[0]
    if out is None:
        out = torch.empty(M, N, dtype=torch.float16, device="cuda")

    st = _cublas_lib.cublasGemmEx(
        _cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
        ctypes.byref(_alpha),
        b.data_ptr(), CUDA_R_16F, K,
        a.data_ptr(), CUDA_R_16F, K,
        ctypes.byref(_beta),
        out.data_ptr(), CUDA_R_16F, N,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    )
    if st != 0:
        raise RuntimeError(f"cublasGemmEx(COMPUTE_16F) failed: status={st}")
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Config & helpers
# ═════════════════════════════════════════════════════════════════════════

BENCH_SIZES: list[tuple[int, int, int]] = [
    (128, 256, 32), (256, 256, 64), (512, 512, 128),
    (256, 1024, 256), (1024, 256, 256), (1024, 1024, 512),
    (2048, 2048, 1024), (1536, 512, 1024), (512, 2048, 1024),
    (4096, 4096, 2048), (4096, 1024, 4096), (1024, 4096, 4096),
    (2048, 8192, 2048), (2048, 512, 2048),
]

WARMUP_ITERS = 10
BENCH_ITERS  = 100


def _tflops(m: int, n: int, k: int, elapsed_s: float) -> float:
    return (2.0 * m * n * k) / (elapsed_s * 1e12)


def _make_inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randn(m, k, dtype=torch.float16, device="cuda").contiguous(),
        torch.randn(n, k, dtype=torch.float16, device="cuda").contiguous(),
    )


def _median_sec(fn, *args, warmup=WARMUP_ITERS, iters=BENCH_ITERS) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record(); fn(*args); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(torch.tensor(times).median()) / 1000.0


def _geomean(xs: list[float]) -> float:
    pos = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in pos) / len(pos)) if pos else float("nan")


# ═════════════════════════════════════════════════════════════════════════
#  Benchmark — 正确性 + 性能合一
# ═════════════════════════════════════════════════════════════════════════

def run_benchmark(sizes=BENCH_SIZES) -> None:
    print("=" * 72)
    print("mygemm vs cuBLAS 16F  (都是 fp16 累加)  — 正确性 & 性能")
    print("=" * 72)
    print(f"  {'(M,N,K)':>20s} │ {'mygemm':>7s} │ {'cuBLAS':>7s} │ "
          f"{'speedup':>7s} │ {'err_my':>9s} │ {'err_cb':>9s}")
    print("-" * 72)

    speedups = []
    for m, n, k in sizes:
        a, b = _make_inputs(m, n, k)
        c_ref = torch.mm(a.float(), b.float().T)

        err_my = (mygemm(a, b).float()       - c_ref).abs().max().item()
        err_cb = (cublas_hgemm(a, b).float() - c_ref).abs().max().item()

        c_buf = torch.empty(m, n, dtype=torch.float16, device="cuda")
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        tf_my = _tflops(m, n, k, _median_sec(mygemm, a, b))
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        tf_cb = _tflops(m, n, k, _median_sec(cublas_hgemm, a, b, c_buf))

        sp = tf_my / tf_cb
        speedups.append(sp)
        print(f"  ({m:>5d},{n:>5d},{k:>5d}) │ {tf_my:>5.2f}T │ "
              f"{tf_cb:>5.2f}T │ {sp:>6.2f}x │ {err_my:>9.2e} │ {err_cb:>9.2e}")

    print("-" * 72)
    print(f"  Geomean speedup: {_geomean(speedups):.3f}x")
    print()


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main() -> None:
    if mygemm is None:
        print("[FATAL] extension_cpp 未安装。"); sys.exit(1)
    if not torch.cuda.is_available():
        print("[FATAL] CUDA 不可用。"); sys.exit(1)

    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  |  CUDA {torch.version.cuda}")
    print(f"Warmup : {WARMUP_ITERS}  |  Bench {BENCH_ITERS} iters\n")

    run_benchmark()


if __name__ == "__main__":
    main()
