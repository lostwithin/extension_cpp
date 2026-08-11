#!/usr/bin/env python3
"""最小 NCU profile 脚本 — 单尺寸 + 少量迭代，专门给 ncu 抓 trace 用。

用法:
    ncu --set full --kernel-name regex:hgemm_mma_stages -o gemm_report python test/profile_gemm.py
    ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --kernel-name regex:hgemm_mma_stages python test/profile_gemm.py
"""

import torch
from extension_cpp.ops import mygemm, mygemm_small

M, N, K = 2048, 2048, 1024  # 典型大尺寸，走 large tile (128×256)

a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(N, K, dtype=torch.float16, device="cuda")
torch.cuda.synchronize()

print(f"M={M} N={N} K={K}  —  large tile  (128×256)")
for _ in range(3):
    mygemm(a, b)
    torch.cuda.synchronize()

# 小尺寸强制走 small tile (64×128)
Ms, Ns, Ks = 512, 512, 128
as_ = torch.randn(Ms, Ks, dtype=torch.float16, device="cuda")
bs_ = torch.randn(Ns, Ks, dtype=torch.float16, device="cuda")
torch.cuda.synchronize()

print(f"M={Ms} N={Ns} K={Ks}  —  small tile (64×128)")
for _ in range(3):
    mygemm(as_, bs_)
    torch.cuda.synchronize()
