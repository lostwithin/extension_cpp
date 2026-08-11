# Flash Attention CuTe Kernel 优化总结

> GPU: NVIDIA GeForce RTX 4060 Laptop (Ada Lovelace, SM89, 24 SMs, 24 MB L2)  
> 基准: Qwen3-0.6B (Hq=16, Hkv=8, D=128, GQA 2:1 → MHA expand)  
> 所有数据: `torch.cuda.Event` 计时, median of 100 iterations, `cos_sim > 0.999` vs SDPA

---

## 优化历程

### Phase 0 — 原点

**文件**: `flashAttention.cu`  
SM80 MMA atoms (`SM80_16x8x8`), half + bf16, 硬编码 `Swizzle<3,3,3>`, 朴素的 K/V 串行加载 (4× `__syncthreads`/KV 迭代), `__expf` 软件模拟, 同步向量化 gmem→smem

| 版本 | LLM prefill vs SDPA | LLM prefill vs tri-dao | decode vs SDPA |
|---|---|---|---|
| 原始 | 0.567x | — | ❌ 不支持 cross-attn |

---

### Phase 1 — 架构无关优化 (`V1`: opt#1–#7)

移植自 `flashAttention_T4.cu` 并适配 SM80/Ada:

| # | 优化 | 内容 | 预期收益 |
|---|---|---|---|
| opt2 | K/V 加载重排 + 同步减半 | K/V gmem→smem 背靠背发射, `__syncthreads` 4→2 | 访存延迟并流 |
| opt3 | 自适应 smem swizzle | 根据 HeadDim 选择 `SmemSwizzleBits` (1/2/3), 替代硬编码 `Swizzle<3,3,3>` | HeadDim≠64 的 bank conflict 消除 |
| opt5 | exp2f + 行因子外提 + 倒数 | `exp2f` (MUFU.EX2 硬件指令), `log2(e)` 折进 Q scaler; 每行一次 rescale 因子替代逐元素 exp; 最终归一化用 reciprocal× 替代逐元素 ÷ | softmax ~2× 加速 |
| opt7 | grid 重排 (L2 复用) | x 维 = QO 分块, 相邻 block 同 `(batch,head)` 共享 K/V 蹭 24 MB L2 | 内存带宽 (~10%) |
| PV fix | `SmemLayoutVtNoSwizzle` | 非 swizzle layout 给 `partition_fragment_B` 定形, 避免动态 stride 导致 `mma_unpack` 断言失败 | 正确性 |
| ENG | 工程健壮性 | `TORCH_CHECK` 替代 `assert`, `at::cuda::getCurrentCUDAStream()`, `OptionalCUDAGuard` | NDEBUG 安全, 多卡, stream |

**基准** (d=64 小 shape):

| Shape | 原始 | V1 | 提升 |
|---|---|---|---|
| (1,8,512,64) | 0.150ms | 0.147ms | 1.02x |
| (1,4,128,256) | 0.093ms | 0.058ms | **1.60x** |
| (1,8,256,32) | 0.032ms | 0.026ms | **1.23x** |

---

### Phase 2 — NCU 性能剖析指导阶段 (`V2`: opt#8–#9)

NCU profiling (d64_s512, SM89):

| 指标 | 值 | 诊断 |
|---|---|---|
| SM throughput | 21.95% | 计算严重欠饱和 |
| DRAM throughput | 13.56% | 带宽远未跑满 |
| Tensor pipe | ~22% | **tensor core 78% 时段闲置** |
| Achieved occupancy | 22.03% | 每 SM 仅 ~11 活跃 warp (上限 48) |
| Local memory spilling | 44,288 req (100%) | **128 regs/thread 不足支撑 8 warps 配置, 大量 spill 到 lmem** |
| IPC (active) | 0.67 | 大量 stall 周期 |
| L1 hit rate | 13.11% | 流式访存模式, L1 不适用 |
| NCU 提示 | uncoalesced global +16.4%, shared excessive +9.1%, SM imbalance +19% | 理论优化上限 ~45% |

**根因诊断**:
1. 瓶颈是 **softmax ↔ tensor-core 的串行依赖**, 不是内存带宽 (<22% DRAM)
2. 寄存器压力 → spill + 低 occupancy → 停顿直接暴露
3. 大序列下效率差距扩大 (长 KV 循环 × 每轮 softmax 串行开销)

**opt9 — Spilling 修复**: d=64 配置从 `(BlockQO=128, BlockKV=128, 8w)` → `(BlockQO=64, BlockKV=128, 4w)`, 寄存器从 ~128 (强制 spill) → ~240 (零 spill)

**opt8 — cp.async K/V 加载**: `SM80_CP_ASYNC_CACHEGLOBAL` 替代同步向量化加载, 利用异步拷贝引擎

| Shape | V1 | V2 | 提升 |
|---|---|---|---|
| LLaMA-7B pref 4K b=4 | 68.98ms | 46.37ms | **1.49x** |
| LLaMA-7B pref 8K b=2 | 135.48ms | 77.35ms | **1.75x** |
| **LLM prefill geomean** | — | — | **1.62x** |

**说明**: opt9 (spilling 修复) 是 V2 的核心贡献 (~1.4–1.7x)。opt8 (cp.async) 在 prefill 上贡献有限 (<5%), 但为 decode 埋下基础。

---

### Phase 3 — 2-stage 双缓冲 cp.async 流水线 (`V3`/`V3b`: opt#10)

**V2 的 cp.async 是"发射即等"** — `cp_async_wait<0>()` 紧跟 copy, 加载与计算零重叠, 只拿到合并访问收益。

**实现真正的 2-stage 流水线**:
- 动态 smem (`extern __shared__`), K/V 各 2 份 ping-pong buffer
- Prologue 预取 KV[0] → stage 0
- 循环: `wait<0>() → barrier → 发射 cp.async KV[i+1] → fence → compute KV[i]`
- K/V 的 copy 与当前块 QK+softmax+PV **重叠**

**V3 (BlockKV=128→80KB smem) vs V3b (BlockKV=32→48KB smem)**:

| 版本 | smem | blocks/SM | prefill vs SDPA | decode vs SDPA |
|---|---|---|---|---|
| V2 (no double-buf) | 48KB | 2 | 0.819x | ❌ |
| V3 (BlockKV=64) | 80KB | 1 | **0.741x** ⬇️ | **0.955x** |
| V3b (BlockKV=32) | 48KB | 2 | **0.821x** | **0.916x** |

**关键认识**: 双缓冲在 prefill 上是净负向 — 吃掉一个 block/SM 换来的 overlap 不如这个 block 本身提供的延迟隐藏。因为 **DRAM 带宽才用了 13–22%, 内存不是瓶颈**。双缓冲的值在于 decode (小 grid) 和未来的内存墙场景。

最终选择 **V3b** — prefill 持平, decode 可用且强。

---

### Phase 4 — flash-decoding / split-KV (`V4`, 评估后**已回退**)

> **结论: 已实现并验证正确, 但在 4060 + 真实 Qwen3 (H=16) 上无收益, 故回退到 V3b。**
> 保留本节作为评估记录。当前部署 = **V3b**。

**需求**: decode 场景 (N_QO≪N_KV), SDPA 自动切到 `flash_fwd_splitkv` 走 KV 并行化。我们的 kernel 缺乏这一路径。

**实现**:
- `flash_attn_cute_kernel<Config, SplitKV>` — 编译期 flag, KV 循环限 `[kv_begin, kv_end)`, epilogue 改写未归一化 numerator (fp32) + 每行 (m,l) 到 workspace
- `flash_attn_combine_kernel` — 合并各 split 的 partial 结果 (online-softmax, base-2)  
   `m = max mᵢ`, `O = Σ O_部分ᵢ · exp2(mᵢ-m) / Σ lᵢ · exp2(mᵢ-m)`
- 分发启发式: `num_splits = 1` 当 base grid ≥ 2×SM (prefill 无改动), 否则按 `ceil(2×SM / base_grid)` 切分 KV

**结果**:

| 场景 | no-split (V3b) | split-KV (V4) | 结论 |
|---|---|---|---|
| H=1 decode KV=8192 | 0.924ms | **0.496ms (1.86x)** | 极端 grid 饥饿显著见效 |
| H=2…16 decode | ~0.92ms | ~0.92ms | 持平 (头数已提供并行) |
| Qwen3-0.6B (H=16) decode geomean vs SDPA | 1.038x | 1.031x | 差距在噪声内 |
| prefill 所有尺寸 | 不变 | 不变 | 无回退 |

**split-KV 在 H=16 Qwen3 上不增产的原因**: 16 个注意力头 = 16 个并行 block, 已在 24-SM GPU 上提供了 split-KV 想制造的并行度。仅当少数头 (H≤4) 或超大 GPU 时, split-KV 的收益才显性化。

**回退决策**: H=1 (grid 完全饥饿) 在真实模型中不存在 —— 任何真实 decode 都有 ≥8 个 query head, 在 24-SM 的 4060 上已基本填满。split-KV 在真实 shape 上不仅无收益, 还因启发式误触发 (num_splits=3) 而带来 ~1-7% 的 combine 开销, 同时多背 ~100 行代码 (模板分支 + combine kernel + workspace + 启发式)。**针对 4060 + Qwen3 的部署目标, 这是负资产, 已回退到纯双缓冲的 V3b。** 若未来跨平台到 A100/H100 (108-132 SM, 16 blocks 严重欠载) 做低-batch decode, 再重新引入 split-KV 并配保守 gate 更合适。

> 备注: split-KV 的完整实现 (SplitKV 模板分支 + `flash_attn_combine_kernel` + 分发) 已在 git 历史/本记录中留存, 需要时可复原。

---

## 精简的最终基准 (删除冗余 tri-dao baseline)

SDPA 在 PyTorch 2.8 的 FLASH 后端 = `pytorch_flash::flash_fwd_kernel` (内置的 FlashAttention-2 实现), 与 `flash_attn_func` 同源。  
Profiler 确认: 所有 benchmark shape 均走 FLASH 后端 (cuDNN 未被选中)。

### PREFILL (N_QO=N_KV) — V4

| shape | custom | SDPA (FA2) | naive (非-flash) | vs SDPA | vs naive |
|---|---|---|---|---|---|
| 512 b=1 | 0.372ms | 0.339ms | 0.440ms | 0.91x | 1.2x |
| 1024 b=1 | 1.164ms | 1.165ms | 1.690ms | **1.00x** | 1.5x |
| 2048 b=1 | 4.339ms | 3.829ms | 6.040ms | 0.88x | 1.4x |
| 4096 b=1 | 12.13ms | 9.625ms | 18.74ms | 0.79x | 1.5x |
| 2048 b=4 | 6.182ms | 5.302ms | 14.46ms | 0.86x | 2.3x |
| 4096 b=2 | 9.539ms | 9.179ms | 29.74ms | **0.96x** | **3.1x** |
| **geomean** | | | | **0.899x** | **1.7x** |

### DECODE-CHUNK (N_QO=64) — V4

| shape | custom | SDPA (FA2,split-KV) | naive | vs SDPA | vs naive |
|---|---|---|---|---|---|
| KV=1024 | 0.042ms | 0.056ms | 0.079ms | **1.34x** | 1.9x |
| KV=2048 | 0.091ms | 0.085ms | 0.115ms | 0.93x | 1.3x |
| KV=4096 | 0.158ms | 0.179ms | 0.347ms | **1.14x** | 2.2x |
| KV=8192 | 0.419ms | 0.333ms | 0.708ms | 0.79x | 1.7x |
| **geomean** | | | | **1.031x** | **1.7x** |

---

## 版本演进速查

| 版本 | 文件 | smem | 核心改动 | vs SDPA (prefill) | vs SDPA (decode) |
|---|---|---|---|---|---|
| 原始 | `.so.bak` | 48KB static | 基础 FA2 kernel | 0.567x | ❌ |
| V1 | `.so.new_version` | 48KB static | T4 移植: swizzle+L2+exp2f | 0.949x* | ❌ |
| V2 | `.so.v2` | 48KB static | spilling fix (d64: 128→64 BlockQO) + cp.async | 0.920x | ❌ |
| V3 | `.so.v3` | 80KB dynamic | 2-stage 双缓冲流水线 (BlockKV=64) | 0.741x | 0.955x |
| **V3b** (当前部署) | **`_C.abi3.so`** = `.so.v3b` | 48KB dynamic | 2-stage 双缓冲 (BlockKV=32, 保 2 blocks/SM) | 0.821x | 0.916x |
| ~~V4~~ (已回退) | `.so.v4` | 48KB dynamic | V3b + split-KV; 真实 shape 无收益, 已回退 | (0.899x) | (1.031x) |

\* V1 数据来自小 shapes benchmark (head_dim 16/32/64/128/256 mixed), 与后续 LLM shapes 不可直接比较

---

## 剩余差距分析

以下数字来自 NCU profiling (d128, SM89, V3b):

| 差距来源 | 量级 | 说明 |
|---|---|---|
| softmax ↔ MMA 串行 | **最大** | 每次 KV 迭代中, softmax (CUDA core FP32 + shuffle) 夹在两次 tensor core MMA 之间, 致 tensor core ~78% 时间停摆 |
| SDPA Q-tile = 128 | ~15–20% | 我们的 Q-tile=64 (受限于寄存器/smem); SDPA 的 128 把每 KV 加载分摊到 2× query, 算术强度高一倍 |
| occupancy = 22% (实测) | ~10% | 理论 33%, 实测更低, 每 SM 仅 ~11 活跃 warp vs 48 上限 |
| 非合并 global access (16%) | ~5% | 部分线程的内存访问未 128B 对齐 |
| shared bank conflict (9%) | ~3% | swizzle 后仍有冲突 |

**要追平 SDPA 需要的最重要的工程工作**: softmax 与 MMA 的调度重叠 — 让 tensor core 不停流, 把 softmax 从关键路径移开与下一块 QK gemm 并行。这在 Ada (无 Hopper WGMMA) 上实现复杂但收益最高。

## 当前部署状态

**部署版本: V3b** (2-stage 双缓冲, 无 split-KV) —— split-KV 评估后回退。  
`.so` 路径: `extension_cpp/extension_cpp/_C.abi3.so` (= `.so.v3b`)  
备份: `.so.v2` / `.so.v3` / `.so.v3b` / `.so.v4`  
源码: `extension_cpp/csrc/cuda/flashAttention.cu` (463 行, split-KV 已移除)  
benchmark 脚本: `test/benchmark_qwen3.py` (三方对比: custom vs SDPA vs naive)

> ⚠️ **构建/运行注意**: 本机 nvcc 是 CUDA 13.1 而 PyTorch 是 cu12.8, 需临时 monkey-patch `_check_cuda_version` 才能编译 (已在会话中用完即撤)。正式构建应在 CUDA 版本匹配的环境。运行测试时**从仓库根目录之外的 cwd** (如 `/tmp`) 启动 python —— 仓库根的外层 `extension_cpp/` 是无 `__init__.py` 的 namespace 目录, 会劫持 `import extension_cpp` 导致 `_C` 先于 `ops` 注册失败 (`operator extension_cpp::mymuladd does not exist`); 从 `/tmp` 或安装路径导入则正常。
