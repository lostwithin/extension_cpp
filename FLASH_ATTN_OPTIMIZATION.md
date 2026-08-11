# Flash Attention CuTe Kernel 优化记录

## 环境

- **GPU**: NVIDIA GeForce RTX 4060 Laptop (Ada Lovelace, SM 8.9, 24 SMs, 100KB shared memory/SM)
- **PyTorch**: 2.8.0+cu128
- **Baseline**: Tri Dao flash-attn 2.8.3.post1, torch SDPA
- **测试形状**: (B, H, N, D), head dim ∈ {16, 32, 64, 128, 256}, N ∈ [128, 4096]

## 优化总览

```
原版:   0.84x vs Tri Dao │ 0.58x vs SDPA │ Compute 17% │ Occupancy 17%
  │
  ├─ Opt #1: __launch_bounds__(256, 2)
  │   → regs 240→128, occupancy 17%→31%, 性能基本持平 (瓶颈在 bank conflict)
  │
  ├─ Opt #2: Swizzle<3,3,3>
  │   → L1 waste 90%→46%, compute 17%→31%, 0.84x→1.14x vs Tri Dao
  │
  ├─ Opt #3: 尝试 cp_async V 预取
  │   → DRAM 才 5%, 不是瓶颈. 与 swizzle 冲突, 回退
  │
  ├─ Opt #4: 尝试 Grid 重排 (Q-block 在 x 轴)
  │   → 低 batch 时 L2 复用收益有限, 回退
  │
  └─ Opt #5: __expf 替换 exp
      → hardware MUFU.EX2, 1.14x→1.34x vs Tri Dao, 0.81x→0.96x vs SDPA
```

## 最终结果

| 指标 | 原版 | 优化后 | 改善 |
|------|------|--------|------|
| vs Tri Dao FA (geomean) | 0.84x | **1.34x** | +59% |
| vs torch SDPA (geomean) | 0.58x | **0.96x** | +65% |
| Registers/Thread | 240 | **128** | -47% |
| Achieved Occupancy | 16.6% | **30.9%** | +86% |
| Compute (SM) Throughput | 17.3% | **~35%** | +102% |
| L1/TEX Throughput | 81.9% | **46%** | -44% (消除浪费) |
| Duration (2048×64) | 884 us | ~770 us | -13% |

---

## 优化详解

### Opt #1: `__launch_bounds__(256, 2)`

**问题诊断 (ncu)**:
```
Registers Per Thread:  240  (上限 255)
Block Limit Registers: 1    (每 SM 只能跑 1 block)
Theoretical Occupancy: 16.67%
```

240 reg/thread × 256 threads = 61,440 regs/block. SM 只有 65,536 regs, 只能跑 1 block.

**修改**:
```cpp
template <typename Config>
__launch_bounds__(Config::NumThreads, 2)  // 告诉编译器 target 2 blocks/SM
__global__ void flash_attn_cute_kernel(...)
```

**效果**: 编译器自动减少寄存器分配, 128 regs/thread, 可以跑 2 blocks/SM.

**为何性能没涨?** L1/shared memory 带宽已达 90% — 瓶颈不是 occupancy, 是 bank conflict.

---

### Opt #2: `Swizzle<3,3,3>` — **关键优化**

**问题诊断 (ncu)**:
```
L1/TEX Throughput: 81.91% → 大量 shared memory 带宽被 bank conflict 浪费
Memory Throughput: 73.07%
Compute Throughput: 17.31%  → Tensor Core 空转等数据
```

`SM75_U32x4_LDSM_N` (ldmatrix) 从 shared memory 读取 Q/K 时, 每 4 个 32-bit 值可能落在同一个 bank, 产生 2-way 或 4-way bank conflict.

**修改**:
```cpp
// 之前: 无 swizzle
auto sQ = make_tensor(make_smem_ptr(psQ),
    make_layout(make_shape(Int<128>{}, Int<64>{}), GenRowMajor{}));

// 之后: 加 swizzle
using SmemSwizzle = Swizzle<3, 3, 3>;  // XOR bits [3,5] → ldmatrix 友好

constexpr int kShmQSize = cosize(composition(SmemSwizzle{},
    make_layout(make_shape(Int<BlockQO>{}, Int<HeadDim>{}), GenRowMajor{})));

__shared__ T psQ[kShmQSize];  // 使用 cosize 确保 swizzle 对齐

auto sQ = make_tensor(make_smem_ptr(psQ),
    composition(SmemSwizzle{},
        make_layout(make_shape(Int<BlockQO>{}, Int<HeadDim>{}), GenRowMajor{})));
```

Q/K/V 三个 shared memory buffer 都加上 swizzle.

**效果**:

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| Duration (2048×64) | 860 us | 489 us |
| L1 Throughput | 90% | 46% |
| Compute Throughput | 18% | 31% |
| Memory Throughput | 81% | 40% |

bank conflict 消除后, L1 带宽需求减半 (同样数据不再浪费在冲突重传上), Tensor Core 利用率翻倍.

---

### Opt #3: 尝试 cp_async V 预取

**分析**:
- DRAM throughput 仅 5% — global memory 不是瓶颈
- 主要瓶颈是 shared memory 带宽 (已通过 swizzle 解决)

**尝试**: 用 `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>` 异步拷贝 V, 在 softmax 计算时后台加载.

**问题**: `cp_async` 需要 16 字节对齐. `Swizzle<3,3,3>` 修改了 bit 3, 破坏 16 字节对齐. 改用 `Swizzle<3,4,3>` 后 d=256 仍出错, 且性能倒退了.

**结论**: DRAM 带宽充裕, 异步预取收益 < 更优 swizzle 收益. 放弃.

---

### Opt #4: 尝试 Grid 重排

**想法**: 将 grid 从 `(batch, head, Q_block)` 改为 `(Q_block, batch*head, 1)`, 使同一 (batch, head) 的连续 Q blocks 相邻调度, 复用 L2 cache 中的 K/V 数据.

**问题**: 单 stream 低 batch 场景下 L2 复用收益有限. 且 `blockIdx` 索引变化需要 kernel 配合修改. 回退.

---

### Opt #5: `__expf` 替换 `exp`

**分析**: softmax 热路径中的 `exp()` 使用软件级双精度实现. CUDA 有硬件 `MUFU.EX2` 指令, 通过 `__expf()` 调用.

**修改**:
```cpp
// 之前 (3 处)
tOrO(...) *= exp(prev_row_max(...) - new_row_max(...));
global_row_denominator(...) *= exp(prev_row_max(...) - new_row_max(...));
tSrS(...) = exp(tSrS(...) - new_row_max(...));

// 之后
tOrO(...) *= __expf(prev_row_max(...) - new_row_max(...));
global_row_denominator(...) *= __expf(prev_row_max(...) - new_row_max(...));
tSrS(...) = __expf(tSrS(...) - new_row_max(...));
```

**效果**: 1.14x → 1.34x vs Tri Dao, 0.81x → 0.96x vs SDPA.

---

## 剩余差距分析 (vs SDPA)

SDPA 底层使用 **CUTLASS 3.x**, 有以下我们未实现的优化:

| SDPA 优化 | 说明 | 改动量 |
|-----------|------|--------|
| Warp specialization | 不同 warp 分工 compute vs load | 大 (kernel 重写) |
| Persistent kernel | 一个 block 循环处理多个 tile | 大 |
| 2-stage KV pipeline | `cp_async` 预取 K/V, 需要双倍 shared memory | 中 |
| PTX-tuned MMA | 手动 `wgmma` 指令替代 CuTe 自动生成 | 大 |
| SMEM reuse across iterations | 减少 shared memory 分配 | 中 |

当前 0.96x vs SDPA 说明仅用**通用优化**就能追到 96%. 最后的 4% 需要**架构级改动**.

---

## 已尝试但无效/负面的优化

| 优化 | 原因 |
|------|------|
| cp_async V 预取 | DRAM 带宽充裕, 不与 swizzle 兼容 |
| Grid 重排 | 单 stream 低 batch 无显著收益 |
| 去掉 V load 前的 `__syncthreads()` | TiledCopy 需要所有线程参与 |
| Softmax state → shared memory | 超出 48KB shared memory 限制 |

---

## 关键教训

1. **用 ncu 数据驱动决策**, 不要靠直觉. DRAM 5% → cp_async 无效.
2. **Shared memory swizzle 是 CUDA kernel 性能第一要素** — bank conflict 消除是本次最大收益.
3. **`__launch_bounds__` 零成本** — 一行编译器提示就能翻倍 occupancy.
4. **`__expf` vs `exp`** — float 精度足够 attention, 硬件加速显著.
5. **RTX 4060 ≠ A100** — 24 SMs, 100KB shared memory 的特殊限制要求专属 tile size.
