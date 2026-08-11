#include <ATen/Operators.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <type_traits>
#include "utils.h"

namespace extension_cpp {

using namespace cute;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));  \
  } while (0)

template <typename T_, int BlockQO_, int BlockKV_, int HeadDim_, int NWarpsPerSM_>
struct FlashAttnConfig {
  using T = T_;
  static constexpr int NWarpsPerSM = NWarpsPerSM_;
  static constexpr int NumThreads = NWarpsPerSM * 32;
  static constexpr int BlockQO = BlockQO_;
  static constexpr int BlockKV = BlockKV_;
  static constexpr int HeadDim = HeadDim_;

  // Gmem to Smem
  using GmemCopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>, T>;
  static constexpr int GmemValsPerLoad = sizeof(uint128_t) / sizeof(T);
  static constexpr int GmemThreadsPerRow = HeadDim / GmemValsPerLoad;
  using TiledCopyQKVO = decltype(make_tiled_copy(
      GmemCopyAtom{},
      make_layout(Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{}, GenRowMajor{}),
      make_layout(Shape<_1, Int<GmemValsPerLoad>>{}, GenRowMajor{})));
  static_assert(Int<NumThreads / GmemThreadsPerRow>::value <= BlockQO, "NumThreads <= BlockQO assertion failed");

  using GmemCopyAtomAsync = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>;
  using TiledCopyKV = decltype(make_tiled_copy(
      GmemCopyAtomAsync{},
      make_layout(Shape<Int<NumThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>{}, GenRowMajor{}),
      make_layout(Shape<_1, Int<GmemValsPerLoad>>{}, GenRowMajor{})));

  // Smem to Rmem
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, T>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, T>;
  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(uint128_t) * 8>, T>;

  static constexpr int SmemAtomWidth = HeadDim < 64 ? HeadDim : 64;
  static constexpr int SmemSwizzleBits = SmemAtomWidth == 64 ? 3 : (SmemAtomWidth == 32 ? 2 : 1);
  using SmemLayoutAtom = decltype(composition(
      Swizzle<SmemSwizzleBits, 3, 3>{},
      Layout<Shape<_8, Int<SmemAtomWidth>>, Stride<Int<SmemAtomWidth>, _1>>{}));
  using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BlockQO>, Int<HeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BlockKV>, Int<HeadDim>>{}));

  using SmemLayoutVt = decltype(composition(
      SmemLayoutKV{}, make_layout(Shape<Int<HeadDim>, Int<BlockKV>>{}, GenRowMajor{})));

  using SmemLayoutVtNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVt{}));

  static constexpr int SmemElemsQ  = cosize(SmemLayoutQ{});
  static constexpr int SmemElemsKV = cosize(SmemLayoutKV{});
  static constexpr int SmemBytes   = int(sizeof(T)) * (SmemElemsQ + 4 * SmemElemsKV);

  static_assert(std::is_same_v<T, half_t> || std::is_same_v<T, bfloat16_t>);
  using MMA_Atom = std::conditional_t<std::is_same_v<T, half_t>,
                                      MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
                                      MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>>;

  using TiledMMA = decltype(make_tiled_mma(
      MMA_Atom{}, make_layout(Shape<Int<NWarpsPerSM>, _1, _1>{}, GenRowMajor{}),
      Tile<Int<16 * NWarpsPerSM>, _16, _16>{}));

  static_assert(16 * NWarpsPerSM <= BlockQO && 16 <= BlockKV && 16 <= HeadDim, "Block and HeadDim size assertions failed");
  static_assert(size(TiledMMA{}) == NumThreads && size(TiledMMA{}) == size(TiledCopyQKVO{}));
};

template <typename FlashAttnConfig_>
__launch_bounds__(FlashAttnConfig_::NumThreads, 2)
__global__ void flash_attn_cute_kernel(typename FlashAttnConfig_::T *pQ,
                                       typename FlashAttnConfig_::T *pK,
                                       typename FlashAttnConfig_::T *pV,
                                       typename FlashAttnConfig_::T *pO, int B,
                                       int H, int N_QO, int N_KV, int D,
                                       float scaler) {
  using namespace cute;
  using T = typename FlashAttnConfig_::T;
  constexpr int BlockQO = FlashAttnConfig_::BlockQO;
  constexpr int BlockKV = FlashAttnConfig_::BlockKV;
  constexpr int HeadDim = FlashAttnConfig_::HeadDim;
  using TiledCopy = typename FlashAttnConfig_::TiledCopyQKVO;
  using TiledCopyKV = typename FlashAttnConfig_::TiledCopyKV;            
  using SmemCopyAtom = typename FlashAttnConfig_::SmemCopyAtom;
  using SmemCopyAtomTransposed = typename FlashAttnConfig_::SmemCopyAtomTransposed;
  using SmemCopyAtomO = typename FlashAttnConfig_::SmemCopyAtomO;
  using TiledMMA = typename FlashAttnConfig_::TiledMMA;
  using SmemLayoutQ = typename FlashAttnConfig_::SmemLayoutQ;
  using SmemLayoutKV = typename FlashAttnConfig_::SmemLayoutKV;
  using SmemLayoutVt = typename FlashAttnConfig_::SmemLayoutVt;
  using SmemLayoutVtNoSwizzle = typename FlashAttnConfig_::SmemLayoutVtNoSwizzle;
  assert(HeadDim == D);


  const int bx = blockIdx.z, by = blockIdx.y, bz = blockIdx.x; // (batch, head, QO分块)
  const int tx = threadIdx.x;

  auto Q = make_tensor(make_gmem_ptr(pQ), make_layout(make_shape(B, H, N_QO, HeadDim), GenRowMajor{}));
  auto O = make_tensor(make_gmem_ptr(pO), make_layout(make_shape(B, H, N_QO, HeadDim), GenRowMajor{}));
  auto K = make_tensor(make_gmem_ptr(pK), make_layout(make_shape(B, H, N_KV, HeadDim), GenRowMajor{}));
  auto V = make_tensor(make_gmem_ptr(pV), make_layout(make_shape(B, H, N_KV, HeadDim), GenRowMajor{}));

  auto gQ = local_tile(Q, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}), make_coord(bx, by, bz, 0))(0, 0, _, _);
  auto gO = local_tile(O, make_shape(_1{}, _1{}, Int<BlockQO>{}, Int<HeadDim>{}), make_coord(bx, by, bz, 0))(0, 0, _, _);
  auto gK = local_tile(K, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}), make_coord(bx, by, _, 0))(0, 0, _, _, _);
  auto gV = local_tile(V, make_shape(_1{}, _1{}, Int<BlockKV>{}, Int<HeadDim>{}), make_coord(bx, by, _, 0))(0, 0, _, _, _);


  extern __shared__ __align__(16) char smem_raw[];
  T *smQ = reinterpret_cast<T *>(smem_raw);
  T *smK = smQ + FlashAttnConfig_::SmemElemsQ;
  T *smV = smK + 2 * FlashAttnConfig_::SmemElemsKV;
  constexpr int kvElems = FlashAttnConfig_::SmemElemsKV;

  auto sQ = make_tensor(make_smem_ptr(smQ), SmemLayoutQ{});


  TiledCopy tiled_copy;
  auto thr_copy = tiled_copy.get_slice(tx);
  auto tQgQ = thr_copy.partition_S(gQ);
  auto tQsQ = thr_copy.partition_D(sQ);

  TiledCopyKV tiled_copy_kv;
  auto thr_copy_kv = tiled_copy_kv.get_slice(tx);
  auto tKgK = thr_copy_kv.partition_S(gK);
  auto tVgV = thr_copy_kv.partition_S(gV);

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tx);
  auto tSrQ = thr_mma.partition_fragment_A(sQ);
  auto tSrK  = thr_mma.partition_fragment_B(make_tensor(make_smem_ptr(smK), SmemLayoutKV{}));
  auto tSrS  = partition_fragment_C(tiled_mma, Shape<Int<BlockQO>, Int<BlockKV>>{});
  auto tOrVt = thr_mma.partition_fragment_B(make_tensor(make_smem_ptr(smV), SmemLayoutVtNoSwizzle{}));
  auto tOrO  = partition_fragment_C(tiled_mma, Shape<Int<BlockQO>, Int<HeadDim>>{});
  clear(tOrO);

  auto tiled_s2r_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_Q = tiled_s2r_copy_Q.get_slice(tx);
  auto tXsQ = thr_s2r_copy_Q.partition_S(sQ);
  auto tXrQ = thr_s2r_copy_Q.retile_D(tSrQ);

  auto tiled_s2r_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
  auto thr_s2r_copy_K = tiled_s2r_copy_K.get_slice(tx);
  auto tXrK = thr_s2r_copy_K.retile_D(tSrK); 

  auto tiled_s2r_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
  auto thr_s2r_copy_V = tiled_s2r_copy_V.get_slice(tx);
  auto tXrVt = thr_s2r_copy_V.retile_D(tOrVt);

  auto prev_row_max = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(prev_row_max, -1e20);
  auto global_row_denominator = make_tensor<float>(make_shape(_2{}, Int<size<1>(tSrS)>{}));
  fill(global_row_denominator, 0);

  copy(tiled_copy, tQgQ, tQsQ);
  for (int i = 0; i < size(tQsQ); i++) {
    tQsQ(i) = static_cast<T>(scaler) * tQsQ(i);
  }
  __syncthreads();
  copy(tiled_s2r_copy_Q, tXsQ, tXrQ);

  const int nKV = size<2>(gK);

  {
    auto sK0 = make_tensor(make_smem_ptr(smK), SmemLayoutKV{});
    auto sV0 = make_tensor(make_smem_ptr(smV), SmemLayoutKV{});
    copy(tiled_copy_kv, tKgK(_, _, _, 0), thr_copy_kv.partition_D(sK0));
    copy(tiled_copy_kv, tVgV(_, _, _, 0), thr_copy_kv.partition_D(sV0));
    cp_async_fence();
  }

  for (int blkKVIdx = 0; blkKVIdx < nKV; ++blkKVIdx) {
    const int cur = blkKVIdx & 1;

    cp_async_wait<0>();
    __syncthreads();

    if (blkKVIdx + 1 < nKV) {
      const int nxt = (blkKVIdx + 1) & 1;
      auto sKn = make_tensor(make_smem_ptr(smK + nxt * kvElems), SmemLayoutKV{});
      auto sVn = make_tensor(make_smem_ptr(smV + nxt * kvElems), SmemLayoutKV{});
      copy(tiled_copy_kv, tKgK(_, _, _, blkKVIdx + 1), thr_copy_kv.partition_D(sKn));
      copy(tiled_copy_kv, tVgV(_, _, _, blkKVIdx + 1), thr_copy_kv.partition_D(sVn));
      cp_async_fence();
    }

    auto sKc  = make_tensor(make_smem_ptr(smK + cur * kvElems), SmemLayoutKV{});
    auto sVtc = make_tensor(make_smem_ptr(smV + cur * kvElems), SmemLayoutVt{});

    copy(tiled_s2r_copy_K, thr_s2r_copy_K.partition_S(sKc), tXrK);
    clear(tSrS);
    gemm(tiled_mma, tSrQ, tSrK, tSrS);

    auto new_row_max = make_fragment_like(prev_row_max);
    fill(new_row_max, -1e20);

    for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
        for (int col_rep_idx = 0; col_rep_idx < size<2>(tSrS); ++col_rep_idx) {
          int row_idx = val_idx / 2;
          new_row_max(row_idx, row_rep_idx) = max(new_row_max(row_idx, row_rep_idx), tSrS(val_idx, row_rep_idx, col_rep_idx));
        }
      }
    }

    for (int row_idx = 0; row_idx < size<0>(new_row_max); ++row_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
        new_row_max(row_idx, row_rep_idx) = max(new_row_max(row_idx, row_rep_idx), __shfl_xor_sync(0xffffffff, new_row_max(row_idx, row_rep_idx), 1));
        new_row_max(row_idx, row_rep_idx) = max(new_row_max(row_idx, row_rep_idx), __shfl_xor_sync(0xffffffff, new_row_max(row_idx, row_rep_idx), 2));
      }
    }

    for (int row_idx = 0; row_idx < size<0>(new_row_max); ++row_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(new_row_max); ++row_rep_idx) {
        new_row_max(row_idx, row_rep_idx) = max(prev_row_max(row_idx, row_rep_idx), new_row_max(row_idx, row_rep_idx));
      }
    }

    auto row_scale = make_fragment_like(prev_row_max);
    for (int row_idx = 0; row_idx < size<0>(new_row_max); ++row_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(new_row_max); ++row_rep_idx) {
        row_scale(row_idx, row_rep_idx) = exp2f(prev_row_max(row_idx, row_rep_idx) - new_row_max(row_idx, row_rep_idx));
        global_row_denominator(row_idx, row_rep_idx) *= row_scale(row_idx, row_rep_idx);
      }
    }

    for (int val_idx = 0; val_idx < size<0>(tOrO); ++val_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tOrO); ++row_rep_idx) {
        for (int col_rep_idx = 0; col_rep_idx < size<2>(tOrO); ++col_rep_idx) {
          int row_idx = val_idx / 2;
          tOrO(val_idx, row_rep_idx, col_rep_idx) *= row_scale(row_idx, row_rep_idx);
        }
      }
    }

    for (int val_idx = 0; val_idx < size<0>(tSrS); ++val_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
        for (int col_rep_idx = 0; col_rep_idx < size<2>(tSrS); ++col_rep_idx) {
          int row_idx = val_idx / 2;
          tSrS(val_idx, row_rep_idx, col_rep_idx) = exp2f(tSrS(val_idx, row_rep_idx, col_rep_idx) - new_row_max(row_idx, row_rep_idx));
          global_row_denominator(row_idx, row_rep_idx) += tSrS(val_idx, row_rep_idx, col_rep_idx);
        }
      }
    }

    for (int row_idx = 0; row_idx < size<0, 0>(tSrS); ++row_idx) {
      for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
        prev_row_max(row_idx, row_rep_idx) = new_row_max(row_idx, row_rep_idx);
      }
    }

    auto tOrS = make_tensor<T>(tSrS.layout());
    for (int i = 0; i < size(tOrS); ++i) {
      tOrS(i) = static_cast<T>(tSrS(i));
    }

    static_assert(tiled_mma.get_layoutA_TV() == tiled_mma.get_layoutC_TV(), "Layout mismatch assertion");

    copy(tiled_s2r_copy_V, thr_s2r_copy_V.partition_S(sVtc), tXrVt);
    gemm(tiled_mma, tOrS, tOrVt, tOrO);
  }

  for (int row_idx = 0; row_idx < size<0, 0>(tSrS); ++row_idx) {
    for (int row_rep_idx = 0; row_rep_idx < size<1>(tSrS); ++row_rep_idx) {
      global_row_denominator(row_idx, row_rep_idx) += __shfl_xor_sync(0xffffffff, global_row_denominator(row_idx, row_rep_idx), 1);
      global_row_denominator(row_idx, row_rep_idx) += __shfl_xor_sync(0xffffffff, global_row_denominator(row_idx, row_rep_idx), 2);
    }
  }

  for (int row_idx = 0; row_idx < size<0>(global_row_denominator); ++row_idx) {
    for (int row_rep_idx = 0; row_rep_idx < size<1>(global_row_denominator); ++row_rep_idx) {
      global_row_denominator(row_idx, row_rep_idx) = 1.0f / global_row_denominator(row_idx, row_rep_idx);
    }
  }
  for (int val_idx = 0; val_idx < size<0>(tOrO); ++val_idx) {
    for (int row_rep_idx = 0; row_rep_idx < size<1>(tOrO); ++row_rep_idx) {
      for (int col_rep_idx = 0; col_rep_idx < size<2>(tOrO); ++col_rep_idx) {
        int row_idx = val_idx / 2;
        tOrO(val_idx, row_rep_idx, col_rep_idx) *= global_row_denominator(row_idx, row_rep_idx);
      }
    }
  }

  auto tiled_r2s_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto thr_r2s_copy_O = tiled_r2s_copy_O.get_slice(tx);
  auto tXrO = thr_r2s_copy_O.retile_S(tOrO);
  auto tXsO = thr_r2s_copy_O.partition_D(gO);
  copy(tiled_r2s_copy_O, tXrO, tXsO);
}

static void sanity_check(const torch::Tensor &Q, const torch::Tensor &K,
                         const torch::Tensor &V, const torch::Tensor &O) {
  TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0) && Q.size(0) == O.size(0), "Batch size mismatch");
  TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(1) == V.size(1) && Q.size(1) == O.size(1), "Head count mismatch (GQA: expand KV heads first)");
  TORCH_CHECK(K.size(2) == V.size(2), "K/V sequence length mismatch");
  TORCH_CHECK(Q.size(2) == O.size(2), "Q/O sequence length mismatch");
  TORCH_CHECK(Q.size(3) == K.size(3) && Q.size(3) == V.size(3) && Q.size(3) == O.size(3), "Hidden size mismatch");
}

template <typename T, int BlockQO, int BlockKV, int HeadDim, int NWarpsPerSM>
static void launch_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  using config = FlashAttnConfig<T, BlockQO, BlockKV, HeadDim, NWarpsPerSM>;
  sanity_check(Q, K, V, O);

  const int b = Q.size(0);
  const int h = Q.size(1);
  const int n_qo = Q.size(2); 
  const int n_kv = K.size(2);
  const int d = Q.size(3);

  float scaler = 1.4426950408889634f / sqrt((float)d);

  TORCH_CHECK(n_qo % BlockQO == 0, "N_QO (", n_qo, ") must be a multiple of BlockQO=", BlockQO);
  TORCH_CHECK(n_kv % BlockKV == 0, "N_KV (", n_kv, ") must be a multiple of BlockKV=", BlockKV);

  dim3 block(size(config::NumThreads));
  dim3 grid(n_qo / BlockQO, h, b);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto kernel = flash_attn_cute_kernel<config>;
  constexpr int smem_bytes = config::SmemBytes;
  if constexpr (smem_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }

  kernel<<<grid, block, smem_bytes, stream>>>(
      reinterpret_cast<T *>(Q.data_ptr()),
      reinterpret_cast<T *>(K.data_ptr()),
      reinterpret_cast<T *>(V.data_ptr()),
      reinterpret_cast<T *>(O.data_ptr()), b, h, n_qo, n_kv, d, scaler);
  CUDA_CHECK(cudaGetLastError());
}

void flash_attn_cute(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  const int d = Q.size(3);

  if (Q.dtype() == at::kHalf) {
    switch (d) {
    case 16:  launch_kernel<half_t, 128, 128, 16, 8>(Q, K, V, O);  break;
    case 32:  launch_kernel<half_t, 128, 128, 32, 8>(Q, K, V, O);  break;
    case 64:  launch_kernel<half_t, 64, 64, 64, 4>(Q, K, V, O);    break;
    case 128: launch_kernel<half_t, 64, 32, 128, 4>(Q, K, V, O);   break; 
    case 256: launch_kernel<half_t, 32, 32, 256, 2>(Q, K, V, O);   break;
    default:
      throw std::runtime_error("Unsupported headdim for half");
    }
  } else if (Q.dtype() == at::kBFloat16) {
    switch (d) {
    case 16:  launch_kernel<bfloat16_t, 128, 128, 16, 8>(Q, K, V, O);  break;
    case 32:  launch_kernel<bfloat16_t, 128, 128, 32, 8>(Q, K, V, O);  break;
    case 64:  launch_kernel<bfloat16_t, 64, 64, 64, 4>(Q, K, V, O);    break;
    case 128: launch_kernel<bfloat16_t, 64, 32, 128, 4>(Q, K, V, O);   break;
    case 256: launch_kernel<bfloat16_t, 32, 32, 256, 2>(Q, K, V, O);   break;
    default:
      throw std::runtime_error("Unsupported headdim for bf16");
    }
  } else {
    TORCH_CHECK(false, "Expected kHalf or kBFloat16, got ", Q.dtype());
  }
}


at::Tensor flash_attn_cuda(const at::Tensor &Q, const at::Tensor &K,
                           const at::Tensor &V) {
  TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
              "Q/K/V must be 4D (B, H, N, D), got Q:", Q.dim(), "D K:", K.dim(), "D V:", V.dim(), "D");
  TORCH_INTERNAL_ASSERT(Q.device().type() == at::DeviceType::CUDA);
  TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
              "Q, K, V must be on the same device");
  TORCH_CHECK(Q.dtype() == K.dtype() && Q.dtype() == V.dtype(),
              "Q, K, V must have the same dtype");
  TORCH_CHECK(Q.dtype() == at::kHalf || Q.dtype() == at::kBFloat16,
              "Expected kHalf or kBFloat16, got ", Q.dtype());

  const at::cuda::OptionalCUDAGuard device_guard(Q.device());

  const int B = Q.size(0);
  const int H = Q.size(1);
  const int N_QO = Q.size(2);
  const int D = Q.size(3);
  const int N_KV = K.size(2);

  TORCH_CHECK(K.size(3) == D, "K head dim mismatch: ", K.size(3), " vs ", D);
  TORCH_CHECK(V.size(3) == D, "V head dim mismatch: ", V.size(3), " vs ", D);
  TORCH_CHECK(V.size(2) == N_KV, "V seq len mismatch: ", V.size(2), " vs ", N_KV);

  at::Tensor q = Q.contiguous(), k = K.contiguous(), v = V.contiguous();
  at::Tensor o = torch::empty_like(q);
  flash_attn_cute(q, k, v, o);

  return o;
}

TORCH_LIBRARY_FRAGMENT(extension_cpp, m) {
  m.def("flash_attn(Tensor Q, Tensor K, Tensor V) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("flash_attn", &flash_attn_cuda);
}

} 
