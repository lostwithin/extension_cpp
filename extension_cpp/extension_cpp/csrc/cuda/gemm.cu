#include <ATen/Operators.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace extension_cpp {

template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC,
          typename S2GCopyC, const bool BlockSwizzle>
__global__ void hgemm_mma_stages_block_swizzle_tn_cute_kernel(T *Aptr, T *Bptr,
                                                              T *Dptr, int m,
                                                              int n, int k) {
  using namespace cute;
  // Initilize shared memory
  extern __shared__ T shm_data[];

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  // Initilize thread block
  int idx = threadIdx.x;
  // BlockSwizzle 0/1 control use block swizzle or not.
  int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  int iy = blockIdx.y;

  if (iy * BM >= m || ix * BN >= n)
    return;

  // use Tensor notation to represent device pointer + dimension
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  // slice the tensor to small one which is used for current thread block.
  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}),
                         make_coord(iy, _)); // (BM, BK, num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}),
                         make_coord(ix, _)); // (BN, BK, num_tile_k)
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}),
                         make_coord(iy, ix)); // (BM, BN)

  // shared memory
  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{}); // (BM, BK, kStage)
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{}); // (BN, BK, kStage)

  // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto tCgD = thr_mma.partition_C(gD); // (MMA,MMA_M, MMA_N)

  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
  clear(tCrD);

  // from global memory to shared memory
  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy =
      g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, num_tile_k)
  auto tAsA_copy =
      g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy =
      g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, num_tile_k)
  auto tBsB_copy =
      g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

  // from shared memory to register, use tiled_mma to generate tiled_copy
  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);     // (CPY, CPY_M, CPY_K, kStage)
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

  /* PREFETCH */
  // submit kStage - 1 tile
  // gmem -> shm
  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();

    ++itile_to_read;
    ++ismem_write;
  }

  // wait one submitted gmem->smem done
  cp_async_wait<kStage - 2>();
  __syncthreads();

  int ik = 0;
  // smem -> reg
  // tAsA: (CPY, CPY_M, CPY_K, kStage) tCrA_view: (CPY, CPY_M, CPY_K)
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

  // loop over k: i. load tile, ii. mma
  int ntile = k / BK;
#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA); // (MMA, MMA_M, MMA_K)

#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;

      if (ik == nk - 1) {
        cp_async_wait<kStage - 2>();
        __syncthreads();

        ismem_read = (ismem_read + 1) % kStage;
      }

      // shm -> reg s[itile][ik + 1] -> r[ik + 1]
      // tAsA: (CPY, CPY_M, CPY_K, kStage), tCrA_view: (CPY, CPY_M, CPY_K)
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      // tBsB: (CPY, CPY_M, CPY_K, kStage), tCrB_view: (CPY, CPY_M, CPY_K)
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));

      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }

        cp_async_fence();
      }

      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    } // for ik
  }

  // use less shared memory as a scratchpad tile to use large wide instuction
  // Dreg -> shm -> reg -> global
  // C epilogue scratchpad: reuse whichever of A/B shmem pipe has room.
  // Large tile (128×256): A pipe = 4Ki ≥ C = 4Ki → use A.
  // Small tile (64×128):   A pipe = 2Ki < C = 4Ki → use B.
  constexpr bool kUseAforC =
      size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{});
  static_assert(kUseAforC ||
                    size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) >=
                        size(SmemLayoutC{}),
                "Neither shmem pipe fits C scratchpad");
  auto sC = make_tensor(
      make_smem_ptr(
          (kUseAforC ? sA(_, _, ismem_read).data().get()
                     : sB(_, _, ismem_read).data().get())),
      SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);  // (CPY, CPY_M, CPY_N)
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY_, CPY_MN)
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY_, CPY_MN)

  int step = size<3>(tCsC_r2s); // pipe
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
// reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
      // we add a temp tensor to cope with accumulator and output data type
      // difference
      auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
      cute::copy(tCrC_r2sx(_, i + j), t);

      cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();

#pragma unroll
    // shm -> global
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();
  } // end for
}

// Templated launcher — BM, BN, BK parameterised so we can instantiate
// large-tile (128×256) and small-tile (64×128) variants from the same code.
template <typename T, int BM_V = 128, int BN_V = 256, int BK_V = 32,
          const int Stages = 2, const bool BlockSwizzle = false>
void launch_hgemm_mma_stages_block_swizzle_tn_cute(T *a, T *b, T *c, int M,
                                                   int N, int K,
                                                   int swizzle_stride,
                                                   cudaStream_t stream) {
  using namespace cute;

  auto BM = Int<BM_V>{};
  auto BN = Int<BN_V>{};
  auto BK = Int<BK_V>{};
  auto KStage = Int<Stages>{};       // default 2
  auto kSmemLayoutCBatch = Int<4>{}; // namely, stages.

  // Define the smem layouts, Swizzle<3, 3, 3> and
  // Swizzle<2, 3, 3> will get the same results.
  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<BK>{}),
                                      make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{}))); // (m,n) -> smem_idx

  // mma
  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  static constexpr int kMmaEURepeatM = 2; // MMA repeat 2 times across M
  static constexpr int kMmaEURepeatN = 2; // MMA repeat 2 times across N
  static constexpr int kMmaEURepeatK = 1; // MMA no repeat across K

  using mma_atom_shape = mma_traits::Shape_MNK; // M,N,K 16,8,16
  static constexpr int kMmaPM =
      1 * kMmaEURepeatM * get<0>(mma_atom_shape{}); // 1*2*16=32
  static constexpr int kMmaPN =
      2 * kMmaEURepeatN * get<1>(mma_atom_shape{}); // 2*2*8 =32
  static constexpr int kMmaPK =
      1 * kMmaEURepeatK * get<2>(mma_atom_shape{}); // 1*1*16=16
  // TiledMMA, more threads, MMAThrLayout(2,2,1), 4 MMA = 4 warps = 32x4
  // threads.
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  // TiledMMA, more values, Permutations(32,32,16)
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  // copy from global memory to shared memory
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  // 32x4 threads, each thread load 1x8 values (128 bits) once.
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}), // Thr layout 32x4 k-major
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{})))); // Val layout 1x8
  using G2SCopyB = G2SCopyA;

  // copy from shared memory to register
  // use mma tiled ,so no tiled here
  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  // epilogue: register to global via shared memory
  // Swizzle<3, 4, 3>: swizzle bits [4..6], so a 128-bit (16-byte) access — whose
  // address differs only in the low 4 bits — always stays within one bank and
  // cannot self-conflict. This is the standard swizzle for uint128_t (vector)
  // shared-memory access, and it removes the residual ~0.3% bank conflicts that
  // the ldmatrix-oriented Swizzle<3,3,3> left on the epilogue s2g read.
  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 4, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), // 32*32
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  // kSmemLayoutCBatch=4, 32x32x4=4096 values=8192 bytes
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                        size(SmemLayoutC{}) ||
                    size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) >=
                        size(SmemLayoutC{}),
                "Neither A nor B shmem pipe has room for C scratchpad");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;
  // NOTE: Apply thread block swizzle across N dim.
  int BZ = BlockSwizzle ? (N + (swizzle_stride)-1) / (swizzle_stride) : 1;
  BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY, BZ);

  // C_shm is shared with A_shm and B_shm; we don't allocate new smem for C_shm.
  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  int shm_size = kShmSize;

  cudaFuncSetAttribute(
      hgemm_mma_stages_block_swizzle_tn_cute_kernel<
          T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA,
          SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC,
          S2GCopyAtomC, S2GCopyC, BlockSwizzle>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);

  hgemm_mma_stages_block_swizzle_tn_cute_kernel<
      T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
      SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC,
      S2GCopyC, BlockSwizzle><<<grid, block, shm_size, stream>>>(a, b, c, M, N,
                                                                 K);
}

// ── mygemm (large tile, auto-dispatch) ──────────────────────────────
// When the grid is large enough to fill the GPU (> 4 full waves) the
// 128×256 tile gives best data reuse.  Otherwise we fall back to the
// small-tile (64×128) kernel for better occupancy.

static at::Tensor mygemm_cuda_impl(const at::Tensor &a, const at::Tensor &b,
                                   bool force_small_tile) {
  TORCH_CHECK(a.dim() == 2, "a must be 2D, got ", a.dim(), "D");
  TORCH_CHECK(b.dim() == 2, "b must be 2D, got ", b.dim(), "D");
  TORCH_CHECK(a.dtype() == at::kHalf, "a must be float16");
  TORCH_CHECK(b.dtype() == at::kHalf, "b must be float16");
  TORCH_CHECK(a.size(1) == b.size(1),
              "a and b must share the K dim: a is (M, K)=", a.sizes(),
              ", b is (N, K)=", b.sizes());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();

  const int M = a_contig.size(0);
  const int K = a_contig.size(1);
  const int N = b_contig.size(0);

  // Tile sizes
  constexpr int BM_large = 128, BN_large = 256;
  constexpr int BM_small = 64,  BN_small = 128;
  constexpr int BK = 32;

  int BM, BN;

  if (force_small_tile) {
    BM = BM_small; BN = BN_small;
  } else {
    // Auto-dispatch: use small tile when the large-tile grid would
    // under-utilise the GPU (< 6 full waves on 24 SMs with ~2 blocks/SM).
    int bx_large = (N + BN_large - 1) / BN_large;
    int by_large = (M + BM_large - 1) / BM_large;
    int total_blocks_large = bx_large * by_large;
    constexpr int sm_count = 24;          // RTX 4060 Laptop
    constexpr int blocks_per_sm = 2;       // large tile → 2 concurrent blocks
    constexpr int min_blocks = sm_count * 2;  // 2 full waves → switch to small

    if (total_blocks_large < min_blocks) {
      BM = BM_small; BN = BN_small;
    } else {
      BM = BM_large; BN = BN_large;
    }
  }

  TORCH_CHECK(M % BM == 0, "M (", M, ") must be multiple of ", BM);
  TORCH_CHECK(N % BN == 0, "N (", N, ") must be multiple of ", BN);
  TORCH_CHECK(K % BK == 0, "K (", K, ") must be multiple of ", BK);

  at::Tensor c = at::empty({M, N}, a_contig.options());

  auto *a_ptr = reinterpret_cast<cute::half_t *>(a_contig.data_ptr<at::Half>());
  auto *b_ptr = reinterpret_cast<cute::half_t *>(b_contig.data_ptr<at::Half>());
  auto *c_ptr = reinterpret_cast<cute::half_t *>(c.data_ptr<at::Half>());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (BM == BM_large) {
    launch_hgemm_mma_stages_block_swizzle_tn_cute<
        cute::half_t, BM_large, BN_large, BK, 2, true>(
        a_ptr, b_ptr, c_ptr, M, N, K, /*swizzle_stride=*/2048, stream);
  } else {
    // Small tile: no block swizzle (would hurt occupancy further)
    launch_hgemm_mma_stages_block_swizzle_tn_cute<
        cute::half_t, BM_small, BN_small, BK, 2, /*BlockSwizzle=*/false>(
        a_ptr, b_ptr, c_ptr, M, N, K, /*swizzle_stride=*/2048, stream);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return c;
}

// Public ops
at::Tensor mygemm_cuda(const at::Tensor &a, const at::Tensor &b) {
  return mygemm_cuda_impl(a, b, /*force_small_tile=*/false);
}

at::Tensor mygemm_small_cuda(const at::Tensor &a, const at::Tensor &b) {
  return mygemm_cuda_impl(a, b, /*force_small_tile=*/true);
}

// Schema registration
TORCH_LIBRARY_FRAGMENT(extension_cpp, m) {
  m.def("mygemm(Tensor a, Tensor b) -> Tensor");
  m.def("mygemm_small(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("mygemm", &mygemm_cuda);
  m.impl("mygemm_small", &mygemm_small_cuda);
}

} // namespace extension_cpp
