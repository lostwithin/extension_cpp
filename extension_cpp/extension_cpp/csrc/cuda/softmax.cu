#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>   // OptionalCUDAGuard(多卡)

#define WARP_SIZE 32
#define FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

namespace extension_cpp{
    template<const int kWarpSize>
    __device__ __forceinline__ float warp_max_reduce(float val)
    {
        #pragma unroll
        for(int k=kWarpSize>>1;k>0;k>>=1){
            val =fmaxf(val, __shfl_xor_sync(0xffffffff, val, k));
        }
        return val;
    }
    template<const int BLOCKSIZE>
    __device__ __forceinline__ float block_max_reduce(float val)
    {
        int tid = threadIdx.x;
        int warp = tid / WARP_SIZE;
        int lane = tid % WARP_SIZE;
        constexpr int nums_warp = (BLOCKSIZE + WARP_SIZE - 1) / WARP_SIZE;
        __shared__ float smem[nums_warp];
        val = warp_max_reduce<WARP_SIZE>(val);
        if(lane == 0) smem[warp] = val;
        __syncthreads();
        val = (lane < nums_warp) ? smem[lane] : -INFINITY;
        val = warp_max_reduce<WARP_SIZE>(val);
        return val;
    }
    template<const int kWarpSize>
    __device__ __forceinline__ float warp_reduce(float val)
    {
        #pragma unroll
        for(int k=kWarpSize>>1;k>0;k>>=1){
            val += __shfl_xor_sync(0xffffffff, val, k);
        }
        return val;
    }
    template<const int BLOCKSIZE>
    __device__ __forceinline__ float block_reduce(float val)
    {
        int tid = threadIdx.x;
        int warp = tid / WARP_SIZE;
        int lane = tid % WARP_SIZE;
        constexpr int nums_warp = (BLOCKSIZE + WARP_SIZE - 1) / WARP_SIZE;
        __shared__ float smem[nums_warp];
        val = warp_reduce<WARP_SIZE>(val);
        if(lane == 0) smem[warp] = val;
        __syncthreads();
        val = (lane < nums_warp) ? smem[lane] : 0.0f;
        val = warp_reduce<WARP_SIZE>(val);
        return val;
    }
    // K % 4 == 0 且 K <= 4096 快速路径:每线程一次 float4 加载持有 4 个元素,
    // 行数据驻留寄存器,1R+1W。线程数为 K/4:K=1024 时 256 线程,一个 SM 可驻多个
    // block 互相掩盖归约屏障的停顿(1024 线程大 block 做不到这点)。
    template<const int BLOCKSIZE>
    __global__ void softmax_f4(const float *x, float *y, int k)
    {
        int bx = blockIdx.x;
        int tid = threadIdx.x;
        const float *x_row = x + (size_t)bx * k;
        float *y_row = y + (size_t)bx * k;
        bool active = tid < (k >> 2);
        float4 v = active ? FLOAT4(x_row[tid << 2])
                          : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        float m = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        m = block_max_reduce<BLOCKSIZE>(m);
        float4 e;
        e.x = active ? __expf(v.x - m) : 0.0f;
        e.y = active ? __expf(v.y - m) : 0.0f;
        e.z = active ? __expf(v.z - m) : 0.0f;
        e.w = active ? __expf(v.w - m) : 0.0f;
        float e_sum = block_reduce<BLOCKSIZE>(e.x + e.y + e.z + e.w);
        if(active){
            float inv = 1.0f / e_sum;
            e.x *= inv; e.y *= inv; e.z *= inv; e.w *= inv;
            reinterpret_cast<float4*>(&y_row[tid << 2])[0] = e;
        }
    }

    // 256 < K <= 4096 且非 4 倍数:线程粗化持久化版。每线程在寄存器数组持有 ITEMS 个
    // 标量元素(无对齐要求),1R+1W 且 exp 不重算。i = tid + j*BLOCKSIZE 保证每轮内
    // 线程访存连续(合并);越界元素载入 -INF,exp 后变 0,天然不污染归约。
    template<const int BLOCKSIZE, const int ITEMS>
    __global__ void softmax_persist(const float *x, float *y, int k)
    {
        int bx = blockIdx.x;
        int tid = threadIdx.x;
        const float *x_row = x + (size_t)bx * k;
        float *y_row = y + (size_t)bx * k;
        float v[ITEMS];
        float m_local = -INFINITY;
        #pragma unroll
        for(int j=0; j<ITEMS; j++){
            int i = tid + j * BLOCKSIZE;
            v[j] = (i < k) ? x_row[i] : -INFINITY;
            m_local = fmaxf(m_local, v[j]);
        }
        float m = block_max_reduce<BLOCKSIZE>(m_local);
        float s_local = 0.0f;
        #pragma unroll
        for(int j=0; j<ITEMS; j++){
            v[j] = __expf(v[j] - m);
            s_local += v[j];
        }
        float s = block_reduce<BLOCKSIZE>(s_local);
        float inv = 1.0f / s;
        #pragma unroll
        for(int j=0; j<ITEMS; j++){
            int i = tid + j * BLOCKSIZE;
            if(i < k) y_row[i] = v[j] * inv;
        }
    }

    // K <= 256 快速路径(K 非 4 倍数时兜底):一个线程持有一个元素,行数据全程驻留
    // 寄存器,全局内存恰好 1 读 + 1 写,exp 只算一次。tid >= k 的线程以恒等元参与归约。
    template<const int BLOCKSIZE>
    __global__ void softmax_per_thread(const float *x, float *y, int k)
    {
        int bx = blockIdx.x;
        int tid = threadIdx.x;
        const float *x_row = x + (size_t)bx * k;
        float *y_row = y + (size_t)bx * k;
        float val = (tid < k) ? x_row[tid] : -INFINITY;
        float m = block_max_reduce<BLOCKSIZE>(val);
        float e = (tid < k) ? __expf(val - m) : 0.0f;
        float e_sum = block_reduce<BLOCKSIZE>(e);
        if(tid < k) y_row[tid] = e / e_sum;
    }

    // 4096 < K <= 16256 且 K%4==0:行缓存进动态 smem,DRAM 恰好 1R+1W。
    // 第二遍把 exp 原位写回 smem,第三遍只做缩放,exp 仅算一次。
    // 动态 smem + 归约的静态 smem 合计受限:默认 48KB/block,>48KB 需
    // cudaFuncSetAttribute 解锁(Turing+ 可到 64KB),上限均留出静态部分的余量。
    template<const int BLOCKSIZE>
    __global__ void softmax_smem(const float *x, float *y, int k)
    {
        extern __shared__ float s_row[];
        int bx = blockIdx.x;
        int tid = threadIdx.x;
        const float *x_row = x + (size_t)bx * k;
        float *y_row = y + (size_t)bx * k;
        int k4 = k >> 2;
        float m_val = -INFINITY;
        for(int i=tid; i<k4; i+=BLOCKSIZE){
            float4 v = FLOAT4(x_row[i << 2]);
            reinterpret_cast<float4*>(s_row)[i] = v;
            m_val = fmaxf(m_val, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
        }
        m_val = block_max_reduce<BLOCKSIZE>(m_val);
        float e_sum = 0.0f;
        for(int i=tid; i<k4; i+=BLOCKSIZE){
            float4 v = reinterpret_cast<float4*>(s_row)[i];
            v.x = __expf(v.x - m_val);
            v.y = __expf(v.y - m_val);
            v.z = __expf(v.z - m_val);
            v.w = __expf(v.w - m_val);
            e_sum += v.x + v.y + v.z + v.w;
            reinterpret_cast<float4*>(s_row)[i] = v;
        }
        e_sum = block_reduce<BLOCKSIZE>(e_sum);
        float inv = 1.0f / e_sum;
        for(int i=tid; i<k4; i+=BLOCKSIZE){
            float4 v = reinterpret_cast<float4*>(s_row)[i];
            v.x *= inv; v.y *= inv; v.z *= inv; v.w *= inv;
            reinterpret_cast<float4*>(&y_row[i << 2])[0] = v;
        }
    }

    // online 的 float4 版(K%4==0 且超出 smem 容量):向量化加载/写回,
    // 块内先对 4 元素取 max 再做一次在线更新,每次迭代仍只有一次缩放修正。
    template<const int BLOCKSIZE>
    __global__ void softmax_online_f4(const float *x, float *y, int k)
    {
        int bx = blockIdx.x;
        int tid = threadIdx.x;
        const float *x_row = x + (size_t)bx * k;
        float *y_row = y + (size_t)bx * k;
        int k4 = k >> 2;
        float m = -INFINITY, d = 0.0f;
        for(int i=tid; i<k4; i+=BLOCKSIZE){
            float4 v = FLOAT4(x_row[i << 2]);
            float m_new = fmaxf(m, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
            d = d * __expf(m - m_new)
              + __expf(v.x - m_new) + __expf(v.y - m_new)
              + __expf(v.z - m_new) + __expf(v.w - m_new);
            m = m_new;
        }
        float m_all = block_max_reduce<BLOCKSIZE>(m);
        float d_all = block_reduce<BLOCKSIZE>(d * __expf(m - m_all));
        float inv = 1.0f / d_all;
        for(int i=tid; i<k4; i+=BLOCKSIZE){
            float4 v = FLOAT4(x_row[i << 2]);
            v.x = __expf(v.x - m_all) * inv;
            v.y = __expf(v.y - m_all) * inv;
            v.z = __expf(v.z - m_all) * inv;
            v.w = __expf(v.w - m_all) * inv;
            reinterpret_cast<float4*>(&y_row[i << 2])[0] = v;
        }  
    }

    // 通用 fallback(任意 K):online softmax,一遍同时得到 max 和 sum,3 遍降为 2 遍
    // (DRAM 2R+1W)。线程本地在线累积 (m, d),块级先归约 max,再把各线程 d 缩放后求和。
    template<const int BLOCKSIZE>
    __global__ void softmax_online(const float *x, float *y, int k)
    {
        int bx = blockIdx.x;
        int tid = threadIdx.x;
        const float *x_row = x + (size_t)bx * k;
        float *y_row = y + (size_t)bx * k;
        float m = -INFINITY, d = 0.0f;
        for(int i=tid; i<k; i+=BLOCKSIZE){
            float v = x_row[i];
            float m_new = fmaxf(m, v);
            d = d * __expf(m - m_new) + __expf(v - m_new);
            m = m_new;
        }
        float m_all = block_max_reduce<BLOCKSIZE>(m);
        float d_all = block_reduce<BLOCKSIZE>(d * __expf(m - m_all));
        float inv = 1.0f / d_all;
        for(int i=tid; i<k; i+=BLOCKSIZE){
            y_row[i] = __expf(x_row[i] - m_all) * inv;
        }
    }

    at::Tensor softmax_cuda(const at::Tensor& x){
        TORCH_CHECK(x.dim() == 2);
        TORCH_CHECK(x.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
        const at::cuda::OptionalCUDAGuard device_guard(x.device());
        at::Tensor x_contig = x.contiguous();
        at::Tensor y = at::empty_like(x_contig);
        int64_t N = x_contig.size(0);
        int64_t K = x_contig.size(1);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        const float *xp = x_contig.data_ptr<float>();
        float       *yp = y.data_ptr<float>();
        // 模板 BLOCKSIZE 必须等于实际 blockDim(归约里 smem 大小依赖它),按 2 的幂分发
        #define LAUNCH_F4(BS) softmax_f4<BS><<<(unsigned)N, BS, 0, stream>>>(xp, yp, (int)K)
        #define LAUNCH_PT(BS) softmax_per_thread<BS><<<(unsigned)N, BS, 0, stream>>>(xp, yp, (int)K)
        if (K % 4 == 0 && K <= 4096) {
            int64_t k4 = K / 4;
            if (k4 <= 32)       LAUNCH_F4(32);
            else if (k4 <= 64)  LAUNCH_F4(64);
            else if (k4 <= 128) LAUNCH_F4(128);
            else if (k4 <= 256) LAUNCH_F4(256);
            else if (k4 <= 512) LAUNCH_F4(512);
            else                LAUNCH_F4(1024);
        }
        else if (K <= 32)   LAUNCH_PT(32);
        else if (K <= 64)   LAUNCH_PT(64);
        else if (K <= 128)  LAUNCH_PT(128);
        else if (K <= 256)  LAUNCH_PT(256);
        else if (K <= 1024)
            softmax_persist<256, 4><<<(unsigned)N, 256, 0, stream>>>(xp, yp, (int)K);
        else if (K <= 2048)
            softmax_persist<256, 8><<<(unsigned)N, 256, 0, stream>>>(xp, yp, (int)K);
        else if (K <= 4096)
            softmax_persist<256, 16><<<(unsigned)N, 256, 0, stream>>>(xp, yp, (int)K);
        else {
            bool smem_ok = K % 4 == 0 && K <= 16256;
            if (smem_ok && K > 12160)   // 超出默认 48KB,尝试解锁;老卡(<sm70)会失败则走兜底
                smem_ok = cudaFuncSetAttribute(softmax_smem<512>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)(K * 4)) == cudaSuccess;
            if (smem_ok)
                softmax_smem<512><<<(unsigned)N, 512, (size_t)K * 4, stream>>>(xp, yp, (int)K);
            else {
                cudaGetLastError();   // 清掉 attribute 失败留下的错误码
                // 大 K 用 1024 线程大 block:SM 驻留行数少,两遍间的重读尽量留在 L2;
                // 此时每线程元素足够多,大 block 的归约屏障开销可被访存掩盖。
                if (K > 16256) {
                    if (K % 4 == 0)
                        softmax_online_f4<1024><<<(unsigned)N, 1024, 0, stream>>>(xp, yp, (int)K);
                    else
                        softmax_online<1024><<<(unsigned)N, 1024, 0, stream>>>(xp, yp, (int)K);
                }
                else if (K % 4 == 0)
                    softmax_online_f4<256><<<(unsigned)N, 256, 0, stream>>>(xp, yp, (int)K);
                else
                    softmax_online<256><<<(unsigned)N, 256, 0, stream>>>(xp, yp, (int)K);
            }
        }
        #undef LAUNCH_F4
        #undef LAUNCH_PT
        return y;
    }
    TORCH_LIBRARY_FRAGMENT(extension_cpp, m){
        m.def("softmax(Tensor x) -> Tensor");
    }
    TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m){
        m.impl("softmax", &softmax_cuda);
    }
}