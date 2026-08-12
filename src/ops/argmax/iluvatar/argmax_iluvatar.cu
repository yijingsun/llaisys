#include "../../../utils.hpp"
#include "argmax_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // Single block reduction for argmax
    __shared__ T svals[256];
    __shared__ int64_t sidxs[256];

    int tid = threadIdx.x;
    T local_max;
    int64_t local_idx = 0;

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        local_max = __float2bfloat16(-FLT_MAX);
    } else if constexpr (std::is_same_v<T, __half>) {
        local_max = __float2half(-FLT_MAX);
    } else {
        local_max = -FLT_MAX;
    }

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        T v = vals[i];
        float fv;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            fv = __bfloat162float(v);
        } else if constexpr (std::is_same_v<T, __half>) {
            fv = __half2float(v);
        } else {
            fv = v;
        }
        float flm;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            flm = __bfloat162float(local_max);
        } else if constexpr (std::is_same_v<T, __half>) {
            flm = __half2float(local_max);
        } else {
            flm = local_max;
        }
        if (fv > flm) {
            local_max = v;
            local_idx = static_cast<int64_t>(i);
        }
    }

    svals[tid] = local_max;
    sidxs[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float fv, fsv;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                fv = __bfloat162float(svals[tid]);
                fsv = __bfloat162float(svals[tid + s]);
            } else if constexpr (std::is_same_v<T, __half>) {
                fv = __half2float(svals[tid]);
                fsv = __half2float(svals[tid + s]);
            } else {
                fv = svals[tid];
                fsv = svals[tid + s];
            }
            if (fsv > fv) {
                svals[tid] = svals[tid + s];
                sidxs[tid] = sidxs[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_val[0] = svals[0];
        max_idx[0] = sidxs[0];
    }
}

namespace llaisys::ops::iluvatar {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    constexpr int block_size = 256;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<<<1, block_size>>>(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                                         reinterpret_cast<const float *>(vals), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<<<1, block_size>>>(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<__nv_bfloat16 *>(max_val),
                                         reinterpret_cast<const __nv_bfloat16 *>(vals), numel);
        return;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<<<1, block_size>>>(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<__half *>(max_val),
                                         reinterpret_cast<const __half *>(vals), numel);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::iluvatar
