#include "../../../utils.hpp"
#include "swiglu_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// out = silu(gate) * up = gate * sigmoid(gate) * up
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float g, u;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            g = __bfloat162float(gate[idx]);
            u = __bfloat162float(up[idx]);
        } else if constexpr (std::is_same_v<T, __half>) {
            g = __half2float(gate[idx]);
            u = __half2float(up[idx]);
        } else {
            g = gate[idx];
            u = up[idx];
        }
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        float result = g * sigmoid_g * u;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            out[idx] = __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, __half>) {
            out[idx] = __float2half(result);
        } else {
            out[idx] = result;
        }
    }
}

namespace llaisys::ops::iluvatar {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<<<grid_size, block_size>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                                                 reinterpret_cast<const float *>(up), numel);
        return;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<<<grid_size, block_size>>>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(gate),
                                                 reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        return;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<<<grid_size, block_size>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(gate),
                                                 reinterpret_cast<const __half *>(up), numel);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::iluvatar
