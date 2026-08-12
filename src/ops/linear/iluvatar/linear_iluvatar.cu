#include "../../../utils.hpp"
#include "linear_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// out[i,j] = sum_k(in[i,k] * weight[j,k]) + bias[j]
template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                              size_t batch_size, size_t in_features, size_t out_features) {
    size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; // batch dim
    size_t col = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y; // out_features dim

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (size_t k = 0; k < in_features; k++) {
            float a, b;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                a = __bfloat162float(in[row * in_features + k]);
                b = __bfloat162float(weight[col * in_features + k]);
            } else if constexpr (std::is_same_v<T, __half>) {
                a = __half2float(in[row * in_features + k]);
                b = __half2float(weight[col * in_features + k]);
            } else {
                a = in[row * in_features + k];
                b = weight[col * in_features + k];
            }
            sum += a * b;
        }
        if (bias != nullptr) {
            float bv;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                bv = __bfloat162float(bias[col]);
            } else if constexpr (std::is_same_v<T, __half>) {
                bv = __half2float(bias[col]);
            } else {
                bv = bias[col];
            }
            sum += bv;
        }
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            out[row * out_features + col] = __float2bfloat16(sum);
        } else if constexpr (std::is_same_v<T, __half>) {
            out[row * out_features + col] = __float2half(sum);
        } else {
            out[row * out_features + col] = sum;
        }
    }
}

namespace llaisys::ops::iluvatar {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((batch_size + block.x - 1) / block.x),
              static_cast<unsigned>((out_features + block.y - 1) / block.y));
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<<<grid, block>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias),
                                       batch_size, in_features, out_features);
        return;
    case LLAISYS_DTYPE_BF16:
        linear_kernel<<<grid, block>>>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
                                       reinterpret_cast<const __nv_bfloat16 *>(weight), reinterpret_cast<const __nv_bfloat16 *>(bias),
                                       batch_size, in_features, out_features);
        return;
    case LLAISYS_DTYPE_F16:
        linear_kernel<<<grid, block>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
                                       reinterpret_cast<const __half *>(weight), reinterpret_cast<const __half *>(bias),
                                       batch_size, in_features, out_features);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::iluvatar
