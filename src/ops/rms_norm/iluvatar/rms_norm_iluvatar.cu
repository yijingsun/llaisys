#include "../../../utils.hpp"
#include "rms_norm_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// Each block handles one row (one sample in the batch)
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, float eps, size_t batch_size, size_t in_features) {
    size_t row = blockIdx.x;
    if (row >= batch_size) return;

    const T *row_in = in + row * in_features;
    T *row_out = out + row * in_features;

    // Compute sum of squares using shared memory reduction
    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < in_features; i += blockDim.x) {
        float v;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            v = __bfloat162float(row_in[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            v = __half2float(row_in[i]);
        } else {
            v = row_in[i];
        }
        local_sum += v * v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / in_features + eps);

    for (size_t i = threadIdx.x; i < in_features; i += blockDim.x) {
        float v;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            v = __bfloat162float(row_in[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            v = __half2float(row_in[i]);
        } else {
            v = row_in[i];
        }
        float w;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            w = __bfloat162float(weight[i]);
        } else if constexpr (std::is_same_v<T, __half>) {
            w = __half2float(weight[i]);
        } else {
            w = weight[i];
        }
        float result = v / rms * w;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            row_out[i] = __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, __half>) {
            row_out[i] = __float2half(result);
        } else {
            row_out[i] = result;
        }
    }
}

namespace llaisys::ops::iluvatar {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t batch_size, size_t in_features) {
    int block_size = 256;
    size_t shared_mem = block_size * sizeof(float);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<batch_size, block_size, shared_mem>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight), eps, batch_size, in_features);
        return;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__nv_bfloat16><<<batch_size, block_size, shared_mem>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight), eps, batch_size, in_features);
        return;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<__half><<<batch_size, block_size, shared_mem>>>(
            reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
            reinterpret_cast<const __half *>(weight), eps, batch_size, in_features);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::iluvatar
