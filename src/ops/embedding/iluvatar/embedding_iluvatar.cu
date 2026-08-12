#include "../../../utils.hpp"
#include "embedding_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight, size_t line_num, size_t embedding_dim) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = line_num * embedding_dim;
    if (tid < total) {
        size_t row = tid / embedding_dim;
        size_t col = tid % embedding_dim;
        int64_t idx = index[row];
        out[tid] = weight[idx * embedding_dim + col];
    }
}

namespace llaisys::ops::iluvatar {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t line_num, size_t vocab_size, size_t embedding_dim) {
    size_t total = line_num * embedding_dim;
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<<<grid_size, block_size>>>(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index),
                                                    reinterpret_cast<const float *>(weight), line_num, embedding_dim);
        return;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<<<grid_size, block_size>>>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const int64_t *>(index),
                                                    reinterpret_cast<const __nv_bfloat16 *>(weight), line_num, embedding_dim);
        return;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<<<grid_size, block_size>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const int64_t *>(index),
                                                    reinterpret_cast<const __half *>(weight), line_num, embedding_dim);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::iluvatar
