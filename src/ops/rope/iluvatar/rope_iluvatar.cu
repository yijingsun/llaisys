#include "../../../utils.hpp"
#include "rope_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// in/out: [seqlen, nhead, d], pos_ids: [seqlen]
// For each (seq, head, j) where j < d/2:
//   angle = pos_ids[seq] / theta^(2*j/d)
//   out[seq,head,j] = in[seq,head,j]*cos(angle) - in[seq,head,j+d/2]*sin(angle)
//   out[seq,head,j+d/2] = in[seq,head,j]*sin(angle) + in[seq,head,j+d/2]*cos(angle)
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, float theta,
                            size_t seqlen, size_t nhead, size_t d) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t half_d = d / 2;
    size_t total = seqlen * nhead * half_d;
    if (tid >= total) return;

    size_t seq = tid / (nhead * half_d);
    size_t rem = tid % (nhead * half_d);
    size_t head = rem / half_d;
    size_t j = rem % half_d;

    size_t idx = seq * nhead * d + head * d + j;
    float angle = static_cast<float>(pos_ids[seq]) / powf(theta, 2.0f * j / d);
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float a, b;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        a = __bfloat162float(in[idx]);
        b = __bfloat162float(in[idx + half_d]);
    } else if constexpr (std::is_same_v<T, __half>) {
        a = __half2float(in[idx]);
        b = __half2float(in[idx + half_d]);
    } else {
        a = in[idx];
        b = in[idx + half_d];
    }

    float out_a = a * cos_a - b * sin_a;
    float out_b = a * sin_a + b * cos_a;

    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        out[idx] = __float2bfloat16(out_a);
        out[idx + half_d] = __float2bfloat16(out_b);
    } else if constexpr (std::is_same_v<T, __half>) {
        out[idx] = __float2half(out_a);
        out[idx + half_d] = __float2half(out_b);
    } else {
        out[idx] = out_a;
        out[idx + half_d] = out_b;
    }
}

namespace llaisys::ops::iluvatar {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
    size_t half_d = d / 2;
    size_t total = seqlen * nhead * half_d;
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((total + block_size - 1) / block_size);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<grid_size, block_size>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                               reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
        return;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<grid_size, block_size>>>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
                                               reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
        return;
    case LLAISYS_DTYPE_F16:
        rope_kernel<<<grid_size, block_size>>>(reinterpret_cast<__half *>(out), reinterpret_cast<const __half *>(in),
                                               reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::iluvatar
