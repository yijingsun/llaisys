#include "../../../utils.hpp"
#include "flash_attention_nvidia.cuh"
#include <cassert>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CEIL(a, b) (((a) + (b) - 1) / (b))

__device__ float warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = max(value, other);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ float warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, value, offset);
        value = value + other;
    }
    return __shfl_sync(0xffffffff, value, 0);
}

constexpr int block_size = 256;
const int TILE_Q = 8;
constexpr int Bc = 32;
template <typename T>
__global__ void flash_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                                      size_t d, size_t dv, float scale) {

    size_t tile_start = blockIdx.x * TILE_Q;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5; // = row_in_tile
    size_t i = tile_start + warp_id; // 可能 ≥ seqlen（tile 不满）
    size_t h = blockIdx.y;
    size_t group = nhead / nkvhead;
    size_t kvh = h / group;
    assert(total_len == seqlen); // 只服务 prefill，见文件头
    size_t limit = i;
    // 协作搬运+__syncthreads() 要求 block 内循环次数一致
    size_t tile_max_limit = min(tile_start + TILE_Q - 1, seqlen - 1);

    extern __shared__ float smem[];
    float *K_chunk = smem;
    float *V_chunk = smem + Bc * d;
    float m = -INFINITY;
    float l = 0.0f;
    float acc[4] = {0.0};
    for (int j = 0; j <= tile_max_limit; j += Bc) {
        // 协作搬运 K/V 进 shared memory，tile 内所有行共享。
        for (size_t flat = threadIdx.x; flat < Bc * d; flat += blockDim.x) {
            size_t dim = flat % d;
            size_t row_in_chunk = flat / d;
            if (j + row_in_chunk <= tile_max_limit) {
                K_chunk[flat] = k[(j + row_in_chunk) * nkvhead * d + kvh * d + dim];
                V_chunk[flat] = v[(j + row_in_chunk) * nkvhead * dv + kvh * dv + dim];
            }
        }
        __syncthreads();

        if (i < seqlen) {
            float score = 0.0;
            for (size_t dim = 0; dim < d; dim++) {
                score += float(q[i * nhead * d + h * d + dim]) * K_chunk[lane * d + dim];
            }
            score *= scale;
            if (j + lane > limit) {
                score = -INFINITY;
            }
            float chunk_max = warp_reduce_max(score);
            if (chunk_max > m) {
                float suofang = exp(m - chunk_max);
                l *= suofang;
                for (int t = 0; t < 4; t++) {
                    acc[t] *= suofang;
                }
                m = chunk_max;
            }
            float p = exp(score - m);
            float sum_exp = warp_reduce_sum(p);
            l += sum_exp;
            for (int src = 0; src < 32; src++) {
                float p_src = __shfl_sync(0xffffffff, p, src);
                for (int t = 0; t < 4; t++) {
                    acc[t] += p_src * V_chunk[src * dv + t * 32 + lane];
                }
            }
        }
        __syncthreads();
    }
    if (i < seqlen) {
        for (size_t t = 0; t < 4; t++) {
            attn_val[i * nhead * dv + h * dv + t * 32 + lane] = acc[t] / l;
        }
    }
}

template <typename T>
void launch_flash_attention(T *attn_val, const T *q, const T *k, const T *v,
                           size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                           size_t d, size_t dv, float scale) {
    unsigned int gridx = CEIL(seqlen, TILE_Q);

    dim3 grid(static_cast<unsigned int>(gridx), static_cast<unsigned int>(nhead));
    size_t shared_bytes = Bc * (d + dv) * sizeof(float);

    flash_attention_kernel<<<grid, block_size, shared_bytes>>>(
        attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
}

namespace llaisys::ops::nvidia {

void flash_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     float scale, llaisysDataType_t type,
                     size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_flash_attention(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_flash_attention(
            reinterpret_cast<__nv_bfloat16 *>(attn_val),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    case LLAISYS_DTYPE_F16:
        launch_flash_attention(
            reinterpret_cast<__half *>(attn_val),
            reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k),
            reinterpret_cast<const __half *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
