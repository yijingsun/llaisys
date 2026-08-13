#include "../../../utils.hpp"
#include "flash_attention_moore.cuh"

#include <algorithm>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM = 128;
constexpr int TILE_K = 32;

__device__ float warp_max(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = max(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ float warp_sum(float value) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

template <typename T>
__global__ void flash_attention_decode_splitkv_phase1_kernel(
    float *partial_m, float *partial_l, float *partial_acc,
    const T *q, const T *k, const T *v,
    size_t total_len, size_t nhead, size_t nkvhead, float scale,
    size_t split_size, size_t num_splits) {
    const int lane = threadIdx.x;
    const size_t head = blockIdx.x;
    const size_t split_idx = blockIdx.y;
    const size_t kv_head = head / (nhead / nkvhead);

    size_t split_start = split_idx * split_size;
    size_t split_end = min(split_start + split_size, total_len);
    if (split_start >= total_len) {
        partial_m[head * num_splits + split_idx] = -INFINITY;
        partial_l[head * num_splits + split_idx] = 0;
        for (size_t t = 0; t < 4; t++) {
            partial_acc[(head * num_splits + split_idx) * HEAD_DIM + t * WARP_SIZE + lane] = 0;
        }
        return;
    }
    extern __shared__ float shared[];
    float *k_tile = shared;
    float *v_tile = shared + TILE_K * HEAD_DIM;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t tile_start = split_start; tile_start < split_end; tile_start += TILE_K) {
        const size_t tile_size = min(static_cast<size_t>(TILE_K), split_end - tile_start);

        for (size_t flat = lane; flat < tile_size * HEAD_DIM; flat += WARP_SIZE) {
            const size_t row = flat / HEAD_DIM;
            const size_t dim = flat % HEAD_DIM;
            k_tile[flat] = static_cast<float>(
                k[(tile_start + row) * nkvhead * HEAD_DIM + kv_head * HEAD_DIM + dim]);
            v_tile[flat] = static_cast<float>(
                v[(tile_start + row) * nkvhead * HEAD_DIM + kv_head * HEAD_DIM + dim]);
        }
        __syncwarp();

        float score = -INFINITY;
        if (lane < tile_size) {
            score = 0.0f;
            for (int dim = 0; dim < HEAD_DIM; ++dim) {
                score += static_cast<float>(q[head * HEAD_DIM + dim]) * k_tile[lane * HEAD_DIM + dim];
            }
            score *= scale;
        }

        const float tile_max = warp_max(score);
        const float new_max = max(max_score, tile_max);
        const float old_scale = expf(max_score - new_max);
        const float probability = lane < tile_size ? expf(score - new_max) : 0.0f;

        sum_exp = sum_exp * old_scale + warp_sum(probability);
        for (int item = 0; item < 4; ++item) {
            acc[item] *= old_scale;
        }

        for (int source = 0; source < TILE_K; ++source) {
            const float weight = __shfl_sync(0xffffffff, probability, source);
            if (source < tile_size) {
                for (int item = 0; item < 4; ++item) {
                    acc[item] += weight * v_tile[source * HEAD_DIM + item * WARP_SIZE + lane];
                }
            }
        }
        max_score = new_max;
        __syncwarp();
    }

    partial_m[head * num_splits + split_idx] = max_score;
    partial_l[head * num_splits + split_idx] = sum_exp;
    for (size_t item = 0; item < 4; item++) {
        partial_acc[(head * num_splits + split_idx) * HEAD_DIM + item * WARP_SIZE + lane] = acc[item];
    }
}

template <typename T>
void launch_flash_attention_decode_splitkv_phase1(
    float *partial_m, float *partial_l, float *partial_acc,
    const T *q, const T *k, const T *v,
    size_t total_len, size_t nhead, size_t nkvhead, float scale,
    size_t split_size, size_t num_splits) {
    const size_t shared_bytes = TILE_K * HEAD_DIM * 2 * sizeof(float);
    dim3 grid(static_cast<unsigned int>(nhead), static_cast<unsigned int>(num_splits));
    flash_attention_decode_splitkv_phase1_kernel<<<grid, WARP_SIZE, shared_bytes>>>(
        partial_m, partial_l, partial_acc, q, k, v,
        total_len, nhead, nkvhead, scale, split_size, num_splits);
}

template <typename T>
__global__ void flash_attention_decode_splitkv_phase2_kernel(
    T *output, const float *partial_m, const float *partial_l, const float *partial_acc,
    size_t num_splits) {
    const int lane = threadIdx.x;
    const size_t head = blockIdx.x;

    float final_max = -INFINITY;
    for (size_t s = 0; s < num_splits; ++s) {
        final_max = max(final_max, partial_m[head * num_splits + s]);
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float final_sum = 0.0f;
    for (size_t s = 0; s < num_splits; ++s) {
        const float m_s = partial_m[head * num_splits + s];
        if (m_s == -INFINITY) {
            continue;
        }
        const float rescale = expf(m_s - final_max);
        final_sum += partial_l[head * num_splits + s] * rescale;
        for (int item = 0; item < 4; ++item) {
            acc[item] += partial_acc[(head * num_splits + s) * HEAD_DIM + item * WARP_SIZE + lane] * rescale;
        }
    }

    for (int item = 0; item < 4; ++item) {
        output[head * HEAD_DIM + item * WARP_SIZE + lane] = static_cast<T>(acc[item] / final_sum);
    }
}

template <typename T>
void launch_flash_attention_decode_splitkv_phase2(
    T *output, const float *partial_m, const float *partial_l, const float *partial_acc,
    size_t nhead, size_t num_splits) {
    flash_attention_decode_splitkv_phase2_kernel<<<static_cast<unsigned int>(nhead), WARP_SIZE>>>(
        output, partial_m, partial_l, partial_acc, num_splits);
}

size_t choose_num_splits(size_t total_len, size_t nhead) {
    constexpr size_t kTargetBlocks = 64;
    constexpr size_t kMinSplitSize = TILE_K;
    constexpr size_t kMaxNumSplits = 32;

    size_t num_splits = (kTargetBlocks + nhead - 1) / nhead;
    num_splits = std::max<size_t>(1, std::min(num_splits, kMaxNumSplits));

    size_t max_useful_splits = std::max<size_t>(1, total_len / kMinSplitSize);
    num_splits = std::min(num_splits, max_useful_splits);
    return num_splits;
}

template <typename T>
void launch_flash_attention_decode_splitkv(
    T *output, const T *q, const T *k, const T *v,
    size_t total_len, size_t nhead, size_t nkvhead, float scale) {
    const size_t num_splits = choose_num_splits(total_len, nhead);
    const size_t split_size = (total_len + num_splits - 1) / num_splits;

    float *partial_m = nullptr;
    float *partial_l = nullptr;
    float *partial_acc = nullptr;
    musaMalloc(reinterpret_cast<void **>(&partial_m), nhead * num_splits * sizeof(float));
    musaMalloc(reinterpret_cast<void **>(&partial_l), nhead * num_splits * sizeof(float));
    musaMalloc(reinterpret_cast<void **>(&partial_acc), nhead * num_splits * HEAD_DIM * sizeof(float));

    launch_flash_attention_decode_splitkv_phase1(
        partial_m, partial_l, partial_acc, q, k, v,
        total_len, nhead, nkvhead, scale, split_size, num_splits);
    launch_flash_attention_decode_splitkv_phase2(
        output, partial_m, partial_l, partial_acc, nhead, num_splits);

    musaFree(partial_m);
    musaFree(partial_l);
    musaFree(partial_acc);
}

} // namespace

namespace llaisys::ops::moore {

void flash_attention_decode_splitkv(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                                    float scale, llaisysDataType_t type,
                                    size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    ASSERT(seqlen == 1 && d == HEAD_DIM && dv == HEAD_DIM,
           "flash_attention_decode_splitkv requires seqlen=1 and d=dv=128");

    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_flash_attention_decode_splitkv(reinterpret_cast<float *>(attn_val),
                                              reinterpret_cast<const float *>(q),
                                              reinterpret_cast<const float *>(k),
                                              reinterpret_cast<const float *>(v),
                                              total_len, nhead, nkvhead, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_flash_attention_decode_splitkv(reinterpret_cast<__mt_bfloat16 *>(attn_val),
                                              reinterpret_cast<const __mt_bfloat16 *>(q),
                                              reinterpret_cast<const __mt_bfloat16 *>(k),
                                              reinterpret_cast<const __mt_bfloat16 *>(v),
                                              total_len, nhead, nkvhead, scale);
        return;
    case LLAISYS_DTYPE_F16:
        launch_flash_attention_decode_splitkv(reinterpret_cast<__half *>(attn_val),
                                              reinterpret_cast<const __half *>(q),
                                              reinterpret_cast<const __half *>(k),
                                              reinterpret_cast<const __half *>(v),
                                              total_len, nhead, nkvhead, scale);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::moore
