#include "../../../utils.hpp"
#include "self_attention_iluvatar.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

// Causal self-attention with GQA support
// attn_val [seqlen, nhead, dv]
// q [seqlen, nhead, d], k [total_len, nkvhead, d], v [total_len, nkvhead, dv]
// Each block handles one (seq_pos, head) pair
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v, float scale,
                                      size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    size_t seq = blockIdx.x;  // sequence position
    size_t head = blockIdx.y; // query head
    size_t kv_head = head * nkvhead / nhead; // GQA: map query head to kv head

    if (seq >= seqlen || head >= nhead) return;

    const T *q_ptr = q + seq * nhead * d + head * d;
    const T *k_base = k + kv_head * d;
    const T *v_base = v + kv_head * dv;
    T *out_ptr = attn_val + seq * nhead * dv + head * dv;

    // Thread-level work over total_len
    extern __shared__ float shared[];
    float *scores = shared; // [total_len]

    // Step 1: compute dot products and apply causal mask
    for (size_t ki = threadIdx.x; ki < total_len; ki += blockDim.x) {
        float dot = 0.0f;
        for (size_t n = 0; n < d; n++) {
            float qv, kv;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                qv = __bfloat162float(q_ptr[n]);
                kv = __bfloat162float(k_base[ki * nkvhead * d + n]);
            } else if constexpr (std::is_same_v<T, __half>) {
                qv = __half2float(q_ptr[n]);
                kv = __half2float(k_base[ki * nkvhead * d + n]);
            } else {
                qv = q_ptr[n];
                kv = k_base[ki * nkvhead * d + n];
            }
            dot += qv * kv;
        }
        dot *= scale;
        // Causal mask: mask out future positions
        // j > total_len - seqlen + i means future
        if (ki > total_len - seqlen + seq) {
            dot = -INFINITY;
        }
        scores[ki] = dot;
    }
    __syncthreads();

    // Step 2: find max for numerical stability (parallel reduction)
    extern __shared__ char shared_raw[];
    float *scores_mem = reinterpret_cast<float *>(shared_raw);
    float *max_mem = reinterpret_cast<float *>(shared_raw + total_len * sizeof(float));

    float local_max = -INFINITY;
    for (size_t ki = threadIdx.x; ki < total_len; ki += blockDim.x) {
        if (scores_mem[ki] > local_max) local_max = scores_mem[ki];
    }
    max_mem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && max_mem[threadIdx.x + s] > max_mem[threadIdx.x]) {
            max_mem[threadIdx.x] = max_mem[threadIdx.x + s];
        }
        __syncthreads();
    }
    float max_score = max_mem[0];

    // Step 3: exp and sum (parallel reduction)
    float *sum_mem = max_mem; // reuse
    float local_sum = 0.0f;
    for (size_t ki = threadIdx.x; ki < total_len; ki += blockDim.x) {
        float e = expf(scores_mem[ki] - max_score);
        scores_mem[ki] = e;
        local_sum += e;
    }
    sum_mem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_mem[threadIdx.x] += sum_mem[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total_sum = sum_mem[0];

    // Step 4: weighted sum over v
    for (size_t nv = threadIdx.x; nv < dv; nv += blockDim.x) {
        float attn_value = 0.0f;
        for (size_t vi = 0; vi < total_len; vi++) {
            float softmax_score = scores_mem[vi] / total_sum;
            float vv;
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                vv = __bfloat162float(v_base[vi * nkvhead * dv + nv]);
            } else if constexpr (std::is_same_v<T, __half>) {
                vv = __half2float(v_base[vi * nkvhead * dv + nv]);
            } else {
                vv = v_base[vi * nkvhead * dv + nv];
            }
            attn_value += softmax_score * vv;
        }
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            out_ptr[nv] = __float2bfloat16(attn_value);
        } else if constexpr (std::is_same_v<T, __half>) {
            out_ptr[nv] = __float2half(attn_value);
        } else {
            out_ptr[nv] = attn_value;
        }
    }
}

namespace llaisys::ops::iluvatar {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, float scale, llaisysDataType_t type,
                    size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    dim3 grid(static_cast<unsigned>(seqlen), static_cast<unsigned>(nhead));
    int block_size = 256;
    // shared memory: scores[total_len] + reduction buffer[block_size]
    size_t shared_mem = total_len * sizeof(float) + block_size * sizeof(float);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float><<<grid, block_size, shared_mem>>>(
            reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
            scale, seqlen, nhead, d, total_len, nkvhead, dv);
        return;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<__nv_bfloat16><<<grid, block_size, shared_mem>>>(
            reinterpret_cast<__nv_bfloat16 *>(attn_val), reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k), reinterpret_cast<const __nv_bfloat16 *>(v),
            scale, seqlen, nhead, d, total_len, nkvhead, dv);
        return;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<__half><<<grid, block_size, shared_mem>>>(
            reinterpret_cast<__half *>(attn_val), reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k), reinterpret_cast<const __half *>(v),
            scale, seqlen, nhead, d, total_len, nkvhead, dv);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::iluvatar
