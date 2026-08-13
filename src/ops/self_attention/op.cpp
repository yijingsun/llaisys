#include "op.hpp"

#include "../../core/core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#include "nvidia/flash_attention_nvidia.cuh"
#endif

#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/self_attention_iluvatar.cuh"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/self_attention_moore.cuh"
#include "moore/flash_attention_moore.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    // Only support contiguous inputs for now.
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "Self-Attention: all tensors must be contiguous.");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    
    // atten_val [seqlen, nhead, dv]
    // q [seqlen, nhead, d], k  [total_len, nkvhead, d], v [total_len, nkvhead, dv]
    // total_len = seqlen + past_len, nkvhead = nhead * nkv_ratio
    CHECK_ARGUMENT(k->shape() == v->shape(), "Self -Attention: key and value tensors must have the same shape.");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0], "Self-Attention: attention value and query tensors must have the same sequence length.");
    CHECK_ARGUMENT(attn_val->shape()[1] == q->shape()[1], "Self-Attention: attention value and query tensors must have the same number of heads.");
    CHECK_ARGUMENT(attn_val->shape()[2] == v->shape()[2], "Self-Attention: attention value and value tensors must have the same dimension.");
    // less kvhead less attention compute, less k/v cache
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "Self-Attention: number of heads in query must be divisible by number of heads in key/value.");
    
    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
        q->shape()[0], q->shape()[1], q->shape()[2], k->shape()[0], k->shape()[1], k->shape()[2]);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
        q->shape()[0], q->shape()[1], q->shape()[2], k->shape()[0], k->shape()[1], k->shape()[2]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        size_t seqlen = q->shape()[0];
        size_t total_len = k->shape()[0];
        size_t nhead = q->shape()[1];
        size_t nkvhead = k->shape()[1];
        size_t d = q->shape()[2];
        size_t dv = v->shape()[2];
        constexpr size_t kSplitKVThreshold = 256;
        if (seqlen > 1 && total_len == seqlen && d == 128 && dv == 128) {
            nvidia::flash_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  scale, attn_val->dtype(), seqlen, nhead, d, total_len, nkvhead, dv);
        } else if (seqlen == 1 && d == 128 && dv == 128 && total_len > kSplitKVThreshold) {
            nvidia::flash_attention_decode_splitkv(attn_val->data(), q->data(), k->data(), v->data(),
                                                 scale, attn_val->dtype(), seqlen, nhead, d, total_len, nkvhead, dv);
        } else if (seqlen == 1 && d == 128 && dv == 128) {
            nvidia::flash_attention_decode(attn_val->data(), q->data(), k->data(), v->data(),
                                         scale, attn_val->dtype(), seqlen, nhead, d, total_len, nkvhead, dv);
        } else {
            nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                 scale, attn_val->dtype(), seqlen, nhead, d, total_len, nkvhead, dv);
        }
        return;
    }
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
        q->shape()[0], q->shape()[1], q->shape()[2], k->shape()[0], k->shape()[1], k->shape()[2]);
#endif
#ifdef ENABLE_MOORE_API
    case LLAISYS_DEVICE_MOORE: {
        size_t seqlen = q->shape()[0];
        size_t total_len = k->shape()[0];
        size_t nhead = q->shape()[1];
        size_t nkvhead = k->shape()[1];
        size_t d = q->shape()[2];
        size_t dv = v->shape()[2];
        constexpr size_t kSplitKVThreshold = 256;
        if (seqlen > 1 && total_len == seqlen && d == 128 && dv == 128) {
            moore::flash_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  scale, attn_val->dtype(), seqlen, nhead, d, total_len, nkvhead, dv);
        } else if (seqlen == 1 && d == 128 && dv == 128 && total_len > kSplitKVThreshold) {
            moore::flash_attention_decode_splitkv(attn_val->data(), q->data(), k->data(), v->data(),
                                                 scale, attn_val->dtype(), seqlen, nhead, d, total_len, nkvhead, dv);
        } else if (seqlen == 1 && d == 128 && dv == 128) {
            moore::flash_attention_decode(attn_val->data(), q->data(), k->data(), v->data(),
                                         scale, attn_val->dtype(), seqlen, nhead, d, total_len, nkvhead, dv);
        } else {
            moore::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                 scale, attn_val->dtype(), seqlen, nhead, d, total_len, nkvhead, dv);
        }
        return;
    }
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
