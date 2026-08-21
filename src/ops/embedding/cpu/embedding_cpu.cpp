#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t seqlen, size_t vocab_size, size_t embedding_dim) {
    int64_t vocab_size_ = static_cast<int64_t>(vocab_size);
    for (size_t i = 0; i < seqlen; i++) {
        if (index[i] < 0 || index[i] >= vocab_size_) {
            throw std::runtime_error("Index out of bounds");
        }
    }
    #pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(seqlen); i++) {
        int64_t idx = index[i];
        for (size_t j = 0; j < embedding_dim; j++) {
            out[i * embedding_dim + j] = weight[idx * embedding_dim + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t seqlen, size_t vocab_size, size_t embedding_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), seqlen, vocab_size, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), seqlen, vocab_size, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), seqlen, vocab_size, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
