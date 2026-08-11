#pragma once

#include "../tensor/tensor.hpp"

#include <cstddef>

namespace llaisys::models {

// single layer kvcache
class KVCache {
public:
    KVCache(size_t maxseq, size_t d_kv, llaisysDataType_t dtype,
            llaisysDeviceType_t device_type, int device_id);

    // append new RoPE K/V
    void append(tensor_t new_k, tensor_t new_v);

    // [total_len, d_kv]
    tensor_t getK(size_t total_len) const;
    tensor_t getV(size_t total_len) const;

    size_t curLen() const { return cur_len_; }

private:
    tensor_t k_cache_;
    tensor_t v_cache_;
    size_t cur_len_ = 0;
};

} // namespace llaisys::models
