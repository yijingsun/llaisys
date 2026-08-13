#include "kv_cache.hpp"

#include "../utils.hpp"

#include <stdexcept>

namespace llaisys::models {

class KVCache;
using kvcache_t = std::shared_ptr<KVCache>;

KVCache::KVCache(size_t maxseq, size_t d_kv, llaisysDataType_t dtype,
                 llaisysDeviceType_t device_type, int device_id) {
    k_cache_ = Tensor::create({maxseq, d_kv}, dtype, device_type, device_id);
    v_cache_ = Tensor::create({maxseq, d_kv}, dtype, device_type, device_id);
}

void KVCache::append(tensor_t new_k, tensor_t new_v) {
    if (!new_k || !new_v) {
        throw std::runtime_error("KVCache::append: new_k or new_v is null");
    }
    CHECK_ARGUMENT(new_k->dtype() == k_cache_->dtype(),
                   "KVCache::append: new_k dtype does not match cache dtype");
    CHECK_ARGUMENT(new_v->dtype() == v_cache_->dtype(),
                   "KVCache::append: new_v dtype does not match cache dtype");
    CHECK_ARGUMENT(new_k->shape().size() == 2,
                   "KVCache::append: new_k must have 2 dimensions [ntoken, d_kv]");
    CHECK_ARGUMENT(new_k->shape()[1] == k_cache_->shape()[1],
                   "KVCache::append: new_k d_kv does not match cache d_kv");
    CHECK_ARGUMENT(new_k->shape() == new_v->shape(),
                   "KVCache::append: new_k and new_v must have the same shape");

    size_t ntoken = new_k->shape()[0];
    CHECK_ARGUMENT(cur_len_ + ntoken <= k_cache_->shape()[0],
                   "KVCache::append: exceeds max sequence length");

    // copy new K/V into cache slice; on GPU both sides live in device memory,
    // so a plain load() (H2D) would treat the device pointer as a host pointer.
    size_t bytes = new_k->numel() * new_k->elementSize();
    auto k_slice = k_cache_->slice(0, cur_len_, cur_len_ + ntoken);
    auto v_slice = v_cache_->slice(0, cur_len_, cur_len_ + ntoken);
    core::context().setDevice(k_cache_->deviceType(), k_cache_->deviceId());
    auto &runtime = core::context().runtime();
    runtime.api()->memcpy_sync(k_slice->data(), new_k->data(), bytes, LLAISYS_MEMCPY_D2D);
    runtime.api()->memcpy_sync(v_slice->data(), new_v->data(), bytes, LLAISYS_MEMCPY_D2D);

    cur_len_ += ntoken;
}

tensor_t KVCache::getK(size_t total_len) const {
    CHECK_ARGUMENT(total_len <= cur_len_,
                   "KVCache::getK: requested total_len exceeds cached length");
    return k_cache_->slice(0, 0, total_len);
}

tensor_t KVCache::getV(size_t total_len) const {
    CHECK_ARGUMENT(total_len <= cur_len_,
                   "KVCache::getV: requested total_len exceeds cached length");
    return v_cache_->slice(0, 0, total_len);
}

} // namespace llaisys::models
