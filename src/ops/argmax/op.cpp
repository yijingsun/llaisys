#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t max_index = 0;
    float max_value = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < numel; i++) {
        float val;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            val = llaisys::utils::cast<float>(vals[i]);
        } else {
            val = vals[i];
        }
        if (val > max_value) {
            max_value = val;
            max_index = i;
        }
    }
    
    max_idx[0] = static_cast<int64_t>(max_index);
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        max_val[0] = llaisys::utils::cast<T>(max_value);
    } else {
        max_val[0] = max_value;
    }
}

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, const tensor_t vals) {
    CHECK_ARGUMENT(vals->numel() != 0, "argmax: input tensor is empty.");
    CHECK_ARGUMENT(max_val->dtype() == vals->dtype(), "argmax: max_val tensor must have the same dtype as vals tensor.");
    
    llaisysDataType_t type = vals->dtype();
    size_t numel = vals->numel();

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx->data()), reinterpret_cast<float *>(max_val->data()), reinterpret_cast<const float *>(vals->data()), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx->data()), reinterpret_cast<llaisys::bf16_t *>(max_val->data()), reinterpret_cast<const llaisys::bf16_t *>(vals->data()), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx->data()), reinterpret_cast<llaisys::fp16_t *>(max_val->data()), reinterpret_cast<const llaisys::fp16_t *>(vals->data()), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
