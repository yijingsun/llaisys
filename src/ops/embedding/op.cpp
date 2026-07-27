#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"


namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    // Only support contiguous inputs for now.
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(index->dtype(), LLAISYS_DTYPE_I64);
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "embedding: output tensor's first dimension must match index tensor's first dimension.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "embedding: output tensor's second dimension must match weight tensor's second dimension.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0], weight->shape()[0], weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(), out->shape()[0], weight->shape()[0], weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
