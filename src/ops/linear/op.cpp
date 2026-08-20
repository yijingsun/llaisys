#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    }
    // Only support contiguous inputs for now.
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!bias || bias->isContiguous()), "Linear: all tensors must be contiguous.");

    // in [batch, in_features], weight [out_features, in_features], bias [out_features], out [batch, out_features]
    // out = in * weight^T + bias
    CHECK_ARGUMENT(in->shape()[0] == out->shape()[0], "linear: input tensor's first dimension must match output tensor's first dimension.");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "linear: input tensor's second dimension must match weight tensor's second dimension.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "linear: output tensor's second dimension must match weight tensor's first dimension.");

    std::byte *bias_data = nullptr;
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->shape()[0] == weight->shape()[0], "linear: bias tensor's first dimension must match weight tensor's first dimension.");
        bias_data = bias->data();
    }

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(),
                           in->shape()[0], in->shape()[1], out->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias_data, out->dtype(),
                           in->shape()[0], in->shape()[1], out->shape()[1]);
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
