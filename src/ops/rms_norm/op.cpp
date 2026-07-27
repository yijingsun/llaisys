#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    // Only support contiguous inputs for now.
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    
    // in [batch, in_features], weight [in_features], out [batch, in_features]
    // RMS normalization: out = in * weight / sqrt(mean(in^2) + eps) (Root Mean Square Normalization)
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[0], "rms_norm: input tensor's second dimension must match weight tensor's first dimension.");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "rms_norm: output tensor's first dimension must match input tensor's first dimension.");
    CHECK_ARGUMENT(out->shape()[1] == in->shape()[1], "rms_norm: output tensor's second dimension must match input tensor's second dimension.");
    

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(),
        in->shape()[0], in->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(),
        in->shape()[0], in->shape()[1]);
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
