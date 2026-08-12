#include "op.hpp"

#include "../../core/core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/swiglu_iluvatar.cuh"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    // Only support contiguous inputs for now.
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    // out, gate, up [seqlen, intermediate_size] 
    CHECK_ARGUMENT(gate->shape() == up->shape() && out->shape() == gate->shape(), "SwiGLU: gate and up tensors must have the same shape.");
    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), up->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), up->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::swiglu(out->data(), gate->data(), up->data(), out->dtype(), up->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
