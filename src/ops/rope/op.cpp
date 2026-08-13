#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

#ifdef ENABLE_ILUVATAR_API
#include "iluvatar/rope_iluvatar.cuh"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/rope_moore.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    // Only support contiguous inputs for now.
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(LLAISYS_DTYPE_I64, pos_ids->dtype());

    // in: original q or k [seqlen, nhead, d] or [seqlen, nkvhead, d], pos_ids [seqlen,]
    // out [seqlen, nhead, d]
    CHECK_ARGUMENT(out->shape() == in->shape(), "rope: output tensor's shape must match input tensor's shape.");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "rope: position IDs tensor's first dimension must match input tensor's first dimension.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "rope: input tensor's last dimension must be even.");
    

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
        in->shape()[0], in->shape()[1], in->shape()[2]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
        in->shape()[0], in->shape()[1], in->shape()[2]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
        in->shape()[0], in->shape()[1], in->shape()[2]);
        return;
#endif
#ifdef ENABLE_ILUVATAR_API
    case LLAISYS_DEVICE_ILUVATAR:
        return iluvatar::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
        in->shape()[0], in->shape()[1], in->shape()[2]);
#endif
#ifdef ENABLE_MOORE_API
    case LLAISYS_DEVICE_MOORE:
        return moore::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(),
        in->shape()[0], in->shape()[1], in->shape()[2]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
