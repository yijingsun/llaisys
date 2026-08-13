#include "../../../utils.hpp"
#include "swiglu_moore.cuh"
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <musa_bf16.h>

// swiglu(gate, up) = silu(gate) * up，silu(x) = x/(1+exp(-x))。
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float gate_val = (float)gate[idx];
        gate_val = gate_val / (1 + std::exp(-gate_val)); // silu(gate)
        float up_val = (float)up[idx];
        float swiglu_val = gate_val * up_val;
        out[idx] = swiglu_val;
    }
}

template <typename T>
void launch_swiglu(T *out, const T *gate, const T *up, size_t numel) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    swiglu_kernel<<<grid_size, block_size>>>(out, gate, up, numel);
}

namespace llaisys::ops::moore {

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_swiglu(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            numel
        );
        return;

    case LLAISYS_DTYPE_BF16:
        launch_swiglu(
            reinterpret_cast<__mt_bfloat16 *>(out),
            reinterpret_cast<const __mt_bfloat16 *>(gate),
            reinterpret_cast<const __mt_bfloat16 *>(up),
            numel
        );
        return;

    case LLAISYS_DTYPE_F16:
        launch_swiglu(
            reinterpret_cast<__half *>(out),
            reinterpret_cast<const __half *>(gate),
            reinterpret_cast<const __half *>(up),
            numel
        );
        return;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
