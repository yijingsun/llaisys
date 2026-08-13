#include "../../../utils.hpp"
#include "linear_moore.cuh"
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <mublas.h>
#include "../../../device/moore/moore_resource.cuh"
#define CEIL(a, b) (((a) + (b) - 1) / (b))

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < M * N) out[idx] = out[idx] + bias[idx % N];
}

// mublasGemmEx on MUSA only supports f16 inputs with a f32 accumulator/output
// (a=b=MUSA_R_16F, c=MUSA_R_32F, compute=MUBLAS_COMPUTE_32F): the combos with
// a f16 output either reject the call or silently return zeros. So the f16
// path computes into a temp f32 buffer and down-converts here.
__global__ void f32_to_f16_kernel(__half *out, const float *in, size_t M, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < M * N) out[idx] = __float2half(in[idx]);
}


// in : [M, K], weight : [N, K], bias : [N], out : [M, N]；out = in * weight^T + bias
namespace llaisys::ops::moore {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K,llaisys::device::DeviceResource *resource) {
    auto handle = static_cast<llaisys::device::moore::Resource*>(resource)->mublasHandle();
    switch (type) {
    case LLAISYS_DTYPE_F32:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_f32 = reinterpret_cast<const float *>(in);
        const auto *weight_f32 = reinterpret_cast<const float *>(weight);
        const auto *bias_f32 = reinterpret_cast<const float *>(bias);
        auto *out_f32 = reinterpret_cast<float *>(out);

        mublasStatus_t status = mublasSgemm(
            handle,
            MUBLAS_OP_T,
            MUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_f32,
            static_cast<int>(K),
            in_f32,
            static_cast<int>(K),
            &beta,
            out_f32,
            static_cast<int>(N)
        );
        if(bias != nullptr){
        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N,block_size));
        add_bias_kernel<<<grid_size, block_size>>>(out_f32, bias_f32 , M, N);
        
        }
        return;
    }
    case LLAISYS_DTYPE_BF16:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_bf16 =  reinterpret_cast<const __mt_bfloat16*>(in);
        const auto *weight_bf16 = reinterpret_cast<const __mt_bfloat16 *>(weight);
        const auto *bias_bf16 = reinterpret_cast<const __mt_bfloat16 *>(bias);
        auto *out_bf16 = reinterpret_cast<__mt_bfloat16 *>(out);

        mublasStatus_t status = mublasGemmEx(
            handle,
            MUBLAS_OP_T,
            MUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_bf16,
            MUSA_R_16BF,
            static_cast<int>(K),
            in_bf16,
            MUSA_R_16BF,
            static_cast<int>(K),
            &beta,
            out_bf16,
            MUSA_R_16BF,
            static_cast<int>(N),
            MUSA_R_32F,
            MUBLAS_GEMM_DEFAULT
        );
        if(bias != nullptr){
        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N,block_size));

        add_bias_kernel<<<grid_size, block_size>>>(out_bf16, bias_bf16 , M, N);
        
        }
        return;
    }

    case LLAISYS_DTYPE_F16:
    {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const auto *in_f16 =  reinterpret_cast<const __half*>(in);
        const auto *weight_f16 = reinterpret_cast<const __half *>(weight);
        const auto *bias_f16 = reinterpret_cast<const __half *>(bias);
        auto *out_f16 = reinterpret_cast<__half *>(out);

        // f16 gemm must accumulate into f32 (see f32_to_f16_kernel note).
        float *tmp_f32 = nullptr;
        musaMalloc(reinterpret_cast<void **>(&tmp_f32), M * N * sizeof(float));

        mublasStatus_t status = mublasGemmEx(
            handle,
            MUBLAS_OP_T,
            MUBLAS_OP_N,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            &alpha,
            weight_f16,
            MUSA_R_16F,
            static_cast<int>(K),
            in_f16,
            MUSA_R_16F,
            static_cast<int>(K),
            &beta,
            tmp_f32,
            MUSA_R_32F,
            static_cast<int>(N),
            MUBLAS_COMPUTE_32F,
            MUBLAS_GEMM_DEFAULT
        );

        constexpr int block_size = 256;
        int grid_size = static_cast<int>(CEIL(M * N, block_size));
        f32_to_f16_kernel<<<grid_size, block_size>>>(out_f16, tmp_f32, M, N);
        if(bias != nullptr){
        add_bias_kernel<<<grid_size, block_size>>>(out_f16, bias_f16 , M, N);
        
        }
        musaFree(tmp_f32);
        return;
    }

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::moore
