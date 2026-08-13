#include "../../../utils.hpp"
#include "linear_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../../../device/nvidia/nvidia_resource.cuh"
#define CEIL(a, b) (((a) + (b) - 1) / (b))

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status_ = (call);                                       \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,     \
                         __LINE__, static_cast<int>(status_));                 \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, cudaGetErrorString(err_));                   \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

template <typename T>
__global__ void broadcast_bias_kernel(const T *bias, T *dst, size_t M, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < M * N) dst[idx] = bias[idx % N];
}

// out[M,N] = in[M,K] * weight[N,K]^T + bias[N]. cublasGemmEx computes in-place
// (out = alpha*A*B + beta*C with C == out), so the bias is written into out
// first and folded in with beta=1: the addition happens inside the fp32
// accumulation and the result is rounded exactly once, matching torch's addmm.
// A separate bf16 bias add would round the GEMM output twice and lose precision.
template <typename T, cudaDataType_t DT>
static void gemm_with_bias(cublasHandle_t handle, T *out, const T *in, const T *weight,
                           const T *bias, size_t M, size_t N, size_t K) {
    const float alpha = 1.0f;
    float beta = 0.0f;
    if (bias != nullptr) {
        if (M == 1) {
            CHECK_CUDA(cudaMemcpy(out, bias, N * sizeof(T), cudaMemcpyDeviceToDevice));
        } else {
            broadcast_bias_kernel<<<static_cast<int>(CEIL(M * N, 256)), 256>>>(
                bias, out, M, N);
        }
        beta = 1.0f;
    }
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        weight,
        DT,
        static_cast<int>(K),
        in,
        DT,
        static_cast<int>(K),
        &beta,
        out,
        DT,
        static_cast<int>(N),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}


// in : [M, K], weight : [N, K], bias : [N], out : [M, N]；out = in * weight^T + bias
namespace llaisys::ops::nvidia {

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t M, size_t N, size_t K,llaisys::device::DeviceResource *resource) {
    auto handle = static_cast<llaisys::device::nvidia::Resource*>(resource)->cublasHandle();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        gemm_with_bias<float, CUDA_R_32F>(handle, reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias), M, N, K);
        return;
    case LLAISYS_DTYPE_BF16:
        gemm_with_bias<__nv_bfloat16, CUDA_R_16BF>(
            handle, reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight),
            reinterpret_cast<const __nv_bfloat16 *>(bias), M, N, K);
        return;
    case LLAISYS_DTYPE_F16:
        gemm_with_bias<__half, CUDA_R_16F>(handle, reinterpret_cast<__half *>(out),
                       reinterpret_cast<const __half *>(in),
                       reinterpret_cast<const __half *>(weight),
                       reinterpret_cast<const __half *>(bias), M, N, K);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
