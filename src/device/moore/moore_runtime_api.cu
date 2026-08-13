#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>
#define CHECK_MUSA(call)                                                   \
    do {                                                                   \
        musaError_t error_ = (call);                                       \
        if (error_ != musaSuccess) {                                       \
            std::fprintf(stderr, "MUSA error at %s:%d: %s\n",              \
                         __FILE__, __LINE__, musaGetErrorString(error_));   \
            std::abort();                                                  \
        }                                                                  \
    } while (0)
namespace llaisys::device::moore {

namespace runtime_api {

static musaMemcpyKind convertMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            return musaMemcpyHostToHost;
        case LLAISYS_MEMCPY_H2D:
            return musaMemcpyHostToDevice;
        case LLAISYS_MEMCPY_D2H:
            return musaMemcpyDeviceToHost;
        case LLAISYS_MEMCPY_D2D:
            return musaMemcpyDeviceToDevice;
        default:
            std::abort();
    }
}


int getDeviceCount() {
    int count = 0;
    CHECK_MUSA(musaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    CHECK_MUSA(musaSetDevice(device));
}

void deviceSynchronize() {
    CHECK_MUSA(musaDeviceSynchronize());
}

llaisysStream_t createStream() {
    musaStream_t stream = nullptr;
    CHECK_MUSA(musaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    CHECK_MUSA(musaStreamDestroy(reinterpret_cast<musaStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    CHECK_MUSA(musaStreamSynchronize(reinterpret_cast<musaStream_t> (stream)));
}

void *mallocDevice(size_t size) {
    void * ptr =nullptr;
    CHECK_MUSA(musaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CHECK_MUSA(musaFree(ptr));
}

void *mallocHost(size_t size) {
    void * ptr =nullptr;
    CHECK_MUSA(musaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    CHECK_MUSA(musaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CHECK_MUSA(musaMemcpy(dst, src, size, convertMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CHECK_MUSA(musaMemcpyAsync(dst, src, size, convertMemcpyKind(kind), reinterpret_cast<musaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::moore
