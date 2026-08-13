#include "moore_resource.cuh"

namespace llaisys::device::moore {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_MOORE, device_id) {
    musaSetDevice(device_id);
    mublasCreate(&_mublas_handle);
}

Resource::~Resource() {
    mublasDestroy(_mublas_handle);
}

DeviceResource *getDeviceResource(int device_id){
    return new Resource(device_id);
}

} // namespace llaisys::device::moore
