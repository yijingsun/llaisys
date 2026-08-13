#pragma once
#include <mublas.h>
#include "../device_resource.hpp"

namespace llaisys::device::moore {
class Resource : public llaisys::device::DeviceResource {
private:
    mublasHandle_t  _mublas_handle;
public:
    Resource(int device_id);
    ~Resource();


    mublasHandle_t mublasHandle() const {
        return _mublas_handle;
    }
};
} // namespace llaisys::device::moore
