#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::moore {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_indices, size_t vocab_size, size_t embd_dim);
}
