#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void flash_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     float scale, llaisysDataType_t type,
                     size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv);

void flash_attention_decode(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                            float scale, llaisysDataType_t type,
                            size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv);

void flash_attention_decode_splitkv(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                                    float scale, llaisysDataType_t type,
                                    size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv);

}
