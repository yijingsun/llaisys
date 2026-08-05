#pragma once
#include "llaisys/models/qwen2.h"

#include "../models/qwen2/model.hpp"

__C {
    typedef struct LlaisysQwen2Model {
        llaisys::models::qwen2::model_t model;
    } LlaisysQwen2Model;

    typedef struct LlaisysQwen2Meta {
        llaisys::models::qwen2::ModelMeta meta;
    } LlaisysQwen2Meta;

    typedef struct LlaisysQwen2Weights {
        llaisys::models::qwen2::model_weights_t weights;
    } LlaisysQwen2Weights;
}
