#include "model.hpp"

#include <numeric>

namespace llaisys::models::qwen2 {

Model::Model(ModelMeta meta, ModelWeights weights)
    : _meta(std::move(meta)), _weights(std::move(weights)) {}

int select_device_id(llaisysDeviceType_t device_type, const std::vector<int> &device_ids, int ndevice) {
    if (device_type == LLAISYS_DEVICE_CPU) {
        return 0;
    }
    if (!device_ids.empty()) {
        return device_ids[0];
    }
    if (ndevice > 0) {
        return 0;
    }
    return 0;
}

model_t Model::create_model(
    const ModelMeta &meta,
    llaisysDeviceType_t device_type,
    const std::vector<int> &device_ids,
    int ndevice) {
    CHECK_ARGUMENT(meta.hidden_size > 0, "hidden_size must be positive");
    CHECK_ARGUMENT(meta.nlayer > 0, "nlayer must be positive");
    CHECK_ARGUMENT(meta.vocab_size > 0, "vocab_size must be positive");
    CHECK_ARGUMENT(meta.nhead > 0, "nhead must be positive");
    CHECK_ARGUMENT(meta.nkvhead > 0, "nkvhead must be positive");
    CHECK_ARGUMENT(meta.d_intermediate > 0, "d_intermediate must be positive");

    const int device_id = select_device_id(device_type, device_ids, ndevice);

    ModelWeights weights;
    
    // global weights
    weights.in_embed = Tensor::create({meta.vocab_size, meta.hidden_size}, meta.dtype, device_type, device_id);
    weights.out_embed = Tensor::create({meta.vocab_size, meta.hidden_size}, meta.dtype, device_type, device_id);
    weights.out_norm_w = Tensor::create({meta.hidden_size}, meta.dtype, device_type, device_id);

    // layer weights
    weights.layers.resize(meta.nlayer);
    for (size_t layer_idx = 0; layer_idx < meta.nlayer; ++layer_idx) {
        auto &layer = weights.layers[layer_idx];
        
        size_t d_kv = meta.nkvhead * (meta.hidden_size / meta.nhead);

        // attention weights
        layer.attn_norm_w = Tensor::create({meta.hidden_size}, meta.dtype, device_type, device_id);

        layer.attn_q_w = Tensor::create({meta.hidden_size, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.attn_q_b = Tensor::create({meta.hidden_size}, meta.dtype, device_type, device_id);

        layer.attn_k_w = Tensor::create({d_kv, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.attn_k_b = Tensor::create({d_kv}, meta.dtype, device_type, device_id);

        layer.attn_v_w = Tensor::create({d_kv, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.attn_v_b = Tensor::create({d_kv}, meta.dtype, device_type, device_id);

        layer.attn_o_w = Tensor::create({meta.hidden_size, meta.hidden_size}, meta.dtype, device_type, device_id);
        
        // mlp weights
        layer.mlp_norm_w = Tensor::create({meta.hidden_size}, meta.dtype, device_type, device_id);
        
        layer.mlp_gate_w = Tensor::create({meta.d_intermediate, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.mlp_up_w = Tensor::create({meta.d_intermediate, meta.hidden_size}, meta.dtype, device_type, device_id);
        layer.mlp_down_w = Tensor::create({meta.hidden_size, meta.d_intermediate}, meta.dtype, device_type, device_id);
    }

    return model_t(new Model(meta, std::move(weights)));
}

model_t create(
    const ModelMeta &meta,
    llaisysDeviceType_t device_type,
    const std::vector<int> &device_ids,
    int ndevice) {
    return Model::create_model(meta, device_type, device_ids, ndevice);
}

ModelWeights_t getWeights(const model_t &model) {
    if (!model) {
        throw std::runtime_error("Qwen2 model is null");
    }
    return std::make_shared<llaisys::models::qwen2::ModelWeights>(model->weights());
}

void loadWeights(
    const model_t &model,
    const std::string &name,
    const void *data,
    const size_t numel) {
    if (!model) {
        throw std::runtime_error("Qwen2 model is null");
    }

    tensor_t tensor = Model::getTensorByName(model->weights(), name);
    if (!tensor) {
        throw std::runtime_error("Tensor with name " + name + " not found in model weights");
    }

    if (tensor->numel() != numel) {
        throw std::runtime_error("Number of elements mismatch for tensor " + name);
    }

    tensor->load(data);

    // std::cout << tensor->info() << std::endl; // debug
}

tensor_t Model::getTensorByName(const ModelWeights &weights, const std::string &name) {
    // Global weights
    if (name == "model.embed_tokens.weight") return weights.in_embed;
    if (name == "model.norm.weight") return weights.out_norm_w;
    if (name == "lm_head.weight") return weights.out_embed;
    
    // Layer weights: "model.layers.{idx}.{field}"
    const std::string prefix = "model.layers.";
    if (name.find(prefix) == 0) {
        size_t idx_start = prefix.length();
        size_t idx_end = name.find('.', idx_start);
        if (idx_end == std::string::npos) return nullptr;
        
        int layer_idx = std::stoi(name.substr(idx_start, idx_end - idx_start));
        if (layer_idx < 0 || layer_idx >= (int)weights.layers.size()) return nullptr;
        
        const auto &layer = weights.layers[layer_idx];
        std::string field = name.substr(idx_end + 1);
        
        if (field == "input_layernorm.weight") return layer.attn_norm_w;

        if (field == "self_attn.q_proj.weight") return layer.attn_q_w;
        if (field == "self_attn.q_proj.bias") return layer.attn_q_b;

        if (field == "self_attn.k_proj.weight") return layer.attn_k_w;
        if (field == "self_attn.k_proj.bias") return layer.attn_k_b;

        if (field == "self_attn.v_proj.weight") return layer.attn_v_w;
        if (field == "self_attn.v_proj.bias") return layer.attn_v_b;

        if (field == "post_attention_layernorm.weight") return layer.mlp_norm_w;

        if (field == "self_attn.o_proj.weight") return layer.attn_o_w;

        if (field == "mlp.gate_proj.weight") return layer.mlp_gate_w;
        if (field == "mlp.up_proj.weight") return layer.mlp_up_w;
        if (field == "mlp.down_proj.weight") return layer.mlp_down_w;
    }
    
    return nullptr;
}

int64_t infer(model_t model, int64_t *token_ids, size_t ntoken) {
    if (!model || !token_ids || ntoken == 0) {
        return 0;
    }
    return static_cast<int64_t>(token_ids[ntoken - 1]);
}

} // namespace llaisys::models::qwen2