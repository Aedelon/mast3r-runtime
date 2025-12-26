// MASt3R Runtime - ViT GPU Buffers Implementation
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#import "vit_buffers.hpp"

#import <Foundation/Foundation.h>
#import <cmath>

namespace mast3r {
namespace metal {

ViTBuffers::ViTBuffers(id<MTLDevice> device, const ModelSpec& spec)
    : device_(device), spec_(spec) {
    encoder_layers_.resize(spec.depth);
    decoder_layers_.resize(spec.decoder_depth);
}

ViTBuffers::~ViTBuffers() {
    @autoreleasepool {
        patch_embed_weight_ = nil;
        patch_embed_bias_ = nil;
        pos_embed_ = nil;
        cls_token_ = nil;

        for (auto& layer : encoder_layers_) {
            layer.norm1_weight = nil;
            layer.norm1_bias = nil;
            layer.norm2_weight = nil;
            layer.norm2_bias = nil;
            layer.qkv_weight = nil;
            layer.qkv_bias = nil;
            layer.proj_weight = nil;
            layer.proj_bias = nil;
            layer.mlp_fc1_weight = nil;
            layer.mlp_fc1_bias = nil;
            layer.mlp_fc2_weight = nil;
            layer.mlp_fc2_bias = nil;
        }

        auto clear_decoder_layer = [](DecoderLayerBuffers& layer) {
            layer.norm1_weight = nil;
            layer.norm1_bias = nil;
            layer.norm2_weight = nil;
            layer.norm2_bias = nil;
            layer.qkv_weight = nil;
            layer.qkv_bias = nil;
            layer.proj_weight = nil;
            layer.proj_bias = nil;
            layer.mlp_fc1_weight = nil;
            layer.mlp_fc1_bias = nil;
            layer.mlp_fc2_weight = nil;
            layer.mlp_fc2_bias = nil;
            layer.cross_norm_weight = nil;
            layer.cross_norm_bias = nil;
            layer.cross_q_weight = nil;
            layer.cross_q_bias = nil;
            layer.cross_kv_weight = nil;
            layer.cross_kv_bias = nil;
            layer.cross_v_weight = nil;
            layer.cross_v_bias = nil;
            layer.cross_proj_weight = nil;
            layer.cross_proj_bias = nil;
        };

        for (auto& layer : decoder_layers_) {
            clear_decoder_layer(layer);
        }
        for (auto& layer : decoder_layers2_) {
            clear_decoder_layer(layer);
        }

        // DPT weights cleanup
        for (int i = 0; i < 4; i++) {
            dpt_act_postprocess_[i].conv1_weight = nil;
            dpt_act_postprocess_[i].conv1_bias = nil;
            dpt_act_postprocess_[i].conv2_weight = nil;
            dpt_act_postprocess_[i].conv2_bias = nil;
            dpt_layer_rn_[i] = nil;
            dpt_refinenet_[i].rcu1_conv1_weight = nil;
            dpt_refinenet_[i].rcu1_conv1_bias = nil;
            dpt_refinenet_[i].rcu1_conv2_weight = nil;
            dpt_refinenet_[i].rcu1_conv2_bias = nil;
            dpt_refinenet_[i].rcu2_conv1_weight = nil;
            dpt_refinenet_[i].rcu2_conv1_bias = nil;
            dpt_refinenet_[i].rcu2_conv2_weight = nil;
            dpt_refinenet_[i].rcu2_conv2_bias = nil;
            dpt_refinenet_[i].out_conv_weight = nil;
            dpt_refinenet_[i].out_conv_bias = nil;
        }
        dpt_head_.conv1_weight = nil;
        dpt_head_.conv1_bias = nil;
        dpt_head_.conv2_weight = nil;
        dpt_head_.conv2_bias = nil;
        dpt_head_.conv3_weight = nil;
        dpt_head_.conv3_bias = nil;
        local_features_mlp_.fc1_weight = nil;
        local_features_mlp_.fc1_bias = nil;
        local_features_mlp_.fc2_weight = nil;
        local_features_mlp_.fc2_bias = nil;
        rope_freqs_ = nil;
    }
}

ViTBuffers::ViTBuffers(ViTBuffers&& other) noexcept
    : device_(other.device_),
      spec_(other.spec_),
      is_uploaded_(other.is_uploaded_),
      patch_embed_weight_(other.patch_embed_weight_),
      patch_embed_bias_(other.patch_embed_bias_),
      pos_embed_(other.pos_embed_),
      cls_token_(other.cls_token_),
      encoder_layers_(std::move(other.encoder_layers_)),
      decoder_layers_(std::move(other.decoder_layers_)),
      decoder_layers2_(std::move(other.decoder_layers2_)),
      dpt_head_(other.dpt_head_),
      local_features_mlp_(other.local_features_mlp_),
      rope_freqs_(other.rope_freqs_) {
    // Copy DPT arrays
    for (int i = 0; i < 4; i++) {
        dpt_hooks_[i] = other.dpt_hooks_[i];
        dpt_act_postprocess_[i] = other.dpt_act_postprocess_[i];
        dpt_layer_rn_[i] = other.dpt_layer_rn_[i];
        dpt_refinenet_[i] = other.dpt_refinenet_[i];
    }
    other.device_ = nil;
    other.is_uploaded_ = false;
}

ViTBuffers& ViTBuffers::operator=(ViTBuffers&& other) noexcept {
    if (this != &other) {
        device_ = other.device_;
        spec_ = other.spec_;
        is_uploaded_ = other.is_uploaded_;
        patch_embed_weight_ = other.patch_embed_weight_;
        patch_embed_bias_ = other.patch_embed_bias_;
        pos_embed_ = other.pos_embed_;
        cls_token_ = other.cls_token_;
        encoder_layers_ = std::move(other.encoder_layers_);
        decoder_layers_ = std::move(other.decoder_layers_);
        decoder_layers2_ = std::move(other.decoder_layers2_);
        dpt_head_ = other.dpt_head_;
        local_features_mlp_ = other.local_features_mlp_;
        rope_freqs_ = other.rope_freqs_;

        // Copy DPT arrays
        for (int i = 0; i < 4; i++) {
            dpt_hooks_[i] = other.dpt_hooks_[i];
            dpt_act_postprocess_[i] = other.dpt_act_postprocess_[i];
            dpt_layer_rn_[i] = other.dpt_layer_rn_[i];
            dpt_refinenet_[i] = other.dpt_refinenet_[i];
        }

        other.device_ = nil;
        other.is_uploaded_ = false;
    }
    return *this;
}

id<MTLBuffer> ViTBuffers::create_buffer_from_tensor(const Tensor& tensor) {
    @autoreleasepool {
        size_t size = tensor.data.size();
        if (size == 0) return nil;

        // Metal shared memory for unified memory architecture
        return [device_ newBufferWithBytes:tensor.raw()
                                    length:size
                                   options:MTLResourceStorageModeShared];
    }
}

void ViTBuffers::upload(const ModelWeights& weights) {
    @autoreleasepool {
        NSLog(@"[mast3r] Uploading %zu tensors to GPU...", weights.tensors.size());

        // Detect weight dtype from first tensor
        weights_fp16_ = false;
        for (const auto& [name, tensor] : weights.tensors) {
            if (tensor.is_bf16()) {
                NSLog(@"[mast3r] ERROR: Model contains BF16 weights. Metal does not support BF16.");
                NSLog(@"[mast3r] Please convert your model to FP16 using the conversion script:");
                NSLog(@"[mast3r]   python -m mast3r_runtime.utils.convert --input model.pth --output model.safetensors --dtype fp16");
                throw std::runtime_error("BF16 weights not supported. Convert to FP16.");
            }
            if (tensor.is_f16()) {
                weights_fp16_ = true;
            }
            break;  // Check first tensor only
        }
        NSLog(@"[mast3r] Weight precision: %s", weights_fp16_ ? "FP16" : "FP32");

        // Detect weight naming convention based on architecture
        // DUNE uses DINOv2 encoder (nested blocks) + MASt3R decoder
        // MASt3R uses CroCoNet encoder/decoder with direct key names
        //
        // DUNE encoder: "encoder.encoder.blocks.0.{i}.*" (DINOv2 nested structure)
        // DUNE decoder: "decoder.mast3r.dec_blocks.{i}.*"
        // DUNE DPT: "decoder.mast3r.downstream_head1.dpt.*"
        // MASt3R: direct keys like "enc_blocks.{i}.*", "dec_blocks.{i}.*", "patch_embed.proj.*"
        std::string enc_prefix = spec_.is_mast3r() ? "" : "encoder.encoder.";
        std::string dec_prefix = spec_.is_mast3r() ? "" : "decoder.mast3r.";

        // Helper to load weight (already validated as FP16 or FP32, no BF16)
        auto load_weight = [&](const std::string& name) -> id<MTLBuffer> {
            if (!weights.has(name)) return nil;
            Tensor tensor = weights.get(name);
            // Weights are loaded as-is (FP16 or FP32)
            // Kernels must match the weight precision
            return create_buffer_from_tensor(tensor);
        };

        // Patch embedding
        std::string patch_w = enc_prefix + "patch_embed.proj.weight";
        std::string patch_b = enc_prefix + "patch_embed.proj.bias";

        patch_embed_weight_ = load_weight(patch_w);
        patch_embed_bias_ = load_weight(patch_b);

        // Position embedding (DUNE only)
        if (spec_.is_dune()) {
            std::string pos_key = enc_prefix + "pos_embed";
            if (weights.has(pos_key)) {
                Tensor tensor = weights.get(pos_key);
                if (tensor.is_bf16()) tensor.bf16_to_f16_inplace();
                pos_embed_ = create_buffer_from_tensor(tensor);
            }

            std::string cls_key = enc_prefix + "cls_token";
            if (weights.has(cls_key)) {
                Tensor tensor = weights.get(cls_key);
                if (tensor.is_bf16()) tensor.bf16_to_f16_inplace();
                cls_token_ = create_buffer_from_tensor(tensor);
            }
        }

        // RoPE frequencies (MASt3R only)
        if (spec_.is_mast3r()) {
            // Precompute RoPE frequencies
            int head_dim = spec_.embed_dim / spec_.num_heads;
            int half_dim = head_dim / 2;
            std::vector<float> freqs(half_dim);

            for (int i = 0; i < half_dim; i++) {
                freqs[i] = 1.0f / std::pow(10000.0f, float(2 * i) / float(head_dim));
            }

            rope_freqs_ = [device_ newBufferWithBytes:freqs.data()
                                               length:freqs.size() * sizeof(float)
                                              options:MTLResourceStorageModeShared];
        }

        // Encoder layers
        // DUNE uses DINOv2 with nested blocks: blocks.0.{i}
        // MASt3R uses enc_blocks.{i}
        for (int i = 0; i < spec_.depth; i++) {
            std::string prefix;
            if (spec_.is_dune()) {
                // DINOv2 nested structure: blocks.0.{i}
                prefix = enc_prefix + "blocks.0." + std::to_string(i) + ".";
            } else {
                // MASt3R: enc_blocks.{i}
                prefix = "enc_blocks." + std::to_string(i) + ".";
            }
            auto& layer = encoder_layers_[i];

            // Use the parent load_weight lambda which already converts to FP16
            layer.norm1_weight = load_weight(prefix + "norm1.weight");
            layer.norm1_bias = load_weight(prefix + "norm1.bias");
            layer.norm2_weight = load_weight(prefix + "norm2.weight");
            layer.norm2_bias = load_weight(prefix + "norm2.bias");
            layer.qkv_weight = load_weight(prefix + "attn.qkv.weight");
            layer.qkv_bias = load_weight(prefix + "attn.qkv.bias");
            layer.proj_weight = load_weight(prefix + "attn.proj.weight");
            layer.proj_bias = load_weight(prefix + "attn.proj.bias");
            layer.mlp_fc1_weight = load_weight(prefix + "mlp.fc1.weight");
            layer.mlp_fc1_bias = load_weight(prefix + "mlp.fc1.bias");
            layer.mlp_fc2_weight = load_weight(prefix + "mlp.fc2.weight");
            layer.mlp_fc2_bias = load_weight(prefix + "mlp.fc2.bias");
        }

        // Decoder layers (dec_blocks for image 1)
        // DUNE uses dec_blocks.{i} with prefix, MASt3R uses dec_blocks.{i} directly
        for (int i = 0; i < spec_.decoder_depth; i++) {
            std::string prefix;
            if (spec_.is_dune()) {
                prefix = dec_prefix + "dec_blocks." + std::to_string(i) + ".";
            } else {
                // MASt3R: dec_blocks.{i}
                prefix = "dec_blocks." + std::to_string(i) + ".";
            }
            auto& layer = decoder_layers_[i];

            // Self-attention (DUNE decoder has no norm before self-attn in safetensors)
            layer.norm1_weight = load_weight(prefix + "norm1.weight");
            layer.norm1_bias = load_weight(prefix + "norm1.bias");
            layer.qkv_weight = load_weight(prefix + "attn.qkv.weight");
            layer.qkv_bias = load_weight(prefix + "attn.qkv.bias");
            layer.proj_weight = load_weight(prefix + "attn.proj.weight");
            layer.proj_bias = load_weight(prefix + "attn.proj.bias");

            // Cross-attention
            // Both DUNE and MASt3R use: cross_attn.projq/projk/projv
            layer.cross_norm_weight = load_weight(prefix + "norm2.weight");
            layer.cross_norm_bias = load_weight(prefix + "norm2.bias");
            layer.cross_q_weight = load_weight(prefix + "cross_attn.projq.weight");
            layer.cross_q_bias = load_weight(prefix + "cross_attn.projq.bias");
            layer.cross_kv_weight = load_weight(prefix + "cross_attn.projk.weight");
            layer.cross_kv_bias = load_weight(prefix + "cross_attn.projk.bias");
            layer.cross_v_weight = load_weight(prefix + "cross_attn.projv.weight");
            layer.cross_v_bias = load_weight(prefix + "cross_attn.projv.bias");
            layer.cross_proj_weight = load_weight(prefix + "cross_attn.proj.weight");
            layer.cross_proj_bias = load_weight(prefix + "cross_attn.proj.bias");

            // MLP (norm3 in MASt3R-style decoders)
            layer.norm2_weight = load_weight(prefix + "norm3.weight");
            layer.norm2_bias = load_weight(prefix + "norm3.bias");
            layer.mlp_fc1_weight = load_weight(prefix + "mlp.fc1.weight");
            layer.mlp_fc1_bias = load_weight(prefix + "mlp.fc1.bias");
            layer.mlp_fc2_weight = load_weight(prefix + "mlp.fc2.weight");
            layer.mlp_fc2_bias = load_weight(prefix + "mlp.fc2.bias");
        }

        // Decoder layers 2 (dec_blocks2 for image 2, MASt3R only)
        if (spec_.is_mast3r()) {
            decoder_layers2_.resize(spec_.decoder_depth);
            for (int i = 0; i < spec_.decoder_depth; i++) {
                std::string prefix = "dec_blocks2." + std::to_string(i) + ".";
                auto& layer = decoder_layers2_[i];

                // Self-attention
                layer.norm1_weight = load_weight(prefix + "norm1.weight");
                layer.norm1_bias = load_weight(prefix + "norm1.bias");
                layer.qkv_weight = load_weight(prefix + "attn.qkv.weight");
                layer.qkv_bias = load_weight(prefix + "attn.qkv.bias");
                layer.proj_weight = load_weight(prefix + "attn.proj.weight");
                layer.proj_bias = load_weight(prefix + "attn.proj.bias");

                // Cross-attention
                layer.cross_norm_weight = load_weight(prefix + "norm2.weight");
                layer.cross_norm_bias = load_weight(prefix + "norm2.bias");
                layer.cross_q_weight = load_weight(prefix + "cross_attn.projq.weight");
                layer.cross_q_bias = load_weight(prefix + "cross_attn.projq.bias");
                layer.cross_kv_weight = load_weight(prefix + "cross_attn.projk.weight");
                layer.cross_kv_bias = load_weight(prefix + "cross_attn.projk.bias");
                layer.cross_v_weight = load_weight(prefix + "cross_attn.projv.weight");
                layer.cross_v_bias = load_weight(prefix + "cross_attn.projv.bias");
                layer.cross_proj_weight = load_weight(prefix + "cross_attn.proj.weight");
                layer.cross_proj_bias = load_weight(prefix + "cross_attn.proj.bias");

                // MLP
                layer.norm2_weight = load_weight(prefix + "norm3.weight");
                layer.norm2_bias = load_weight(prefix + "norm3.bias");
                layer.mlp_fc1_weight = load_weight(prefix + "mlp.fc1.weight");
                layer.mlp_fc1_bias = load_weight(prefix + "mlp.fc1.bias");
                layer.mlp_fc2_weight = load_weight(prefix + "mlp.fc2.weight");
                layer.mlp_fc2_bias = load_weight(prefix + "mlp.fc2.bias");
            }
        }

        // =========================================================================
        // DPT (Dense Prediction Transformer) weights
        // Full architecture for multi-scale feature fusion
        // =========================================================================
        // Reuse same load_weight helper for DPT weights
        auto load_dpt = load_weight;

        std::string dpt_prefix;
        if (spec_.is_dune()) {
            dpt_prefix = dec_prefix + "downstream_head1.dpt.";
        } else {
            dpt_prefix = "downstream_head1.dpt.";
        }

        // act_postprocess[0-3]: Channel projection + scale adjustment
        // [0]: Conv1x1 + ConvTranspose4x4 (4x upsample)
        // [1]: Conv1x1 + ConvTranspose2x2 (2x upsample)
        // [2]: Conv1x1 only (same scale)
        // [3]: Conv1x1 + Conv3x3 stride2 (0.5x downsample)
        for (int i = 0; i < 4; i++) {
            std::string ap = dpt_prefix + "act_postprocess." + std::to_string(i) + ".";
            // First module (0) is Conv1x1 or Identity
            dpt_act_postprocess_[i].conv1_weight = load_dpt(ap + "0.weight");
            dpt_act_postprocess_[i].conv1_bias = load_dpt(ap + "0.bias");
            // Second module (1) is ConvTranspose or Conv3x3 stride2 or None
            dpt_act_postprocess_[i].conv2_weight = load_dpt(ap + "1.weight");
            dpt_act_postprocess_[i].conv2_bias = load_dpt(ap + "1.bias");
        }

        // scratch.layer{1-4}_rn: Conv3x3 projection to feature_dim (256)
        // Note: indices are 1-4 in the model, we use 0-3 internally
        for (int i = 0; i < 4; i++) {
            std::string rn = dpt_prefix + "scratch.layer" + std::to_string(i + 1) + "_rn.weight";
            dpt_layer_rn_[i] = load_dpt(rn);  // No bias
        }

        // scratch.refinenet{1-4}: FeatureFusionBlocks
        // Each has resConfUnit1, resConfUnit2, and out_conv
        for (int i = 0; i < 4; i++) {
            std::string rf = dpt_prefix + "scratch.refinenet" + std::to_string(i + 1) + ".";
            auto& rn = dpt_refinenet_[i];

            // resConfUnit1: 2x Conv3x3
            rn.rcu1_conv1_weight = load_dpt(rf + "resConfUnit1.conv1.weight");
            rn.rcu1_conv1_bias = load_dpt(rf + "resConfUnit1.conv1.bias");
            rn.rcu1_conv2_weight = load_dpt(rf + "resConfUnit1.conv2.weight");
            rn.rcu1_conv2_bias = load_dpt(rf + "resConfUnit1.conv2.bias");

            // resConfUnit2: 2x Conv3x3
            rn.rcu2_conv1_weight = load_dpt(rf + "resConfUnit2.conv1.weight");
            rn.rcu2_conv1_bias = load_dpt(rf + "resConfUnit2.conv1.bias");
            rn.rcu2_conv2_weight = load_dpt(rf + "resConfUnit2.conv2.weight");
            rn.rcu2_conv2_bias = load_dpt(rf + "resConfUnit2.conv2.bias");

            // out_conv: Conv1x1
            rn.out_conv_weight = load_dpt(rf + "out_conv.weight");
            rn.out_conv_bias = load_dpt(rf + "out_conv.bias");
        }

        // DPT head: Conv3x3 -> Upsample -> Conv3x3 -> ReLU -> Conv1x1
        // head.0 = Conv2d 3x3 (256 -> 128)
        // head.2 = Conv2d 3x3 (128 -> 128)
        // head.4 = Conv2d 1x1 (128 -> 4)  [outputs pts3d(3) + conf(1)]
        dpt_head_.conv1_weight = load_dpt(dpt_prefix + "head.0.weight");
        dpt_head_.conv1_bias = load_dpt(dpt_prefix + "head.0.bias");
        dpt_head_.conv2_weight = load_dpt(dpt_prefix + "head.2.weight");
        dpt_head_.conv2_bias = load_dpt(dpt_prefix + "head.2.bias");
        dpt_head_.conv3_weight = load_dpt(dpt_prefix + "head.4.weight");
        dpt_head_.conv3_bias = load_dpt(dpt_prefix + "head.4.bias");

        // Local features MLP: fc1 -> GELU -> fc2 -> pixel_shuffle
        // Produces descriptors at full resolution
        std::string local_prefix;
        if (spec_.is_dune()) {
            local_prefix = dec_prefix + "downstream_head1.head_local_features.";
        } else {
            local_prefix = "downstream_head1.head_local_features.";
        }
        local_features_mlp_.fc1_weight = load_dpt(local_prefix + "fc1.weight");
        local_features_mlp_.fc1_bias = load_dpt(local_prefix + "fc1.bias");
        local_features_mlp_.fc2_weight = load_dpt(local_prefix + "fc2.weight");
        local_features_mlp_.fc2_bias = load_dpt(local_prefix + "fc2.bias");

        // Log what was loaded
        int dpt_count = 0;
        for (int i = 0; i < 4; i++) {
            if (dpt_act_postprocess_[i].conv1_weight) dpt_count++;
            if (dpt_layer_rn_[i]) dpt_count++;
            if (dpt_refinenet_[i].rcu1_conv1_weight) dpt_count++;
        }
        if (dpt_head_.conv1_weight) dpt_count++;
        if (local_features_mlp_.fc1_weight) dpt_count++;

        is_uploaded_ = true;
        NSLog(@"[mast3r] Weights uploaded: %d encoder + %d decoder + %d decoder2 layers, %d DPT modules",
              (int)encoder_layers_.size(), (int)decoder_layers_.size(), (int)decoder_layers2_.size(), dpt_count);
    }
}

// Encoder layer accessors
id<MTLBuffer> ViTBuffers::encoder_norm1_weight(int layer) const {
    return encoder_layers_[layer].norm1_weight;
}

id<MTLBuffer> ViTBuffers::encoder_norm1_bias(int layer) const {
    return encoder_layers_[layer].norm1_bias;
}

id<MTLBuffer> ViTBuffers::encoder_norm2_weight(int layer) const {
    return encoder_layers_[layer].norm2_weight;
}

id<MTLBuffer> ViTBuffers::encoder_norm2_bias(int layer) const {
    return encoder_layers_[layer].norm2_bias;
}

id<MTLBuffer> ViTBuffers::encoder_qkv_weight(int layer) const {
    return encoder_layers_[layer].qkv_weight;
}

id<MTLBuffer> ViTBuffers::encoder_qkv_bias(int layer) const {
    return encoder_layers_[layer].qkv_bias;
}

id<MTLBuffer> ViTBuffers::encoder_proj_weight(int layer) const {
    return encoder_layers_[layer].proj_weight;
}

id<MTLBuffer> ViTBuffers::encoder_proj_bias(int layer) const {
    return encoder_layers_[layer].proj_bias;
}

id<MTLBuffer> ViTBuffers::encoder_mlp_fc1_weight(int layer) const {
    return encoder_layers_[layer].mlp_fc1_weight;
}

id<MTLBuffer> ViTBuffers::encoder_mlp_fc1_bias(int layer) const {
    return encoder_layers_[layer].mlp_fc1_bias;
}

id<MTLBuffer> ViTBuffers::encoder_mlp_fc2_weight(int layer) const {
    return encoder_layers_[layer].mlp_fc2_weight;
}

id<MTLBuffer> ViTBuffers::encoder_mlp_fc2_bias(int layer) const {
    return encoder_layers_[layer].mlp_fc2_bias;
}

// Decoder layer accessors
id<MTLBuffer> ViTBuffers::decoder_norm1_weight(int layer) const {
    return decoder_layers_[layer].norm1_weight;
}

id<MTLBuffer> ViTBuffers::decoder_norm1_bias(int layer) const {
    return decoder_layers_[layer].norm1_bias;
}

id<MTLBuffer> ViTBuffers::decoder_norm2_weight(int layer) const {
    return decoder_layers_[layer].norm2_weight;
}

id<MTLBuffer> ViTBuffers::decoder_norm2_bias(int layer) const {
    return decoder_layers_[layer].norm2_bias;
}

id<MTLBuffer> ViTBuffers::decoder_qkv_weight(int layer) const {
    return decoder_layers_[layer].qkv_weight;
}

id<MTLBuffer> ViTBuffers::decoder_qkv_bias(int layer) const {
    return decoder_layers_[layer].qkv_bias;
}

id<MTLBuffer> ViTBuffers::decoder_proj_weight(int layer) const {
    return decoder_layers_[layer].proj_weight;
}

id<MTLBuffer> ViTBuffers::decoder_proj_bias(int layer) const {
    return decoder_layers_[layer].proj_bias;
}

id<MTLBuffer> ViTBuffers::decoder_mlp_fc1_weight(int layer) const {
    return decoder_layers_[layer].mlp_fc1_weight;
}

id<MTLBuffer> ViTBuffers::decoder_mlp_fc1_bias(int layer) const {
    return decoder_layers_[layer].mlp_fc1_bias;
}

id<MTLBuffer> ViTBuffers::decoder_mlp_fc2_weight(int layer) const {
    return decoder_layers_[layer].mlp_fc2_weight;
}

id<MTLBuffer> ViTBuffers::decoder_mlp_fc2_bias(int layer) const {
    return decoder_layers_[layer].mlp_fc2_bias;
}

// Cross-attention accessors
id<MTLBuffer> ViTBuffers::decoder_cross_norm_weight(int layer) const {
    return decoder_layers_[layer].cross_norm_weight;
}

id<MTLBuffer> ViTBuffers::decoder_cross_norm_bias(int layer) const {
    return decoder_layers_[layer].cross_norm_bias;
}

id<MTLBuffer> ViTBuffers::decoder_cross_q_weight(int layer) const {
    return decoder_layers_[layer].cross_q_weight;
}

id<MTLBuffer> ViTBuffers::decoder_cross_q_bias(int layer) const {
    return decoder_layers_[layer].cross_q_bias;
}

id<MTLBuffer> ViTBuffers::decoder_cross_kv_weight(int layer) const {
    return decoder_layers_[layer].cross_kv_weight;
}

id<MTLBuffer> ViTBuffers::decoder_cross_kv_bias(int layer) const {
    return decoder_layers_[layer].cross_kv_bias;
}

id<MTLBuffer> ViTBuffers::decoder_cross_proj_weight(int layer) const {
    return decoder_layers_[layer].cross_proj_weight;
}

id<MTLBuffer> ViTBuffers::decoder_cross_proj_bias(int layer) const {
    return decoder_layers_[layer].cross_proj_bias;
}

// Decoder2 layer accessors (for dec_blocks2)
id<MTLBuffer> ViTBuffers::decoder2_norm1_weight(int layer) const {
    return decoder_layers2_[layer].norm1_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_norm1_bias(int layer) const {
    return decoder_layers2_[layer].norm1_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_norm2_weight(int layer) const {
    return decoder_layers2_[layer].norm2_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_norm2_bias(int layer) const {
    return decoder_layers2_[layer].norm2_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_qkv_weight(int layer) const {
    return decoder_layers2_[layer].qkv_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_qkv_bias(int layer) const {
    return decoder_layers2_[layer].qkv_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_proj_weight(int layer) const {
    return decoder_layers2_[layer].proj_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_proj_bias(int layer) const {
    return decoder_layers2_[layer].proj_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_mlp_fc1_weight(int layer) const {
    return decoder_layers2_[layer].mlp_fc1_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_mlp_fc1_bias(int layer) const {
    return decoder_layers2_[layer].mlp_fc1_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_mlp_fc2_weight(int layer) const {
    return decoder_layers2_[layer].mlp_fc2_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_mlp_fc2_bias(int layer) const {
    return decoder_layers2_[layer].mlp_fc2_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_cross_norm_weight(int layer) const {
    return decoder_layers2_[layer].cross_norm_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_cross_norm_bias(int layer) const {
    return decoder_layers2_[layer].cross_norm_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_cross_q_weight(int layer) const {
    return decoder_layers2_[layer].cross_q_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_cross_q_bias(int layer) const {
    return decoder_layers2_[layer].cross_q_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_cross_kv_weight(int layer) const {
    return decoder_layers2_[layer].cross_kv_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_cross_kv_bias(int layer) const {
    return decoder_layers2_[layer].cross_kv_bias;
}
id<MTLBuffer> ViTBuffers::decoder2_cross_proj_weight(int layer) const {
    return decoder_layers2_[layer].cross_proj_weight;
}
id<MTLBuffer> ViTBuffers::decoder2_cross_proj_bias(int layer) const {
    return decoder_layers2_[layer].cross_proj_bias;
}

}  // namespace metal
}  // namespace mast3r