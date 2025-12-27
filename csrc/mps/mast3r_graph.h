// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0 License.
// Complete MASt3R model using MPSGraph with native SDPA

#pragma once

#include "../common/types.hpp"
#include <memory>
#include <string>

namespace mast3r {
namespace mpsgraph {

/**
 * Complete MASt3R inference using MPSGraph.
 * Uses Apple's native SDPA (macOS 15+) for ~10-15x speedup vs manual Metal kernels.
 */
class MASt3RGraph {
public:
    MASt3RGraph();
    ~MASt3RGraph();

    // Non-copyable
    MASt3RGraph(const MASt3RGraph&) = delete;
    MASt3RGraph& operator=(const MASt3RGraph&) = delete;

    /**
     * Initialize the graph for a specific model.
     * @param spec Model specification (variant, dimensions, etc.)
     * @return true if initialization succeeded
     */
    bool initialize(const ModelSpec& spec);

    /**
     * Load weights from safetensors file.
     * @param path Path to unified.safetensors
     * @return true if weights loaded successfully
     */
    bool load_weights(const std::string& path);

    /**
     * Run inference on a pair of images.
     * @param img1 First image (RGB, HWC format)
     * @param img2 Second image (RGB, HWC format)
     * @param width Image width
     * @param height Image height
     * @return Inference results (pointmaps, confidence, descriptors)
     */
    InferenceResult infer(const uint8_t* img1, const uint8_t* img2,
                          int width, int height);

    /**
     * Run inference with pre-normalized float images.
     * @param img1 First image normalized to [-1, 1]
     * @param img2 Second image normalized to [-1, 1]
     * @param width Image width
     * @param height Image height
     * @return Inference results
     */
    InferenceResult infer_normalized(const float* img1, const float* img2,
                                     int width, int height);

    /**
     * Check if MPSGraph SDPA is available.
     * Requires macOS 15.0+ and Apple Silicon.
     */
    static bool is_available();

    /**
     * Get timing breakdown for last inference.
     * @return Map of stage name to milliseconds
     */
    std::unordered_map<std::string, double> get_timings() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mpsgraph
}  // namespace mast3r
