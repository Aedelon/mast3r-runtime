// MASt3R Runtime - MPS Backend Python Bindings
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mpsgraph_context.hpp"
#include "mpsgraph_engine.hpp"

namespace py = pybind11;
using namespace mast3r;
using namespace mast3r::mpsgraph;

// Helper to convert numpy array to ImageView
ImageView numpy_to_image_view(py::array_t<uint8_t> arr) {
    auto buf = arr.request();

    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("Expected [H, W, 3] uint8 array");
    }

    ImageView view;
    view.data = static_cast<const uint8_t*>(buf.ptr);
    view.height = static_cast<int>(buf.shape[0]);
    view.width = static_cast<int>(buf.shape[1]);
    view.channels = 3;

    return view;
}

// Python wrapper for InferenceResult
struct PyInferenceResult {
    py::array_t<float> pts3d_1;
    py::array_t<float> pts3d_2;
    py::array_t<float> desc_1;
    py::array_t<float> desc_2;
    py::array_t<float> conf_1;
    py::array_t<float> conf_2;
    py::dict timing;
};

// Python wrapper for MatchResult
struct PyMatchResult {
    py::array_t<int64_t> idx_1;
    py::array_t<int64_t> idx_2;
    py::array_t<float> pts2d_1;
    py::array_t<float> pts2d_2;
    py::array_t<float> pts3d_1;
    py::array_t<float> pts3d_2;
    py::array_t<float> confidence;
    py::dict timing;
};

// Python wrapper for RetrievalResult
struct PyRetrievalResult {
    py::array_t<float> features;   // [N, D] whitened features
    py::array_t<float> attention;  // [N] L2 attention scores
    py::dict timing;
};

// Python wrapper for GPUTensor - lazy copy pattern
class PyGPUTensor {
public:
    PyGPUTensor(GPUTensor&& tensor) : tensor_(std::move(tensor)) {}

    py::tuple shape() const {
        const auto& s = tensor_.shape();
        py::tuple result(s.size());
        for (size_t i = 0; i < s.size(); i++) {
            result[i] = s[i];
        }
        return result;
    }

    size_t numel() const { return tensor_.numel(); }
    size_t nbytes() const { return tensor_.nbytes(); }
    size_t ndim() const { return tensor_.ndim(); }
    bool is_valid() const { return tensor_.is_valid(); }

    // The expensive copy operation - called explicitly by user
    py::array_t<float> numpy() const {
        const auto& s = tensor_.shape();
        std::vector<py::ssize_t> py_shape(s.begin(), s.end());
        py::array_t<float> result(py_shape);
        tensor_.copy_to(result.mutable_data());
        return result;
    }

private:
    GPUTensor tensor_;
};

// Python wrapper for GPUInferenceResult
struct PyGPUInferenceResult {
    std::shared_ptr<PyGPUTensor> pts3d_1;
    std::shared_ptr<PyGPUTensor> pts3d_2;
    std::shared_ptr<PyGPUTensor> conf_1;
    std::shared_ptr<PyGPUTensor> conf_2;
    std::shared_ptr<PyGPUTensor> desc_1;
    std::shared_ptr<PyGPUTensor> desc_2;
    py::dict timing;
};

// Python wrapper class
class PyMPSEngine {
public:
    PyMPSEngine(
        const std::string& variant,
        int resolution,
        const std::string& precision,
        int num_threads
    ) {
        RuntimeConfig config;
        config.resolution = resolution;
        config.num_threads = num_threads;

        if (variant == "dune_vit_small_336") {
            config.variant = ModelVariant::DUNE_VIT_SMALL_336;
        } else if (variant == "dune_vit_small_448") {
            config.variant = ModelVariant::DUNE_VIT_SMALL_448;
        } else if (variant == "dune_vit_base_336") {
            config.variant = ModelVariant::DUNE_VIT_BASE_336;
        } else if (variant == "dune_vit_base_448") {
            config.variant = ModelVariant::DUNE_VIT_BASE_448;
        } else if (variant == "mast3r_vit_large") {
            config.variant = ModelVariant::MAST3R_VIT_LARGE;
        } else {
            throw std::runtime_error("Unknown variant: " + variant);
        }

        if (precision == "fp32") {
            config.precision = Precision::FP32;
        } else if (precision == "fp16") {
            config.precision = Precision::FP16;
        } else if (precision == "int8") {
            config.precision = Precision::INT8;
        }

        engine_ = std::make_unique<MPSGraphEngine>(config);
    }

    void load(const std::string& model_path) {
        engine_->load(model_path);
    }

    bool is_ready() const {
        return engine_->is_ready();
    }

    std::string name() const {
        return engine_->name();
    }

    void warmup(int num_iterations) {
        engine_->warmup(num_iterations);
    }

    PyInferenceResult infer(
        py::array_t<uint8_t> img1,
        py::array_t<uint8_t> img2
    ) {
        auto view1 = numpy_to_image_view(img1);
        auto view2 = numpy_to_image_view(img2);

        auto result = engine_->infer(view1, view2);

        PyInferenceResult py_result;

        const int H = result.height;
        const int W = result.width;
        const int D = result.desc_dim;

        py_result.pts3d_1 = py::array_t<float>({H, W, 3});
        py_result.pts3d_2 = py::array_t<float>({H, W, 3});
        py_result.desc_1 = py::array_t<float>({H, W, D});
        py_result.desc_2 = py::array_t<float>({H, W, D});
        py_result.conf_1 = py::array_t<float>({H, W});
        py_result.conf_2 = py::array_t<float>({H, W});

        std::memcpy(py_result.pts3d_1.mutable_data(), result.pts3d_1, H * W * 3 * sizeof(float));
        std::memcpy(py_result.pts3d_2.mutable_data(), result.pts3d_2, H * W * 3 * sizeof(float));
        std::memcpy(py_result.desc_1.mutable_data(), result.desc_1, H * W * D * sizeof(float));
        std::memcpy(py_result.desc_2.mutable_data(), result.desc_2, H * W * D * sizeof(float));
        std::memcpy(py_result.conf_1.mutable_data(), result.conf_1, H * W * sizeof(float));
        std::memcpy(py_result.conf_2.mutable_data(), result.conf_2, H * W * sizeof(float));

        // Clean up C++ allocations
        delete[] result.pts3d_1;
        delete[] result.pts3d_2;
        delete[] result.desc_1;
        delete[] result.desc_2;
        delete[] result.conf_1;
        delete[] result.conf_2;

        py_result.timing["preprocess_ms"] = result.preprocess_ms;
        py_result.timing["inference_ms"] = result.inference_ms;
        py_result.timing["total_ms"] = result.total_ms;

        return py_result;
    }

    // GPU tensor version - data stays on GPU until .numpy() is called
    PyGPUInferenceResult infer_gpu(
        py::array_t<uint8_t> img1,
        py::array_t<uint8_t> img2
    ) {
        auto view1 = numpy_to_image_view(img1);
        auto view2 = numpy_to_image_view(img2);

        auto result = engine_->infer_gpu(view1, view2);

        PyGPUInferenceResult py_result;

        // Wrap GPU tensors - NO COPY here, data stays on GPU
        py_result.pts3d_1 = std::make_shared<PyGPUTensor>(std::move(result.pts3d_1));
        py_result.pts3d_2 = std::make_shared<PyGPUTensor>(std::move(result.pts3d_2));
        py_result.conf_1 = std::make_shared<PyGPUTensor>(std::move(result.conf_1));
        py_result.conf_2 = std::make_shared<PyGPUTensor>(std::move(result.conf_2));
        py_result.desc_1 = std::make_shared<PyGPUTensor>(std::move(result.desc_1));
        py_result.desc_2 = std::make_shared<PyGPUTensor>(std::move(result.desc_2));

        py_result.timing["preprocess_ms"] = result.preprocess_ms;
        py_result.timing["inference_ms"] = result.inference_ms;
        py_result.timing["total_ms"] = result.total_ms;

        return py_result;
    }

    PyMatchResult match(
        py::array_t<float> desc_1,
        py::array_t<float> desc_2,
        int top_k,
        bool reciprocal,
        float confidence_threshold
    ) {
        auto buf1 = desc_1.request();
        auto buf2 = desc_2.request();

        if (buf1.ndim != 3 || buf2.ndim != 3) {
            throw std::runtime_error("Expected [H, W, D] arrays");
        }

        const int H = static_cast<int>(buf1.shape[0]);
        const int W = static_cast<int>(buf1.shape[1]);
        const int D = static_cast<int>(buf1.shape[2]);

        MatchingConfig config;
        config.top_k = top_k;
        config.reciprocal = reciprocal;
        config.confidence_threshold = confidence_threshold;

        auto result = engine_->match(
            static_cast<const float*>(buf1.ptr),
            static_cast<const float*>(buf2.ptr),
            H, W, D, config
        );

        PyMatchResult py_result;
        const int N = static_cast<int>(result.num_matches());

        py_result.idx_1 = py::array_t<int64_t>(N);
        py_result.idx_2 = py::array_t<int64_t>(N);
        py_result.pts2d_1 = py::array_t<float>({N, 2});
        py_result.pts2d_2 = py::array_t<float>({N, 2});
        py_result.pts3d_1 = py::array_t<float>({N, 3});
        py_result.pts3d_2 = py::array_t<float>({N, 3});
        py_result.confidence = py::array_t<float>(N);

        if (N > 0) {
            std::memcpy(py_result.idx_1.mutable_data(), result.idx_1.data(), N * sizeof(int64_t));
            std::memcpy(py_result.idx_2.mutable_data(), result.idx_2.data(), N * sizeof(int64_t));
            std::memcpy(py_result.pts2d_1.mutable_data(), result.pts2d_1.data(), N * 2 * sizeof(float));
            std::memcpy(py_result.pts2d_2.mutable_data(), result.pts2d_2.data(), N * 2 * sizeof(float));
            std::memcpy(py_result.confidence.mutable_data(), result.confidence.data(), N * sizeof(float));
        }

        py_result.timing["match_ms"] = result.match_ms;

        return py_result;
    }

    // Weight sharing mode (requires main model loaded)
    void load_retrieval(const std::string& retrieval_path) {
        engine_->load_retrieval(retrieval_path);
    }

    // Standalone mode (encoder + whitening only, no main model needed)
    void load_retrieval_standalone(const std::string& model_path, const std::string& retrieval_path) {
        engine_->load_retrieval(model_path, retrieval_path);
    }

    bool is_retrieval_ready() const {
        return engine_->is_retrieval_ready();
    }

    bool is_retrieval_standalone() const {
        return engine_->is_retrieval_standalone();
    }

    PyRetrievalResult encode_retrieval(py::array_t<uint8_t> img) {
        auto view = numpy_to_image_view(img);
        auto result = engine_->encode_retrieval(view);

        PyRetrievalResult py_result;

        const int N = result.num_patches;
        const int D = result.feature_dim;

        py_result.features = py::array_t<float>({N, D});
        py_result.attention = py::array_t<float>(N);

        std::memcpy(py_result.features.mutable_data(), result.features, N * D * sizeof(float));
        std::memcpy(py_result.attention.mutable_data(), result.attention, N * sizeof(float));

        // Clean up C++ allocations
        delete[] result.features;
        delete[] result.attention;

        py_result.timing["preprocess_ms"] = result.preprocess_ms;
        py_result.timing["encoder_ms"] = result.encoder_ms;
        py_result.timing["whiten_ms"] = result.whiten_ms;
        py_result.timing["total_ms"] = result.total_ms;

        return py_result;
    }

    // Batch inference with pipelining (encoder[N+1] || decoder[N])
    std::vector<PyGPUInferenceResult> infer_batch_pipelined(py::list images) {
        std::vector<ImageView> views;
        views.reserve(py::len(images));

        for (auto& item : images) {
            views.push_back(numpy_to_image_view(item.cast<py::array_t<uint8_t>>()));
        }

        auto results = engine_->infer_batch_pipelined(views);

        std::vector<PyGPUInferenceResult> py_results;
        py_results.reserve(results.size());

        for (auto& result : results) {
            PyGPUInferenceResult py_result;

            py_result.pts3d_1 = std::make_shared<PyGPUTensor>(std::move(result.pts3d_1));
            py_result.pts3d_2 = std::make_shared<PyGPUTensor>(std::move(result.pts3d_2));
            py_result.conf_1 = std::make_shared<PyGPUTensor>(std::move(result.conf_1));
            py_result.conf_2 = std::make_shared<PyGPUTensor>(std::move(result.conf_2));
            py_result.desc_1 = std::make_shared<PyGPUTensor>(std::move(result.desc_1));
            py_result.desc_2 = std::make_shared<PyGPUTensor>(std::move(result.desc_2));

            py_result.timing["preprocess_ms"] = result.preprocess_ms;
            py_result.timing["inference_ms"] = result.inference_ms;
            py_result.timing["total_ms"] = result.total_ms;

            py_results.push_back(std::move(py_result));
        }

        return py_results;
    }

    // Batch retrieval with async pipelining
    std::vector<PyRetrievalResult> encode_retrieval_batch(py::list images) {
        std::vector<ImageView> views;
        views.reserve(py::len(images));

        for (auto& item : images) {
            views.push_back(numpy_to_image_view(item.cast<py::array_t<uint8_t>>()));
        }

        auto results = engine_->encode_retrieval_batch(views);

        std::vector<PyRetrievalResult> py_results;
        py_results.reserve(results.size());

        for (auto& result : results) {
            PyRetrievalResult py_result;

            const int N = result.num_patches;
            const int D = result.feature_dim;

            py_result.features = py::array_t<float>({N, D});
            py_result.attention = py::array_t<float>(N);

            std::memcpy(py_result.features.mutable_data(), result.features, N * D * sizeof(float));
            std::memcpy(py_result.attention.mutable_data(), result.attention, N * sizeof(float));

            delete[] result.features;
            delete[] result.attention;

            py_result.timing["encoder_ms"] = result.encoder_ms;
            py_result.timing["whiten_ms"] = result.whiten_ms;
            py_result.timing["total_ms"] = result.total_ms;

            py_results.push_back(std::move(py_result));
        }

        return py_results;
    }

private:
    std::unique_ptr<MPSGraphEngine> engine_;
};


PYBIND11_MODULE(_mps, m) {
    m.doc() = "MASt3R Runtime MPS Backend (Apple Silicon with MPSGraph SDPA)";

    py::class_<PyInferenceResult>(m, "InferenceResult")
        .def_readonly("pts3d_1", &PyInferenceResult::pts3d_1)
        .def_readonly("pts3d_2", &PyInferenceResult::pts3d_2)
        .def_readonly("desc_1", &PyInferenceResult::desc_1)
        .def_readonly("desc_2", &PyInferenceResult::desc_2)
        .def_readonly("conf_1", &PyInferenceResult::conf_1)
        .def_readonly("conf_2", &PyInferenceResult::conf_2)
        .def_readonly("timing", &PyInferenceResult::timing);

    py::class_<PyMatchResult>(m, "MatchResult")
        .def_readonly("idx_1", &PyMatchResult::idx_1)
        .def_readonly("idx_2", &PyMatchResult::idx_2)
        .def_readonly("pts2d_1", &PyMatchResult::pts2d_1)
        .def_readonly("pts2d_2", &PyMatchResult::pts2d_2)
        .def_readonly("pts3d_1", &PyMatchResult::pts3d_1)
        .def_readonly("pts3d_2", &PyMatchResult::pts3d_2)
        .def_readonly("confidence", &PyMatchResult::confidence)
        .def_readonly("timing", &PyMatchResult::timing);

    py::class_<PyRetrievalResult>(m, "RetrievalResult")
        .def_readonly("features", &PyRetrievalResult::features)
        .def_readonly("attention", &PyRetrievalResult::attention)
        .def_readonly("timing", &PyRetrievalResult::timing);

    // GPU Tensor - lazy copy pattern
    py::class_<PyGPUTensor, std::shared_ptr<PyGPUTensor>>(m, "GPUTensor")
        .def_property_readonly("shape", &PyGPUTensor::shape)
        .def_property_readonly("numel", &PyGPUTensor::numel)
        .def_property_readonly("nbytes", &PyGPUTensor::nbytes)
        .def_property_readonly("ndim", &PyGPUTensor::ndim)
        .def_property_readonly("is_valid", &PyGPUTensor::is_valid)
        .def("numpy", &PyGPUTensor::numpy,
             "Copy data from GPU to CPU and return as NumPy array. "
             "This is the expensive operation - call only when needed.");

    // GPU Inference Result - holds GPU tensor handles
    py::class_<PyGPUInferenceResult>(m, "GPUInferenceResult")
        .def_readonly("pts3d_1", &PyGPUInferenceResult::pts3d_1)
        .def_readonly("pts3d_2", &PyGPUInferenceResult::pts3d_2)
        .def_readonly("conf_1", &PyGPUInferenceResult::conf_1)
        .def_readonly("conf_2", &PyGPUInferenceResult::conf_2)
        .def_readonly("desc_1", &PyGPUInferenceResult::desc_1)
        .def_readonly("desc_2", &PyGPUInferenceResult::desc_2)
        .def_readonly("timing", &PyGPUInferenceResult::timing);

    py::class_<PyMPSEngine>(m, "MPSEngine")
        .def(py::init<const std::string&, int, const std::string&, int>(),
             py::arg("variant") = "mast3r_vit_large",
             py::arg("resolution") = 512,
             py::arg("precision") = "fp32",
             py::arg("num_threads") = 4)
        .def("load", &PyMPSEngine::load)
        .def("is_ready", &PyMPSEngine::is_ready)
        .def("name", &PyMPSEngine::name)
        .def("warmup", &PyMPSEngine::warmup, py::arg("num_iterations") = 3)
        .def("infer", &PyMPSEngine::infer,
             "Run inference and copy results to CPU (legacy). "
             "Use infer_gpu() for lazy-copy pattern.")
        .def("infer_gpu", &PyMPSEngine::infer_gpu,
             "Run inference returning GPU tensor handles (fast). "
             "Data stays on GPU until .numpy() is called on each tensor. "
             "This eliminates ~420ms of copy overhead when data isn't needed immediately.")
        .def("infer_batch_pipelined", &PyMPSEngine::infer_batch_pipelined,
             py::arg("images"),
             "Batch inference with encoder/decoder pipelining for higher throughput. "
             "Uses async execution with double-buffering for optimal GPU utilization.")
        .def("match", &PyMPSEngine::match,
             py::arg("desc_1"),
             py::arg("desc_2"),
             py::arg("top_k") = 512,
             py::arg("reciprocal") = true,
             py::arg("confidence_threshold") = 0.5f)
        .def("load_retrieval", &PyMPSEngine::load_retrieval,
             py::arg("retrieval_path"),
             "Load retrieval weights (requires main model loaded for weight sharing)")
        .def("load_retrieval_standalone", &PyMPSEngine::load_retrieval_standalone,
             py::arg("model_path"), py::arg("retrieval_path"),
             "Load retrieval in standalone mode (encoder + whitening only, no main model needed)")
        .def("is_retrieval_ready", &PyMPSEngine::is_retrieval_ready)
        .def("is_retrieval_standalone", &PyMPSEngine::is_retrieval_standalone)
        .def("encode_retrieval", &PyMPSEngine::encode_retrieval)
        .def("encode_retrieval_batch", &PyMPSEngine::encode_retrieval_batch,
             py::arg("images"),
             "Batch retrieval with async pipelining for higher throughput");

    // Module info
    m.def("is_available", []() {
        return MPSGraphContext::is_available();
    });
    m.def("get_name", []() {
        return "MPSGraph SDPA (Apple Silicon)";
    });

    // Context info
    m.def("get_device_name", []() {
        if (@available(macOS 15.0, *)) {
            return MPSGraphContext::shared()->device_name();
        }
        return std::string("unavailable");
    });
    m.def("get_recommended_working_set_size", []() {
        if (@available(macOS 15.0, *)) {
            return MPSGraphContext::shared()->recommended_working_set_size();
        }
        return size_t(0);
    });
    m.def("get_max_buffer_length", []() {
        if (@available(macOS 15.0, *)) {
            return MPSGraphContext::shared()->max_buffer_length();
        }
        return size_t(0);
    });
    m.def("get_context_info", []() {
        py::dict info;
        if (@available(macOS 15.0, *)) {
            auto ctx = MPSGraphContext::shared();
            info["device_name"] = ctx->device_name();
            info["recommended_working_set_size"] = ctx->recommended_working_set_size();
            info["max_buffer_length"] = ctx->max_buffer_length();
            info["buffer_pool_count"] = ctx->buffer_pool().pooled_count();
            info["buffer_pool_bytes"] = ctx->buffer_pool().total_bytes();
        } else {
            info["error"] = "macOS 15.0+ required";
        }
        return info;
    });
    m.def("synchronize", []() {
        if (@available(macOS 15.0, *)) {
            MPSGraphContext::shared()->synchronize();
        }
    });
    m.def("clear_buffer_pool", []() {
        if (@available(macOS 15.0, *)) {
            MPSGraphContext::shared()->buffer_pool().clear();
        }
    });
}
