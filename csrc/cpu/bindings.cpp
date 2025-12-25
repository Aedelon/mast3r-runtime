// MASt3R Runtime - CPU Backend Python Bindings
// Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cpu_engine.hpp"

namespace py = pybind11;
using namespace mast3r;
using namespace mast3r::cpu;

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

// Python wrapper class
class PyCPUEngine {
public:
    PyCPUEngine(
        const std::string& variant,
        int resolution,
        const std::string& precision,
        int num_threads
    ) {
        RuntimeConfig config;
        config.resolution = resolution;
        config.num_threads = num_threads;

        // Parse variant
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

        // Parse precision
        if (precision == "fp32") {
            config.precision = Precision::FP32;
        } else if (precision == "fp16") {
            config.precision = Precision::FP16;
        } else if (precision == "int8") {
            config.precision = Precision::INT8;
        }

        engine_ = std::make_unique<CPUEngine>(config);
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

        // Copy data to numpy arrays
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
            std::memcpy(py_result.pts3d_1.mutable_data(), result.pts3d_1.data(), N * 3 * sizeof(float));
            std::memcpy(py_result.pts3d_2.mutable_data(), result.pts3d_2.data(), N * 3 * sizeof(float));
            std::memcpy(py_result.confidence.mutable_data(), result.confidence.data(), N * sizeof(float));
        }

        py_result.timing["match_ms"] = result.match_ms;

        return py_result;
    }

private:
    std::unique_ptr<CPUEngine> engine_;
};


PYBIND11_MODULE(_cpu, m) {
    m.doc() = "MASt3R Runtime CPU Backend";

    // InferenceResult
    py::class_<PyInferenceResult>(m, "InferenceResult")
        .def_readonly("pts3d_1", &PyInferenceResult::pts3d_1)
        .def_readonly("pts3d_2", &PyInferenceResult::pts3d_2)
        .def_readonly("desc_1", &PyInferenceResult::desc_1)
        .def_readonly("desc_2", &PyInferenceResult::desc_2)
        .def_readonly("conf_1", &PyInferenceResult::conf_1)
        .def_readonly("conf_2", &PyInferenceResult::conf_2)
        .def_readonly("timing", &PyInferenceResult::timing);

    // MatchResult
    py::class_<PyMatchResult>(m, "MatchResult")
        .def_readonly("idx_1", &PyMatchResult::idx_1)
        .def_readonly("idx_2", &PyMatchResult::idx_2)
        .def_readonly("pts2d_1", &PyMatchResult::pts2d_1)
        .def_readonly("pts2d_2", &PyMatchResult::pts2d_2)
        .def_readonly("pts3d_1", &PyMatchResult::pts3d_1)
        .def_readonly("pts3d_2", &PyMatchResult::pts3d_2)
        .def_readonly("confidence", &PyMatchResult::confidence)
        .def_readonly("timing", &PyMatchResult::timing);

    // CPUEngine
    py::class_<PyCPUEngine>(m, "CPUEngine")
        .def(py::init<const std::string&, int, const std::string&, int>(),
             py::arg("variant") = "dune_vit_small_336",
             py::arg("resolution") = 336,
             py::arg("precision") = "fp16",
             py::arg("num_threads") = 4)
        .def("load", &PyCPUEngine::load)
        .def("is_ready", &PyCPUEngine::is_ready)
        .def("name", &PyCPUEngine::name)
        .def("warmup", &PyCPUEngine::warmup, py::arg("num_iterations") = 3)
        .def("infer", &PyCPUEngine::infer)
        .def("match", &PyCPUEngine::match,
             py::arg("desc_1"),
             py::arg("desc_2"),
             py::arg("top_k") = 512,
             py::arg("reciprocal") = true,
             py::arg("confidence_threshold") = 0.5f);

    // Module info
    m.def("is_available", []() { return true; });
    m.def("get_name", []() { return "CPU"; });
}