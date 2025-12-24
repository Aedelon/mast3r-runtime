/**
 * Python bindings for CUDA backend.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuda_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cuda, m) {
    m.doc() = "MASt3R CUDA backend (cuBLAS + custom kernels)";

    // Availability check
    m.def("is_available", &mast3r::CUDAEngine::is_available,
          "Check if CUDA is available");

    m.def("get_device_name", &mast3r::CUDAEngine::get_device_name,
          "Get CUDA device name");

    m.def("get_compute_capability", []() {
        auto cc = mast3r::CUDAEngine::get_compute_capability();
        return py::make_tuple(cc.first, cc.second);
    }, "Get CUDA compute capability (major, minor)");

    // Engine class
    py::class_<mast3r::CUDAEngine>(m, "CUDAEngine")
        .def(py::init<const std::string&, int, const std::string&, int>(),
             py::arg("variant"),
             py::arg("resolution"),
             py::arg("precision"),
             py::arg("num_threads") = 4)
        .def("load_weights", &mast3r::CUDAEngine::load_weights)
        .def("warmup", &mast3r::CUDAEngine::warmup,
             py::arg("iterations") = 3)
        .def("infer", [](mast3r::CUDAEngine& self,
                         py::array_t<uint8_t> img1,
                         py::array_t<uint8_t> img2) {
            auto buf1 = img1.request();
            auto buf2 = img2.request();

            if (buf1.ndim != 3 || buf2.ndim != 3) {
                throw std::runtime_error("Images must be 3D arrays [H, W, C]");
            }

            return self.infer(
                static_cast<uint8_t*>(buf1.ptr),
                buf1.shape[0], buf1.shape[1],
                static_cast<uint8_t*>(buf2.ptr),
                buf2.shape[0], buf2.shape[1]
            );
        })
        .def("match", [](mast3r::CUDAEngine& self,
                         py::array_t<float> desc1,
                         py::array_t<float> desc2,
                         std::optional<py::array_t<float>> conf1,
                         std::optional<py::array_t<float>> conf2,
                         int top_k,
                         bool reciprocal,
                         float conf_threshold) {
            auto buf1 = desc1.request();
            auto buf2 = desc2.request();

            int h = buf1.shape[0];
            int w = buf1.shape[1];
            int d = buf1.shape[2];

            const float* c1 = conf1 ? static_cast<float*>(conf1->request().ptr) : nullptr;
            const float* c2 = conf2 ? static_cast<float*>(conf2->request().ptr) : nullptr;

            return self.match(
                static_cast<float*>(buf1.ptr),
                static_cast<float*>(buf2.ptr),
                h, w, d, c1, c2,
                top_k, reciprocal, conf_threshold
            );
        },
        py::arg("desc1"),
        py::arg("desc2"),
        py::arg("conf1") = py::none(),
        py::arg("conf2") = py::none(),
        py::arg("top_k") = 512,
        py::arg("reciprocal") = true,
        py::arg("conf_threshold") = 0.5f)
        .def("is_ready", &mast3r::CUDAEngine::is_ready)
        .def("name", &mast3r::CUDAEngine::name);

    // Result types
    py::class_<mast3r::InferenceResult>(m, "InferenceResult")
        .def_readonly("pts3d_1", &mast3r::InferenceResult::pts3d_1)
        .def_readonly("pts3d_2", &mast3r::InferenceResult::pts3d_2)
        .def_readonly("desc_1", &mast3r::InferenceResult::desc_1)
        .def_readonly("desc_2", &mast3r::InferenceResult::desc_2)
        .def_readonly("conf_1", &mast3r::InferenceResult::conf_1)
        .def_readonly("conf_2", &mast3r::InferenceResult::conf_2)
        .def_readonly("timing_ms", &mast3r::InferenceResult::timing_ms);

    py::class_<mast3r::MatchResult>(m, "MatchResult")
        .def_readonly("indices_1", &mast3r::MatchResult::indices_1)
        .def_readonly("indices_2", &mast3r::MatchResult::indices_2)
        .def_readonly("scores", &mast3r::MatchResult::scores)
        .def_readonly("timing_ms", &mast3r::MatchResult::timing_ms);
}
