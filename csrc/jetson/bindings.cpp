/**
 * Python bindings for Jetson backend.
 *
 * Copyright 2024 Delanoe Pirard / Aedelon. Apache 2.0.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "jetson_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_jetson, m) {
    m.doc() = "MASt3R Jetson backend (TensorRT + DLA)";

    // Availability checks
    m.def("is_available", &mast3r::JetsonEngine::is_available,
          "Check if Jetson/TensorRT is available");

    m.def("get_device_name", &mast3r::JetsonEngine::get_device_name,
          "Get Jetson device name");

    m.def("get_dla_count", &mast3r::JetsonEngine::get_dla_count,
          "Get number of DLA cores");

    m.def("has_dla", &mast3r::JetsonEngine::has_dla,
          "Check if DLA is available");

    // Engine class
    py::class_<mast3r::JetsonEngine>(m, "JetsonEngine")
        .def(py::init<const std::string&, int, const std::string&, int>(),
             py::arg("variant"),
             py::arg("resolution"),
             py::arg("precision"),
             py::arg("num_threads") = 4)
        .def("load_weights", &mast3r::JetsonEngine::load_weights)
        .def("load_engine", &mast3r::JetsonEngine::load_engine,
             py::arg("engine_path"))
        .def("build_engine", &mast3r::JetsonEngine::build_engine,
             py::arg("onnx_path"),
             py::arg("engine_path"))
        .def("warmup", &mast3r::JetsonEngine::warmup,
             py::arg("iterations") = 3)
        .def("set_dla_core", &mast3r::JetsonEngine::set_dla_core,
             py::arg("core"),
             "Set DLA core (-1 for GPU only)")
        .def("get_dla_core", &mast3r::JetsonEngine::get_dla_core)
        .def("infer", [](mast3r::JetsonEngine& self,
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
        .def("match", [](mast3r::JetsonEngine& self,
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
        .def("is_ready", &mast3r::JetsonEngine::is_ready)
        .def("name", &mast3r::JetsonEngine::name);

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
