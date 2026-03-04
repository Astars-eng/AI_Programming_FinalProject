#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "cnn_module.h"

namespace py = pybind11;

// 辅助函数：将 numpy 数组复制到 GPU
float* numpy_to_gpu(py::array_t<float> b) {
    py::buffer_info info = b.request();
    float* d_ptr;
    cudaMalloc(&d_ptr, info.size * sizeof(float));
    cudaMemcpy(d_ptr, info.ptr, info.size * sizeof(float), cudaMemcpyHostToDevice);
    return d_ptr;
}

// 辅助函数：将 GPU 数据复制回 numpy
py::array_t<float> gpu_to_numpy(float* d_ptr, std::vector<ssize_t> shape) {
    ssize_t size = 1;
    for (auto s : shape) size *= s;
    auto result = py::array_t<float>(shape);
    cudaMemcpy(result.mutable_data(), d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

PYBIND11_MODULE(cuda_net, m) {
    m.def("init_context", []() { init_cublas(); });
    m.def("destroy_context", []() { destroy_cublas(); });

    // 显存管理
    m.def("alloc_gpu", [](size_t size) {
        float* ptr;
        cudaMalloc(&ptr, size * sizeof(float));
        cudaMemset(ptr, 0, size * sizeof(float));
        return reinterpret_cast<uintptr_t>(ptr);
    });
    m.def("free_gpu", [](uintptr_t ptr) { cudaFree(reinterpret_cast<void*>(ptr)); });
    m.def("to_gpu", [](py::array_t<float> b) {
        return reinterpret_cast<uintptr_t>(numpy_to_gpu(b));
    });
    m.def("to_cpu", [](uintptr_t ptr, std::vector<ssize_t> shape) {
        return gpu_to_numpy(reinterpret_cast<float*>(ptr), shape);
    });

    // 算子封装
    m.def("forward_conv", [](uintptr_t in, uintptr_t out, uintptr_t w, uintptr_t b, int N, int Ci, int H, int W, int Co) {
        forward_conv((float*)in, (float*)out, (float*)w, (float*)b, N, Ci, H, W, Co);
    });

    m.def("backward_conv", [](uintptr_t in, uintptr_t w, uintptr_t go, uintptr_t gi, uintptr_t gw, uintptr_t gb, int N, int Ci, int H, int W, int Co) {
        backward_conv((float*)in, (float*)w, (float*)go, (float*)gi, (float*)gw, (float*)gb, N, Ci, H, W, Co);
    });

    m.def("forward_fc", [](uintptr_t in, uintptr_t out, uintptr_t w, uintptr_t b, int B, int I, int O) {
        forward_fc((float*)in, (float*)out, (float*)w, (float*)b, B, I, O);
    });

    m.def("backward_fc", [](uintptr_t in, uintptr_t out, uintptr_t w, uintptr_t b, int B, int I, int O, uintptr_t go, uintptr_t gi, uintptr_t gw, uintptr_t gb) {
        backward_fc((float*)in, (float*)out, (float*)w, (float*)b, B, I, O, (float*)go, (float*)gi, (float*)gw, (float*)gb);
    });

    m.def("forward_relu", [](uintptr_t in, uintptr_t out, int size) {
        forward_relu_gpu((float*)in, (float*)out, size);
    });

    m.def("backward_relu", [](uintptr_t in, uintptr_t go, uintptr_t gi, int size) {
        backward_relu_gpu((float*)in, (float*)go, (float*)gi, size);
    });

    m.def("forward_pool", [](uintptr_t in, uintptr_t out, uintptr_t mask, int N, int C, int H, int W) {
        forward_max_pool((float*)in, (float*)out, (float*)mask, N, C, H, W);
    });

    m.def("backward_pool", [](uintptr_t go, uintptr_t mask, uintptr_t gi, int N, int C, int H, int W) {
        backward_max_pool((float*)go, (float*)mask, (float*)gi, N, C, H, W);
    });

    m.def("cross_entropy", [](uintptr_t logits, py::array_t<int> labels, uintptr_t grad, int B, int F) {
        float loss = 0;
        cross_entropy_softmax((float*)logits, (int*)labels.data(), &loss, (float*)grad, B, F);
        return loss;
    });

    m.def("forward_dropout", [](uintptr_t in, uintptr_t out, uintptr_t mask, int size, float prob) {
        forward_dropout((float*)in, (float*)out, (float*)mask, size, prob);
    });

    m.def("backward_dropout", [](uintptr_t go, uintptr_t gi, uintptr_t mask, int size, float prob) {
        backward_dropout((float*)go, (float*)gi, (float*)mask, size, prob);
    });

    // BN
    m.def("forward_bn_train", [](uintptr_t in, uintptr_t out, uintptr_t sm, uintptr_t sv, uintptr_t rm, uintptr_t rv, uintptr_t g, uintptr_t b, int N, int C, int H, int W, float mom, float eps) {
        forward_bn_train((float*)in, (float*)out, (float*)sm, (float*)sv, (float*)rm, (float*)rv, (float*)g, (float*)b, N, C, H, W, mom, eps);
    });

    m.def("forward_bn_test", [](uintptr_t in, uintptr_t out, uintptr_t rm, uintptr_t rv, uintptr_t g, uintptr_t b, int N, int C, int H, int W, float eps) {
        forward_bn_test((float*)in, (float*)out, (float*)rm, (float*)rv, (float*)g, (float*)b, N, C, H, W, eps);
    });

    m.def("backward_bn", [](uintptr_t go, uintptr_t in, uintptr_t gi, uintptr_t gg, uintptr_t gb, uintptr_t sm, uintptr_t sv, uintptr_t g, int N, int C, int H, int W, float eps) {
        backward_bn((float*)go, (float*)in, (float*)gi, (float*)gg, (float*)gb, (float*)sm, (float*)sv, (float*)g, N, C, H, W, eps);
    });
}