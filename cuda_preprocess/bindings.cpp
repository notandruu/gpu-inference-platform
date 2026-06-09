#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

// Forward declaration
void preprocess_nhwc_to_nchw(
    const uint8_t* src_host,
    float*         dst_host,
    int N, int H, int W, int C);

/**
 * Python binding.
 *
 * Args:
 *   images: uint8 numpy array of shape (N, H, W, C)
 *
 * Returns:
 *   float32 numpy array of shape (N, C, H, W) normalized by ImageNet stats
 */
py::array_t<float> cuda_preprocess(py::array_t<uint8_t, py::array::c_contiguous> images) {
    auto buf = images.request();
    if (buf.ndim != 4) {
        throw std::invalid_argument("Expected NHWC array with 4 dimensions");
    }

    int N = buf.shape[0];
    int H = buf.shape[1];
    int W = buf.shape[2];
    int C = buf.shape[3];

    if (C != 3) {
        throw std::invalid_argument("Expected 3-channel (RGB) input");
    }

    py::array_t<float> output({N, C, H, W});
    auto out_buf = output.request();

    preprocess_nhwc_to_nchw(
        static_cast<const uint8_t*>(buf.ptr),
        static_cast<float*>(out_buf.ptr),
        N, H, W, C);

    return output;
}

PYBIND11_MODULE(cuda_preprocess, m) {
    m.doc() = "CUDA-accelerated image preprocessing: NHWC uint8 → NCHW float32 (ImageNet normalized)";
    m.def(
        "preprocess",
        &cuda_preprocess,
        "Convert NHWC uint8 images to NCHW float32 tensors normalized by ImageNet mean/std",
        py::arg("images")
    );
}
