#include <cuda_runtime.h>
#include <stdint.h>
#include <stdexcept>
#include <string>

// Forward declaration from .cu
extern "C" {
void launch_nhwc_to_nchw_normalize(
    const uint8_t* src,
    float*         dst,
    int N, int H, int W, int C,
    cudaStream_t stream);
}

/**
 * Host-side wrapper: allocates device memory, copies input, launches kernel,
 * copies result back, and frees device memory.
 *
 * src_host: uint8 NHWC buffer of size N*H*W*C bytes
 * dst_host: float32 NCHW buffer of size N*C*H*W floats (caller-allocated)
 */
void preprocess_nhwc_to_nchw(
    const uint8_t* src_host,
    float*         dst_host,
    int N, int H, int W, int C)
{
    size_t src_bytes = static_cast<size_t>(N) * H * W * C * sizeof(uint8_t);
    size_t dst_bytes = static_cast<size_t>(N) * C * H * W * sizeof(float);

    uint8_t* d_src = nullptr;
    float*   d_dst = nullptr;

    auto check = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string(msg) + ": " + cudaGetErrorString(err));
        }
    };

    check(cudaMalloc(&d_src, src_bytes), "cudaMalloc src");
    check(cudaMalloc(&d_dst, dst_bytes), "cudaMalloc dst");
    check(cudaMemcpy(d_src, src_host, src_bytes, cudaMemcpyHostToDevice), "H2D src");

    launch_nhwc_to_nchw_normalize(d_src, d_dst, N, H, W, C, /*stream=*/0);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");

    check(cudaMemcpy(dst_host, d_dst, dst_bytes, cudaMemcpyDeviceToHost), "D2H dst");

    cudaFree(d_src);
    cudaFree(d_dst);
}
