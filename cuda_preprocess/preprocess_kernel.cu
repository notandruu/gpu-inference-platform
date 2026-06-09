#include <cuda_runtime.h>
#include <stdint.h>

// ImageNet normalization constants (mean and std per channel)
__constant__ float kMean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float kStd[3]  = {0.229f, 0.224f, 0.225f};

/**
 * Convert a batch of uint8 NHWC images to float32 NCHW tensors.
 *
 * Input:  uint8  [N, H, W, C] in [0, 255]
 * Output: float32 [N, C, H, W] normalized by ImageNet mean/std
 *
 * Each thread processes one pixel across all channels.
 */
__global__ void nhwc_to_nchw_normalize_kernel(
    const uint8_t* __restrict__ src,   // [N, H, W, C]
    float*         __restrict__ dst,   // [N, C, H, W]
    int N, int H, int W, int C)
{
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * H * W;
    if (pixel_idx >= total_pixels) return;

    int n = pixel_idx / (H * W);
    int hw = pixel_idx % (H * W);
    int h = hw / W;
    int w = hw % W;

    for (int c = 0; c < C; ++c) {
        // NHWC index
        int src_idx = ((n * H + h) * W + w) * C + c;
        // NCHW index
        int dst_idx = ((n * C + c) * H + h) * W + w;

        float val = static_cast<float>(src[src_idx]) / 255.0f;
        dst[dst_idx] = (val - kMean[c]) / kStd[c];
    }
}

extern "C" {

/**
 * Launch the NHWC→NCHW normalization kernel.
 *
 * src and dst must be device pointers.
 * stream may be 0 for the default stream.
 */
void launch_nhwc_to_nchw_normalize(
    const uint8_t* src,
    float*         dst,
    int N, int H, int W, int C,
    cudaStream_t stream)
{
    int total = N * H * W;
    int block = 256;
    int grid  = (total + block - 1) / block;
    nhwc_to_nchw_normalize_kernel<<<grid, block, 0, stream>>>(src, dst, N, H, W, C);
}

} // extern "C"
