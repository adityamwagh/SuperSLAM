#include "DescriptorGather.h"

#include <cuda_fp16.h>

namespace superslam {

namespace {

// One block per keypoint; threads split the channels. Gather the descriptor
// at the keypoint's grid cell (strided: grid[c, h, w] = grid[c*grid_h*grid_w +
// h*grid_w + w], nearest cell), reduce the squared L2 norm across the block,
// then write the normalized row. Grid and output are FP16 (the norm is reduced
// in fp32).
__global__ void gather_normalize_kernel(const __half* grid,
                                        int channels,
                                        int grid_h,
                                        int grid_w,
                                        const int* cell_h,
                                        const int* cell_w,
                                        int num_keypoints,
                                        __half* out) {
  const int n = blockIdx.x;
  if (n >= num_keypoints)
    return;

  const int h = cell_h[n];
  const int w = cell_w[n];
  const int plane = grid_h * grid_w;
  const int base = h * grid_w + w;

  // Pass 1: accumulate the partial squared norm over this thread's channels
  // (in fp32).
  float partial = 0.0f;
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    const float v = __half2float(grid[c * plane + base]);
    partial += v * v;
  }

  // Block reduction of the squared norm into shared memory.
  extern __shared__ float shared[];
  shared[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float inv_norm = rsqrtf(shared[0] + 1e-12f);

  // Pass 2: write the normalized descriptor row.
  __half* row = out + static_cast<long>(n) * channels;
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    const float v = __half2float(grid[c * plane + base]);
    row[c] = __float2half(v * inv_norm);
  }
}

} // namespace

void launch_gather_descriptors(const void* grid_fp16,
                               int channels,
                               int grid_h,
                               int grid_w,
                               const int* cell_h,
                               const int* cell_w,
                               int num_keypoints,
                               void* out_fp16,
                               cudaStream_t stream) {
  if (num_keypoints <= 0)
    return;
  const int threads = 256;
  const int shared_bytes = threads * static_cast<int>(sizeof(float));
  gather_normalize_kernel<<<num_keypoints, threads, shared_bytes, stream>>>(
      static_cast<const __half*>(grid_fp16),
      channels,
      grid_h,
      grid_w,
      cell_h,
      cell_w,
      num_keypoints,
      static_cast<__half*>(out_fp16));
}

} // namespace superslam
