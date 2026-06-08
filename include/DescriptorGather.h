#pragma once
#include <cuda_runtime.h>

namespace superslam {

// Gather and L2-normalize SuperPoint descriptors on the device. Sample the dense
// FP16 descriptor grid (layout [channels, grid_h, grid_w]) at the host-selected
// keypoint cells (nearest cell: cell = keypoint_score_coord / 8), producing an
// [num_keypoints, channels] FP16 buffer with each row L2-normalized. All
// pointers are device pointers; cell_h and cell_w are device arrays of the
// per-keypoint grid cells.
void launch_gather_descriptors(const void* grid_fp16,
                               int channels,
                               int grid_h,
                               int grid_w,
                               const int* cell_h,
                               const int* cell_w,
                               int num_keypoints,
                               void* out_fp16,
                               cudaStream_t stream);

} // namespace superslam
