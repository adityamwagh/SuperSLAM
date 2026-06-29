#include "DescriptorPool.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "Logging.h"

namespace superslam {

DescriptorPool::DescriptorPool(int num_slots, int max_keypoints, int dim)
    : max_keypoints_(max_keypoints), dim_(dim),
      slot_bytes_(static_cast<std::size_t>(max_keypoints) * dim * sizeof(__half)),
      slots_(num_slots, nullptr), free_(std::make_shared<FreeList>(num_slots)) {
  for (int i = 0; i < num_slots; ++i) {
    if (cudaMalloc(&slots_[i], slot_bytes_) != cudaSuccess) {
      SLOG_ERROR("DescriptorPool: cudaMalloc failed for slot {} ({} bytes)", i, slot_bytes_);
      slots_[i] = nullptr;
    }
  }
  SLOG_INFO("DescriptorPool: {} slots x {} bytes ({} kpts x {} dim, FP16)",
            num_slots,
            slot_bytes_,
            max_keypoints,
            dim);
}

DescriptorPool::~DescriptorPool() {
  for (void* p : slots_) {
    if (p != nullptr)
      cudaFree(p);
  }
}

void* DescriptorPool::slot_ptr(int slot) const {
  if (slot < 0 || slot >= static_cast<int>(slots_.size()))
    return nullptr;
  return slots_[slot];
}

} // namespace superslam
