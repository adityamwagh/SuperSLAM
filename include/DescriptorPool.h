#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace superslam {

// Hold a device pointer and shape for descriptors resident in a DescriptorPool
// slot. Opaque to the GPU-free core (no nvinfer or CUDA types here); the
// inference layer reads them as FP16. Copies share the slot (refcounted):
// slot_ref's deleter returns the slot to the pool when the last copy dies.
// slot -1 or null data means no device residency.
struct DeviceDescriptors {
  void* data = nullptr;           // device pointer (into the owning pool slot)
  int count = 0;                  // number of descriptors (keypoints)
  int dim = 0;                    // descriptor dimension
  int slot = -1;                  // owning DescriptorPool slot
  std::shared_ptr<void> slot_ref; // deleter releases `slot` back to the pool
  bool empty() const { return data == nullptr || count == 0; }
};

// Manage a fixed free-list over N slot indices. CUDA-free slot bookkeeping,
// unit-testable without a device, separate from the device allocation that
// DescriptorPool layers on top.
class FreeList {
public:
  explicit FreeList(int n) : n_(n) {
    for (int i = n - 1; i >= 0; --i)
      free_slots_.push_back(i);
  }
  int acquire() {
    if (free_slots_.empty())
      return -1;
    const int slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
  }
  void release(int slot) { free_slots_.push_back(slot); }
  int in_use() const { return n_ - static_cast<int>(free_slots_.size()); }

private:
  int n_;
  std::vector<int> free_slots_;
};

// Hold a fixed pool of N device slots of FP16 descriptor rows. Size each slot
// for the worst case (max_keypoints * dim half-floats); DeviceDescriptors::count
// tracks the actual fill. The embedded FreeList manages slot lifetime. CUDA
// lives in the .cc (cudaMalloc); the header is nvinfer-free.
class DescriptorPool {
public:
  DescriptorPool(int num_slots, int max_keypoints, int dim);
  ~DescriptorPool();

  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;

  // Acquire a slot and wrap it in a refcounted DeviceDescriptors handle sized
  // for `count` keypoints. The handle's slot_ref returns the slot to the pool
  // on last destruction. Return an empty() handle (slot == -1) if the pool is
  // exhausted.
  DeviceDescriptors make(int count) {
    const int slot = free_->acquire();
    DeviceDescriptors d;
    d.count = count;
    d.dim = dim_;
    d.slot = slot;
    if (slot < 0)
      return d; // exhausted
    d.data = slot_ptr(slot);
    // Capture the shared free-list, not `this`; a handle may outlive the pool.
    auto free_list = free_;
    d.slot_ref =
        std::shared_ptr<void>(d.data, [free_list, slot](void*) { free_list->release(slot); });
    return d;
  }

  // Return the base device pointer for a slot (nullptr if out of range).
  void* slot_ptr(int slot) const;

  int dim() const { return dim_; }
  int max_keypoints() const { return max_keypoints_; }
  int in_use() const { return free_->in_use(); }

private:
  int max_keypoints_;
  int dim_;
  std::size_t slot_bytes_;
  std::vector<void*> slots_;       // device pointers, one per slot
  std::shared_ptr<FreeList> free_; // shared; handles may outlive the pool
};

} // namespace superslam
