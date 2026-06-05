#include "KeyframeDatabase.h"

#include <stdexcept>

namespace superslam {

void KeyframeDatabase::add(KeyframeRecord rec) {
  const size_t keyframe_id = rec.keyframe_id;
  id_to_index_[keyframe_id] = keyframes_.size();
  keyframes_.push_back(std::move(rec));
}

const KeyframeRecord& KeyframeDatabase::get(size_t keyframe_id) const {
  auto it = id_to_index_.find(keyframe_id);
  if (it == id_to_index_.end())
    throw std::out_of_range("KeyframeDatabase: unknown keyframe id");
  return keyframes_[it->second];
}

} // namespace superslam
