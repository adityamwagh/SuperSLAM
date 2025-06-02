/**
 * This file is part of SuperSLAM.
 *
 * Copyright (C) Aditya Wagh <adityamwagh at outlook dot com>
 * For more information see <https://github.com/adityamwagh/SuperSLAM>
 *
 * SuperSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SuperSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SuperSLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <list>
#include <mutex>
#include <set>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "SPVocabulary.h"

namespace SuperSLAM {

class KeyFrame;
class Frame;

class KeyFrameDatabase {
 public:
  KeyFrameDatabase(const ORBVocabulary& voc);

  void add(KeyFrame* pKF);

  void erase(KeyFrame* pKF);

  void clear();

  // Loop Detection
  std::vector<KeyFrame*> DetectLoopCandidates(KeyFrame* pKF, float minScore);

  // Relocalization
  std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);

 protected:
  // Associated vocabulary
  const ORBVocabulary* mpVoc;

  // Inverted file
  std::vector<std::list<KeyFrame*>> mvInvertedFile;

  // Mutex
  std::mutex mMutex;
};

}  // namespace SuperSLAM

#endif
