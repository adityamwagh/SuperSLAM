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

#include "KeyFrameDatabase.h"

#include <mutex>

#include "KeyFrame.h"
#include "Logging.h"
#include "thirdparty/DBoW3/src/DBoW3.h"

namespace SuperSLAM {

KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary& voc) : mpVoc(&voc) {
  mvInvertedFile.resize(voc.size());
}

void KeyFrameDatabase::add(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutex);

  for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(),
                                        vend = pKF->mBowVec.end();
       vit != vend; vit++)
    mvInvertedFile[vit->first].push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF) {
  std::unique_lock<std::mutex> lock(mMutex);

  // Erase elements in the Inverse File for the entry
  for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(),
                                        vend = pKF->mBowVec.end();
       vit != vend; vit++) {
    // List of keyframes that share the word
    std::list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];

    for (std::list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end();
         lit != lend; lit++) {
      if (pKF == *lit) {
        lKFs.erase(lit);
        break;
      }
    }
  }
}

void KeyFrameDatabase::clear() {
  mvInvertedFile.clear();
  mvInvertedFile.resize(mpVoc->size());
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF,
                                                              float minScore) {
  std::set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
  std::list<KeyFrame*> lKFsSharingWords;

  // Search all keyframes that share a word with current keyframes
  // Discard keyframes connected to the query keyframe
  {
    std::unique_lock<std::mutex> lock(mMutex);

    for (DBoW3::BowVector::const_iterator vit = pKF->mBowVec.begin(),
                                          vend = pKF->mBowVec.end();
         vit != vend; vit++) {
      std::list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];

      for (std::list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end();
           lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        if (pKFi->mnLoopQuery != pKF->mnId) {
          pKFi->mnLoopWords = 0;
          if (!spConnectedKeyFrames.count(pKFi)) {
            pKFi->mnLoopQuery = pKF->mnId;
            lKFsSharingWords.push_back(pKFi);
          }
        }
        pKFi->mnLoopWords++;
      }
    }
  }

  if (lKFsSharingWords.empty()) return std::vector<KeyFrame*>();

  std::list<std::pair<float, KeyFrame*>> lScoreAndMatch;

  // Only compare against those keyframes that share enough words
  int maxCommonWords = 0;
  for (std::list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                                      lend = lKFsSharingWords.end();
       lit != lend; lit++) {
    if ((*lit)->mnLoopWords > maxCommonWords)
      maxCommonWords = (*lit)->mnLoopWords;
  }

  int minCommonWords = maxCommonWords * 0.8f;

  int nscores = 0;

  // Compute similarity score. Retain the matches whose score is higher than
  // minScore
  for (std::list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                                      lend = lKFsSharingWords.end();
       lit != lend; lit++) {
    KeyFrame* pKFi = *lit;

    if (pKFi->mnLoopWords > minCommonWords) {
      nscores++;

      float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);

      pKFi->mLoopScore = si;
      if (si >= minScore) lScoreAndMatch.push_back(std::make_pair(si, pKFi));
    }
  }

  if (lScoreAndMatch.empty()) return std::vector<KeyFrame*>();

  std::list<std::pair<float, KeyFrame*>> lAccScoreAndMatch;
  float bestAccScore = minScore;

  // Lets now accumulate score by covisibility
  for (std::list<std::pair<float, KeyFrame*>>::iterator
           it = lScoreAndMatch.begin(),
           itend = lScoreAndMatch.end();
       it != itend; it++) {
    KeyFrame* pKFi = it->second;
    std::vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

    float bestScore = it->first;
    float accScore = it->first;
    KeyFrame* pBestKF = pKFi;
    for (std::vector<KeyFrame*>::iterator vit = vpNeighs.begin(),
                                          vend = vpNeighs.end();
         vit != vend; vit++) {
      KeyFrame* pKF2 = *vit;
      if (pKF2->mnLoopQuery == pKF->mnId &&
          pKF2->mnLoopWords > minCommonWords) {
        accScore += pKF2->mLoopScore;
        if (pKF2->mLoopScore > bestScore) {
          pBestKF = pKF2;
          bestScore = pKF2->mLoopScore;
        }
      }
    }

    lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
    if (accScore > bestAccScore) bestAccScore = accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f * bestAccScore;

  std::set<KeyFrame*> spAlreadyAddedKF;
  std::vector<KeyFrame*> vpLoopCandidates;
  vpLoopCandidates.reserve(lAccScoreAndMatch.size());

  for (std::list<std::pair<float, KeyFrame*>>::iterator
           it = lAccScoreAndMatch.begin(),
           itend = lAccScoreAndMatch.end();
       it != itend; it++) {
    if (it->first > minScoreToRetain) {
      KeyFrame* pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpLoopCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }

  return vpLoopCandidates;
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(
    Frame* F) {
  std::list<KeyFrame*> lKFsSharingWords;

  // Search all keyframes that share a word with current frame
  {
    std::unique_lock<std::mutex> lock(mMutex);

    for (DBoW3::BowVector::const_iterator vit = F->mBowVec.begin(),
                                          vend = F->mBowVec.end();
         vit != vend; vit++) {
      std::list<KeyFrame*>& lKFs = mvInvertedFile[vit->first];

      for (std::list<KeyFrame*>::iterator lit = lKFs.begin(), lend = lKFs.end();
           lit != lend; lit++) {
        KeyFrame* pKFi = *lit;
        if (pKFi->mnRelocQuery != F->mnId) {
          pKFi->mnRelocWords = 0;
          pKFi->mnRelocQuery = F->mnId;
          lKFsSharingWords.push_back(pKFi);
        }
        pKFi->mnRelocWords++;
      }
    }
  }
  if (lKFsSharingWords.empty()) return std::vector<KeyFrame*>();

  // Only compare against those keyframes that share enough words
  int maxCommonWords = 0;
  for (std::list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                                      lend = lKFsSharingWords.end();
       lit != lend; lit++) {
    if ((*lit)->mnRelocWords > maxCommonWords)
      maxCommonWords = (*lit)->mnRelocWords;
  }

  int minCommonWords = maxCommonWords * 0.8f;

  std::list<std::pair<float, KeyFrame*>> lScoreAndMatch;

  int nscores = 0;

  // Compute similarity score.
  for (std::list<KeyFrame*>::iterator lit = lKFsSharingWords.begin(),
                                      lend = lKFsSharingWords.end();
       lit != lend; lit++) {
    KeyFrame* pKFi = *lit;

    if (pKFi->mnRelocWords > minCommonWords) {
      nscores++;
      float si = mpVoc->score(F->mBowVec, pKFi->mBowVec);
      pKFi->mRelocScore = si;
      lScoreAndMatch.push_back(std::make_pair(si, pKFi));
    }
  }

  if (lScoreAndMatch.empty()) return std::vector<KeyFrame*>();

  std::list<std::pair<float, KeyFrame*>> lAccScoreAndMatch;
  float bestAccScore = 0;

  // Lets now accumulate score by covisibility
  for (std::list<std::pair<float, KeyFrame*>>::iterator
           it = lScoreAndMatch.begin(),
           itend = lScoreAndMatch.end();
       it != itend; it++) {
    KeyFrame* pKFi = it->second;
    std::vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

    float bestScore = it->first;
    float accScore = bestScore;
    KeyFrame* pBestKF = pKFi;
    for (std::vector<KeyFrame*>::iterator vit = vpNeighs.begin(),
                                          vend = vpNeighs.end();
         vit != vend; vit++) {
      KeyFrame* pKF2 = *vit;
      if (pKF2->mnRelocQuery != F->mnId) continue;

      accScore += pKF2->mRelocScore;
      if (pKF2->mRelocScore > bestScore) {
        pBestKF = pKF2;
        bestScore = pKF2->mRelocScore;
      }
    }
    lAccScoreAndMatch.push_back(std::make_pair(accScore, pBestKF));
    if (accScore > bestAccScore) bestAccScore = accScore;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  float minScoreToRetain = 0.75f * bestAccScore;
  std::set<KeyFrame*> spAlreadyAddedKF;
  std::vector<KeyFrame*> vpRelocCandidates;
  vpRelocCandidates.reserve(lAccScoreAndMatch.size());
  for (std::list<std::pair<float, KeyFrame*>>::iterator
           it = lAccScoreAndMatch.begin(),
           itend = lAccScoreAndMatch.end();
       it != itend; it++) {
    const float& si = it->first;
    if (si > minScoreToRetain) {
      KeyFrame* pKFi = it->second;
      if (!spAlreadyAddedKF.count(pKFi)) {
        vpRelocCandidates.push_back(pKFi);
        spAlreadyAddedKF.insert(pKFi);
      }
    }
  }

  return vpRelocCandidates;
}

}  // namespace SuperSLAM
