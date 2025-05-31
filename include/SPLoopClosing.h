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

#ifndef SPLOOPCLOSING_H
#define SPLOOPCLOSING_H

#include <vector>
#include <list>
#include <memory>
#include <thread>
#include <mutex>

#include <opencv4/opencv2/opencv.hpp>
#include "thirdparty/DBoW3/src/DBoW3.h"

#include "KeyFrame.h"
#include "Map.h"
#include "SPVocabulary.h"
#include "SPBowVector.h"
#include "SuperGlueTRT.h"

namespace SuperSLAM {

class LocalMapping;
class KeyFrameDatabase;

/**
 * @brief SuperPoint-based Loop Closure Detection and Correction
 * 
 * This class detects loop closures using SuperPoint descriptors and DBoW3 vocabulary,
 * then performs loop closure correction using SuperGlue feature matching for robust
 * geometric verification.
 */
class SPLoopClosing {
public:
    /**
     * @brief Loop closure detection result
     */
    struct LoopCandidate {
        KeyFrame* keyframe;
        double score;
        std::vector<cv::DMatch> matches;
        cv::Mat relative_pose;
        bool valid;
        
        LoopCandidate() : keyframe(nullptr), score(0.0), valid(false) {}
    };

    SPLoopClosing(Map* pMap, KeyFrameDatabase* pKFDB, SPVocabulary* pVoc,
                  const bool bFixScale, std::shared_ptr<SuperGlueTRT> pSuperGlue);
    
    void SetTracker(void* pTracker);
    void SetLocalMapper(LocalMapping* pLocalMapper);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame *pKF);

    void RequestReset();
    void RequestFinish();
    bool isFinished();

    // Loop detection parameters
    void SetMinScore(double min_score) { min_score_ = min_score; }
    void SetCovisibilityConsistencyThreshold(int threshold) { covisibility_consistency_threshold_ = threshold; }
    void SetMinMatches(int min_matches) { min_matches_ = min_matches; }

protected:
    bool CheckNewKeyFrames();
    bool DetectLoop();
    bool ComputeSim3();
    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);
    void CorrectLoop();

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    Map* mpMap;
    void* mpTracker;
    
    KeyFrameDatabase* mpKeyFrameDB;
    SPVocabulary* mpVocabulary;
    
    LocalMapping *mpLocalMapper;

    std::list<KeyFrame*> mlpLoopKeyFrameQueue;
    std::mutex mMutexLoopQueue;

    // Loop detector variables
    std::vector<LoopCandidate> mvpEnoughConsistentCandidates;
    std::vector<LoopCandidate> mvpCurrentConnectedKFs;
    KeyFrame* mpCurrentKF;
    KeyFrame* mpMatchedKF;
    std::vector<MapPoint*> mvpCurrentMatchedPoints;
    std::vector<MapPoint*> mvpLoopMapPoints;

    // SuperPoint/SuperGlue components
    std::shared_ptr<SuperGlueTRT> mpSuperGlue;

    // Parameters
    double min_score_;
    int covisibility_consistency_threshold_;
    int min_matches_;
    bool mbFixScale;

    // Threading
    std::thread* mptLoopClosing;

    // Consistency checking
    struct ConsistentGroup {
        std::set<KeyFrame*> spKF;
        int nKF;
        
        ConsistentGroup() : nKF(0) {}
        ConsistentGroup(std::set<KeyFrame*> s, int n) : spKF(s), nKF(n) {}
    };

    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame*> mvpCurrentConsistentKFs;
    int mnCovisibilityConsistencyTh;

private:
    /**
     * @brief Detect loop closure candidates using SuperPoint BoW
     * @param pKF Current keyframe
     * @param candidates Output loop candidates
     * @return True if candidates found
     */
    bool detectLoopCandidates(KeyFrame* pKF, std::vector<LoopCandidate>& candidates);

    /**
     * @brief Verify loop closure using SuperGlue feature matching
     * @param current_kf Current keyframe
     * @param candidate_kf Candidate keyframe
     * @param matches Output feature matches
     * @return True if loop verified
     */
    bool verifyLoopWithSuperGlue(KeyFrame* current_kf, KeyFrame* candidate_kf, 
                                std::vector<cv::DMatch>& matches);

    /**
     * @brief Compute relative pose from feature matches
     * @param current_kf Current keyframe
     * @param candidate_kf Loop candidate keyframe
     * @param matches Feature matches
     * @param relative_pose Output relative pose
     * @return True if pose computed successfully
     */
    bool computeRelativePose(KeyFrame* current_kf, KeyFrame* candidate_kf,
                           const std::vector<cv::DMatch>& matches,
                           cv::Mat& relative_pose);

    /**
     * @brief Check geometric consistency of loop closure
     * @param current_kf Current keyframe
     * @param candidate_kf Candidate keyframe  
     * @param matches Feature matches
     * @return True if geometrically consistent
     */
    bool checkGeometricConsistency(KeyFrame* current_kf, KeyFrame* candidate_kf,
                                 const std::vector<cv::DMatch>& matches);

    /**
     * @brief Update covisibility consistency tracking
     * @param candidates Current loop candidates
     */
    void updateConsistencyTracking(const std::vector<LoopCandidate>& candidates);

    /**
     * @brief Get SuperPoint descriptors from keyframe
     * @param pKF Keyframe
     * @param descriptors Output descriptors matrix
     * @return True if descriptors extracted
     */
    bool getSuperPointDescriptors(KeyFrame* pKF, cv::Mat& descriptors);

    /**
     * @brief Extract keypoints from keyframe for SuperGlue
     * @param pKF Keyframe
     * @param keypoints Output keypoints vector
     * @return True if keypoints extracted
     */
    bool getSuperPointKeypoints(KeyFrame* pKF, std::vector<cv::KeyPoint>& keypoints);
};

} // namespace SuperSLAM

#endif // SPLOOPCLOSING_H