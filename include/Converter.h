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

#ifndef CONVERTER_H
#define CONVERTER_H

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Similarity3.h>

#include <Eigen/Dense>
#include <opencv4/opencv2/core/core.hpp>

namespace SuperSLAM {

class Converter {
 public:
  static std::vector<cv::Mat> toDescriptorVector(const cv::Mat& Descriptors);

  // GTSAM conversions
  static gtsam::Pose3 toPose3(const cv::Mat& cvT);
  static gtsam::Point3 toPoint3(const cv::Mat& cvPoint);
  static gtsam::Similarity3 toSimilarity3(const cv::Mat& cvT,
                                          double scale = 1.0);

  // Legacy g2o conversion helpers
  static gtsam::Similarity3 g2oSim3ToGTSAM(const Eigen::Matrix3d& R,
                                           const Eigen::Vector3d& t, double s);

  static cv::Mat toCvMat(const gtsam::Pose3& pose);
  static cv::Mat toCvMat(const gtsam::Point3& point);
  static cv::Mat toCvMat(const gtsam::Similarity3& sim3);
  static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4>& m);
  static cv::Mat toCvMat(const Eigen::Matrix3d& m);
  static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3>& R,
                         const Eigen::Matrix<double, 3, 1>& t);

  static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat& cvVector);
  static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f& cvPoint);
  static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat& cvMat3);

  static std::vector<float> toQuaternion(const cv::Mat& M);
};

}  // namespace SuperSLAM

#endif  // CONVERTER_H
