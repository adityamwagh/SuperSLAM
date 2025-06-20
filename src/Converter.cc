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

#include "Converter.h"

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Similarity3.h>

#include "Logging.h"

namespace SuperSLAM {

std::vector<cv::Mat> Converter::toDescriptorVector(const cv::Mat &Descriptors) {
  std::vector<cv::Mat> vDesc;
  vDesc.reserve(Descriptors.rows);
  for (int j = 0; j < Descriptors.rows; j++)
    vDesc.push_back(Descriptors.row(j));

  return vDesc;
}

gtsam::Pose3 Converter::toPose3(const cv::Mat &cvT) {
  // Extract rotation matrix
  Eigen::Matrix3d R;
  R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
      cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
      cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

  // Extract translation vector
  Eigen::Vector3d t(cvT.at<float>(0, 3), cvT.at<float>(1, 3),
                    cvT.at<float>(2, 3));

  // Create GTSAM Pose3
  gtsam::Rot3 rot3(R);
  gtsam::Point3 point3(t);

  return gtsam::Pose3(rot3, point3);
}

gtsam::Point3 Converter::toPoint3(const cv::Mat &cvPoint) {
  if (cvPoint.rows == 3 && cvPoint.cols == 1) {
    return gtsam::Point3(cvPoint.at<float>(0), cvPoint.at<float>(1),
                         cvPoint.at<float>(2));
  } else if (cvPoint.rows == 1 && cvPoint.cols == 3) {
    return gtsam::Point3(cvPoint.at<float>(0, 0), cvPoint.at<float>(0, 1),
                         cvPoint.at<float>(0, 2));
  } else {
    SPDLOG_ERROR("Invalid cv::Mat dimensions for Point3 conversion");
    return gtsam::Point3(0, 0, 0);
  }
}

gtsam::Similarity3 Converter::toSimilarity3(const cv::Mat &cvT, double scale) {
  // Extract rotation matrix
  Eigen::Matrix3d R;
  R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
      cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
      cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

  // Extract translation vector
  Eigen::Vector3d t(cvT.at<float>(0, 3), cvT.at<float>(1, 3),
                    cvT.at<float>(2, 3));

  // Create GTSAM Similarity3
  gtsam::Rot3 rot3(R);
  gtsam::Point3 point3(t);

  return gtsam::Similarity3(rot3, point3, scale);
}

gtsam::Similarity3 Converter::g2oSim3ToGTSAM(const Eigen::Matrix3d &R,
                                             const Eigen::Vector3d &t,
                                             double s) {
  gtsam::Rot3 rot3(R);
  gtsam::Point3 point3(t);
  return gtsam::Similarity3(rot3, point3, s);
}

cv::Mat Converter::toCvMat(const gtsam::Pose3 &pose) {
  // Get rotation matrix and translation vector
  gtsam::Matrix3 R = pose.rotation().matrix();
  gtsam::Vector3 t = pose.translation();

  // Create 4x4 transformation matrix
  cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);

  // Fill rotation part
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cvMat.at<float>(i, j) = static_cast<float>(R(i, j));
    }
  }

  // Fill translation part
  for (int i = 0; i < 3; i++) {
    cvMat.at<float>(i, 3) = static_cast<float>(t(i));
  }

  return cvMat.clone();
}

cv::Mat Converter::toCvMat(const gtsam::Point3 &point) {
  cv::Mat cvMat(3, 1, CV_32F);
  cvMat.at<float>(0) = static_cast<float>(point.x());
  cvMat.at<float>(1) = static_cast<float>(point.y());
  cvMat.at<float>(2) = static_cast<float>(point.z());

  return cvMat.clone();
}

cv::Mat Converter::toCvMat(const gtsam::Similarity3 &sim3) {
  // Get rotation matrix, translation vector, and scale
  gtsam::Matrix3 R = sim3.rotation().matrix();
  gtsam::Vector3 t = sim3.translation();
  double s = sim3.scale();

  // Apply scale to rotation matrix
  Eigen::Matrix3d scaledR = s * R;

  return toCvSE3(scaledR, t);
}

cv::Mat Converter::toCvMat(const Eigen::Matrix<double, 4, 4> &m) {
  cv::Mat cvMat(4, 4, CV_32F);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      cvMat.at<float>(i, j) = static_cast<float>(m(i, j));

  return cvMat.clone();
}

cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m) {
  cv::Mat cvMat(3, 3, CV_32F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      cvMat.at<float>(i, j) = static_cast<float>(m(i, j));

  return cvMat.clone();
}

cv::Mat Converter::toCvSE3(const Eigen::Matrix<double, 3, 3> &R,
                           const Eigen::Matrix<double, 3, 1> &t) {
  cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cvMat.at<float>(i, j) = static_cast<float>(R(i, j));
    }
  }
  for (int i = 0; i < 3; i++) {
    cvMat.at<float>(i, 3) = static_cast<float>(t(i));
  }

  return cvMat.clone();
}

Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Mat &cvVector) {
  Eigen::Matrix<double, 3, 1> v;
  v << static_cast<double>(cvVector.at<float>(0)),
      static_cast<double>(cvVector.at<float>(1)),
      static_cast<double>(cvVector.at<float>(2));

  return v;
}

Eigen::Matrix<double, 3, 1> Converter::toVector3d(const cv::Point3f &cvPoint) {
  Eigen::Matrix<double, 3, 1> v;
  v << static_cast<double>(cvPoint.x), static_cast<double>(cvPoint.y),
      static_cast<double>(cvPoint.z);

  return v;
}

Eigen::Matrix<double, 3, 3> Converter::toMatrix3d(const cv::Mat &cvMat3) {
  Eigen::Matrix<double, 3, 3> M;

  M << static_cast<double>(cvMat3.at<float>(0, 0)),
      static_cast<double>(cvMat3.at<float>(0, 1)),
      static_cast<double>(cvMat3.at<float>(0, 2)),
      static_cast<double>(cvMat3.at<float>(1, 0)),
      static_cast<double>(cvMat3.at<float>(1, 1)),
      static_cast<double>(cvMat3.at<float>(1, 2)),
      static_cast<double>(cvMat3.at<float>(2, 0)),
      static_cast<double>(cvMat3.at<float>(2, 1)),
      static_cast<double>(cvMat3.at<float>(2, 2));

  return M;
}

std::vector<float> Converter::toQuaternion(const cv::Mat &M) {
  Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
  Eigen::Quaterniond q(eigMat);

  std::vector<float> v(4);
  v[0] = static_cast<float>(q.x());
  v[1] = static_cast<float>(q.y());
  v[2] = static_cast<float>(q.z());
  v[3] = static_cast<float>(q.w());

  return v;
}

}  // namespace SuperSLAM