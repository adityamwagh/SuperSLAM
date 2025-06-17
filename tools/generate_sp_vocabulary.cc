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

#include <Logging.h>
#include <SPExtractor.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ReadConfig.h"
#include "SPBowVector.h"
#include "SuperPointTRT.h"
#include "thirdparty/DBoW3/src/DBoW3.h"

namespace fs = std::filesystem;

/**
 * @brief SuperPoint Vocabulary Generation Tool
 *
 * This tool generates a DBoW3 vocabulary from SuperPoint descriptors
 * extracted from a collection of images. The vocabulary can then be used
 * for loop closure detection in SuperSLAM.
 */
class SPVocabularyGenerator {
 private:
  std::unique_ptr<SuperPointTRT> superpoint_;
  std::shared_ptr<Configs> configs_;

 public:
  SPVocabularyGenerator(const std::string& config_path,
                        const std::string& model_dir) {
    configs_ = std::make_shared<Configs>(config_path, model_dir);
    superpoint_ = std::make_unique<SuperPointTRT>(configs_->superpoint_config);

    if (!superpoint_->initialize()) {
      throw std::runtime_error("Failed to initialize SuperPointTRT");
    }

    SLOG_INFO("SuperPoint initialized successfully for vocabulary generation");
  }

  /**
   * @brief Extract SuperPoint features from a single image
   */
  bool extractFeatures(const cv::Mat& image,
                       std::vector<cv::KeyPoint>& keypoints,
                       cv::Mat& descriptors) {
    cv::Mat gray_image;
    if (image.channels() == 3) {
      cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
      gray_image = image;
    }

    return superpoint_->infer(gray_image, keypoints, descriptors);
  }

  /**
   * @brief Process a directory of images and extract all features
   */
  bool processImageDirectory(
      const std::string& image_dir,
      std::vector<std::vector<cv::KeyPoint>>& all_keypoints,
      std::vector<cv::Mat>& all_descriptors, int max_images = -1) {
    std::vector<std::string> image_files;

    // Collect image files
    for (const auto& entry : fs::directory_iterator(image_dir)) {
      if (entry.is_regular_file()) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
            ext == ".tiff") {
          image_files.push_back(entry.path().string());
        }
      }
    }

    if (image_files.empty()) {
      SLOG_ERROR("No image files found in directory: {}", image_dir);
      return false;
    }

    std::sort(image_files.begin(), image_files.end());

    if (max_images > 0 && static_cast<int>(image_files.size()) > max_images) {
      image_files.resize(max_images);
    }

    SLOG_INFO("Processing {} images...", image_files.size());

    all_keypoints.clear();
    all_descriptors.clear();
    all_keypoints.reserve(image_files.size());
    all_descriptors.reserve(image_files.size());

    int processed = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& image_path : image_files) {
      cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
      if (image.empty()) {
        SLOG_WARN("Failed to load image: {}", image_path);
        continue;
      }

      std::vector<cv::KeyPoint> keypoints;
      cv::Mat descriptors;

      if (extractFeatures(image, keypoints, descriptors)) {
        all_keypoints.push_back(keypoints);
        all_descriptors.push_back(descriptors);

        processed++;
        if (processed % 100 == 0) {
          auto current_time = std::chrono::high_resolution_clock::now();
          auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
              current_time - start_time);
          SLOG_INFO("Processed {}/{} images ({}s)", processed, image_files.size(), elapsed.count());
        }
      } else {
        SLOG_WARN("Failed to extract features from: {}", image_path);
      }
    }

    SLOG_INFO("Successfully processed {} images", processed);
    return processed > 0;
  }

  /**
   * @brief Generate vocabulary from extracted features
   */
  bool generateVocabulary(
      const std::vector<std::vector<cv::KeyPoint>>& all_keypoints,
      const std::vector<cv::Mat>& all_descriptors,
      const std::string& output_path, int k = 10, int levels = 6,
      int max_features_per_image = 500) {
    if (all_keypoints.size() != all_descriptors.size()) {
      SLOG_ERROR("Keypoints and descriptors size mismatch");
      return false;
    }

    SLOG_INFO("Preparing training data...");

    // Prepare training descriptors
    std::vector<cv::Mat> training_descriptors;
    SuperSLAM::SPBowVector::prepareTrainingData(all_keypoints, all_descriptors,
                                                training_descriptors,
                                                max_features_per_image);

    if (training_descriptors.empty()) {
      SLOG_ERROR("No training descriptors prepared");
      return false;
    }

    SLOG_INFO("Creating vocabulary with {} descriptors...", training_descriptors.size());
    SLOG_INFO("Vocabulary parameters: k={}, levels={}", k, levels);

    // Create vocabulary
    DBoW3::Vocabulary vocabulary;

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
      vocabulary.create(training_descriptors, k, levels);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::minutes>(
          end_time - start_time);

      SLOG_INFO("Vocabulary created in {} minutes", duration.count());
      SLOG_INFO("Vocabulary size: {} words", vocabulary.size());

      // Save vocabulary
      SLOG_INFO("Saving vocabulary to: {}", output_path);
      vocabulary.save(output_path);

      SLOG_INFO("Vocabulary saved successfully!");
      return true;

    } catch (const std::exception& e) {
      SLOG_ERROR("Error creating vocabulary: {}", e.what());
      return false;
    }
  }
};

void printUsage() {
  const std::string usage = R"(
SuperPoint Vocabulary Generator
Usage: ./generate_sp_vocabulary <config_path> <model_dir> <image_dir> <output_vocab> [options]

Arguments:
  config_path     Path to SuperPoint configuration file (e.g., utils/config.yaml)
  model_dir       Directory containing SuperPoint model files (e.g., weights/)
  image_dir       Directory containing training images
  output_vocab    Output vocabulary file path (e.g., sp_vocabulary.yml.gz)

Options:
  --k K                    Branching factor (default: 10)
  --levels L               Depth levels (default: 6)
  --max-images N           Maximum images to process (default: all)
  --max-features-per-image Maximum features per image (default: 500)

Example:
  ./generate_sp_vocabulary utils/config.yaml weights/ /dataset/images/ sp_vocabulary.yml.gz --k 10 --levels 6
)";
  SLOG_INFO("{}", usage);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    printUsage();
    return 1;
  }

  // Initialize logging
  SuperSLAM::Logger::initialize();

  std::string config_path = argv[1];
  std::string model_dir = argv[2];
  std::string image_dir = argv[3];
  std::string output_vocab = argv[4];

  // Parse optional arguments
  int k = 10;
  int levels = 6;
  int max_images = -1;
  int max_features_per_image = 500;

  for (int i = 5; i < argc; i += 2) {
    if (i + 1 >= argc) {
      break;
    }

    std::string arg = argv[i];
    if (arg == "--k") {
      k = std::stoi(argv[i + 1]);
    } else if (arg == "--levels") {
      levels = std::stoi(argv[i + 1]);
    } else if (arg == "--max-images") {
      max_images = std::stoi(argv[i + 1]);
    } else if (arg == "--max-features-per-image") {
      max_features_per_image = std::stoi(argv[i + 1]);
    }
  }

  try {
    SLOG_INFO("Initializing SuperPoint Vocabulary Generator...");
    SPVocabularyGenerator generator(config_path, model_dir);

    SLOG_INFO("Processing images from: {}", image_dir);
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<cv::Mat> all_descriptors;

    if (!generator.processImageDirectory(image_dir, all_keypoints,
                                         all_descriptors, max_images)) {
      SLOG_ERROR("Failed to process images");
      return 1;
    }

    SLOG_INFO("Generating vocabulary...");
    if (!generator.generateVocabulary(all_keypoints, all_descriptors,
                                      output_vocab, k, levels,
                                      max_features_per_image)) {
      SLOG_ERROR("Failed to generate vocabulary");
      return 1;
    }

    SLOG_INFO("Vocabulary generation completed successfully!");

  } catch (const std::exception& e) {
    SLOG_ERROR("Error: {}", e.what());
    return 1;
  }

  return 0;
}
