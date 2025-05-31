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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
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

    std::cout << "SuperPoint initialized successfully for vocabulary generation"
              << "\n";
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
      std::cerr << "No image files found in directory: " << image_dir << "\n";
      return false;
    }

    std::sort(image_files.begin(), image_files.end());

    if (max_images > 0 && static_cast<int>(image_files.size()) > max_images) {
      image_files.resize(max_images);
    }

    std::cout << "Processing " << image_files.size() << " images..."
              << "\n";

    all_keypoints.clear();
    all_descriptors.clear();
    all_keypoints.reserve(image_files.size());
    all_descriptors.reserve(image_files.size());

    int processed = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& image_path : image_files) {
      cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
      if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
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
          std::cout << "Processed " << processed << "/" << image_files.size()
                    << " images (" << elapsed.count() << "s)" << "\n";
        }
      } else {
        std::cerr << "Failed to extract features from: " << image_path << "\n";
      }
    }

    std::cout << "Successfully processed " << processed << " images"
              << "\n";
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
      std::cerr << "Keypoints and descriptors size mismatch" << "\n";
      return false;
    }

    std::cout << "Preparing training data..." << "\n";

    // Prepare training descriptors
    std::vector<cv::Mat> training_descriptors;
    SuperSLAM::SPBowVector::prepareTrainingData(all_keypoints, all_descriptors,
                                                training_descriptors,
                                                max_features_per_image);

    if (training_descriptors.empty()) {
      std::cerr << "No training descriptors prepared" << "\n";
      return false;
    }

    std::cout << "Creating vocabulary with " << training_descriptors.size()
              << " descriptors..." << "\n";
    std::cout << "Vocabulary parameters: k=" << k << ", levels=" << levels
              << "\n";

    // Create vocabulary
    DBoW3::Vocabulary vocabulary;

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
      vocabulary.create(training_descriptors, k, levels);

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::minutes>(
          end_time - start_time);

      std::cout << "Vocabulary created in " << duration.count() << " minutes"
                << "\n";
      std::cout << "Vocabulary size: " << vocabulary.size() << " words"
                << "\n";

      // Save vocabulary
      std::cout << "Saving vocabulary to: " << output_path << "\n";
      vocabulary.save(output_path);

      std::cout << "Vocabulary saved successfully!" << "\n";
      return true;

    } catch (const std::exception& e) {
      std::cerr << "Error creating vocabulary: " << e.what() << "\n";
      return false;
    }
  }
};

void printUsage() {
  std::cout << "SuperPoint Vocabulary Generator" << "\n";
  std::cout << "Usage: ./generate_sp_vocabulary <config_path> <model_dir> "
               "<image_dir> "\n "<output_vocab> [options]"
            << "\n";
  std::cout << "\n";
  std::cout << "Arguments:" << "\n";
  std::cout << "  config_path     Path to SuperPoint configuration file (e.g., "
               "utils/config.yaml)"
            << "\n";
  std::cout << "  model_dir       Directory containing SuperPoint model files "
               "(e.g., "\n "weights/)"
            << "\n";
  std::cout << "  image_dir       Directory containing training images"
            << "\n";
  std::cout << "  output_vocab    Output vocabulary file path (e.g., "
               "sp_vocabulary.yml.gz)"
            << "\n";
  std::cout << "\n";
  std::cout << "Options:" << "\n";
  std::cout << "  --k K                    Branching factor (default: 10)"
            << "\n";
  std::cout << "  --levels L               Depth levels (default: 6)"
            << "\n";
  std::cout
      << "  --max-images N           Maximum images to process (default: all)"
      << "\n";
  std::cout
      << "  --max-features-per-image Maximum features per image (default: 500)"
      << "\n";
  std::cout << "\n";
  std::cout << "Example:" << "\n";
  std::cout << "  ./generate_sp_vocabulary utils/config.yaml weights/ "
               "/dataset/images/ "\n "sp_vocabulary.yml.gz --k 10 --levels 6"
            << "\n";
}

int main(int argc, char** argv) {
  if (argc < 5) {
    printUsage();
    return 1;
  }

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
    if (i + 1 >= argc) break;

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
    std::cout << "Initializing SuperPoint Vocabulary Generator..." << "\n";
    SPVocabularyGenerator generator(config_path, model_dir);

    std::cout << "Processing images from: " << image_dir << "\n";
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<cv::Mat> all_descriptors;

    if (!generator.processImageDirectory(image_dir, all_keypoints,
                                         all_descriptors, max_images)) {
      std::cerr << "Failed to process images" << "\n";
      return 1;
    }

    std::cout << "Generating vocabulary..." << "\n";
    if (!generator.generateVocabulary(all_keypoints, all_descriptors,
                                      output_vocab, k, levels,
                                      max_features_per_image)) {
      std::cerr << "Failed to generate vocabulary" << "\n";
      return 1;
    }

    std::cout << "Vocabulary generation completed successfully!" << "\n";

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}