#include <spdlog/spdlog.h>

#include <fstream>

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::info);
  spdlog::info("Testing basic functionality...");

  if (argc != 4) {
    spdlog::info("Usage: ./test_example_simple vocab_path settings_path sequence_path");
    return 1;
  }

  std::string vocab_path = argv[1];
  std::string settings_path = argv[2];
  std::string sequence_path = argv[3];

  spdlog::info("Checking files...");
  spdlog::info("Vocab: {}", vocab_path);
  spdlog::info("Settings: {}", settings_path);
  spdlog::info("Sequence: {}", sequence_path);

  // Check files exist
  if (!std::ifstream(vocab_path).good()) {
    spdlog::error("Cannot find vocabulary file: {}", vocab_path);
    return 1;
  } else {
    spdlog::info("✓ Vocabulary file found");
  }

  if (!std::ifstream(settings_path).good()) {
    spdlog::error("Cannot find settings file: {}", settings_path);
    return 1;
  } else {
    spdlog::info("✓ Settings file found");
  }

  std::string times_file = sequence_path + "/times.txt";
  if (!std::ifstream(times_file).good()) {
    spdlog::error("Cannot find times.txt: {}", times_file);
    return 1;
  } else {
    spdlog::info("✓ Times file found");
  }

  std::string sample_image = sequence_path + "/image_0/000000.png";
  if (!std::ifstream(sample_image).good()) {
    spdlog::error("Cannot find sample image: {}", sample_image);
    return 1;
  } else {
    spdlog::info("✓ Sample image found");
  }

  spdlog::info(
      "All basic checks passed! The issue is likely in SLAM initialization.");
  return 0;
}