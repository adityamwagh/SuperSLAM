#!/bin/bash

# SuperPoint Vocabulary Generation Tool Build Script
# This script builds the vocabulary generation tool independently

echo "Building SuperPoint Vocabulary Generation Tool..."

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the SuperSLAM root directory"
    exit 1
fi

# Create tools build directory
TOOLS_BUILD_DIR="build_tools"
mkdir -p $TOOLS_BUILD_DIR
cd $TOOLS_BUILD_DIR

# Configure CMake for vocabulary tool
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(SPVocabularyTool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/include  # TensorRT
)

# Add source files
set(VOCAB_TOOL_SOURCES
    ../tools/generate_sp_vocabulary.cc
    ../src/SPBowVector.cc
    ../src/SuperPointTRT.cc
    ../src/Random.cpp
    ../src/Timestamp.cpp
)

# Create executable
add_executable(generate_sp_vocabulary ${VOCAB_TOOL_SOURCES})

# Link libraries
target_link_libraries(generate_sp_vocabulary
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    yaml-cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/DBoW3/lib/libDBoW3.so
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/tensorrtbuffer/lib/libtensorrtbuffer.so
)

# Set output directory
set_target_properties(generate_sp_vocabulary PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
EOF

# Build the tool
echo "Configuring CMake..."
cmake .

echo "Building vocabulary generation tool..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "✅ Vocabulary generation tool built successfully!"
    echo "📍 Executable location: $(pwd)/../generate_sp_vocabulary"
    echo ""
    echo "Usage:"
    echo "  ./generate_sp_vocabulary <config_path> <model_dir> <image_dir> <output_vocab> [options]"
    echo ""
    echo "Example:"
    echo "  ./generate_sp_vocabulary utils/config.yaml weights/ /path/to/images/ sp_vocabulary.yml.gz"
    echo ""
    echo "Options:"
    echo "  --k 10                    Branching factor"
    echo "  --levels 6                Depth levels"
    echo "  --max-images 1000         Maximum images to process"
    echo "  --max-features-per-image 500  Maximum features per image"
else
    echo "❌ Build failed!"
    exit 1
fi

# Return to original directory
cd ..

echo "Build complete! 🎉"
