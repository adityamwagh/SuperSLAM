cmake_minimum_required(VERSION 3.20)
project(SuperSLAM)

# Include FetchContent module
include(FetchContent)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_FLAGS="-allow-unsupported-compiler")
# set(CMAKE_CXX_COMPILER "/home/linuxbrew/.linuxbrew/bin/g++-11")
# set(CMAKE_C_COMPILER "/home/linuxbrew/.linuxbrew/bin/gcc-11")

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()


# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Enable export of compile commands for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w -Wno-unused-variable -march=native -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -Wno-unused-variable -march=native -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add debug and backtrace support
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -rdynamic")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -rdynamic")
endif()

# Add custom CMake modules path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

# Configure spdlog
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)

# Set PIC flag for spdlog when building as static library
set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Configure GTSAM
FetchContent_Declare(
  GTSAM
  GIT_REPOSITORY https://github.com/borglab/gtsam.git
  GIT_TAG 4.2.0
)
set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE BOOL "")
set(GTSAM_BUILD_TESTS OFF CACHE BOOL "")
set(GTSAM_BUILD_DOCS OFF CACHE BOOL "")
set(GTSAM_BUILD_UNSTABLE OFF CACHE BOOL "")
set(GTSAM_USE_SYSTEM_EIGEN ON CACHE BOOL "")
FetchContent_MakeAvailable(GTSAM)

# Configure Rerun
FetchContent_Declare(
  rerun_sdk
  URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

# Set PIC flag for rerun_sdk
set_target_properties(rerun_sdk PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Configure Torch
# FetchContent_Declare(
#  Torch
#  URL https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu118.zip
#)
# FetchContent_MakeAvailable(Torch)
#list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")
#find_package(Torch REQUIRED)


# list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")
# find_package(Torch REQUIRED)

# Add subdirectories for third-party libraries
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/DBoW3)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/tensorrtbuffer)

# Define library output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Create the main library target using wildcard logic
file(GLOB_RECURSE SRC_FILES
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/*.cc"
)
# Remove SuperPointLT since torch is not available
list(FILTER SRC_FILES EXCLUDE REGEX ".*SuperPointLT\\.cc$")

# Create the main library target
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# Include directories for the main library
target_include_directories(${PROJECT_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/thirdparty
  ${EIGEN3_INCLUDE_DIR}
  ${rerun_sdk_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  /usr/include # TensorRT include directory
)

# Link dependencies for the main library
target_link_libraries(${PROJECT_NAME} PUBLIC
  nvinfer
  nvonnxparser
  yaml-cpp
  spdlog::spdlog
  tensorrtbuffer
  rerun_sdk
  gtsam
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${EIGEN3_LIBS}
  ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/lib/libDBoW3.so
)

# Build examples
set(EXAMPLES_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/examples)

# KITTI Monocular examples with Rerun visualization
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXAMPLES_OUTPUT_DIR}/monocular)
add_executable(mono_kitti_rerun examples/monocular/mono_kitti_rerun.cc)
target_link_libraries(mono_kitti_rerun ${PROJECT_NAME})

# Stereo examples with Rerun visualization
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXAMPLES_OUTPUT_DIR}/stereo)
add_executable(stereo_kitti_rerun examples/stereo/stereo_kitti_rerun.cc)
target_link_libraries(stereo_kitti_rerun ${PROJECT_NAME})

# Stereo example without viewer (for debugging)
add_executable(stereo_kitti_no_viewer examples/stereo/stereo_kitti_no_viewer.cc)
target_link_libraries(stereo_kitti_no_viewer ${PROJECT_NAME})

# Test executable for SuperPoint/SuperGlue
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
# Commented out test executables that need to be updated
# add_executable(test_superpoint_superglue test_superpoint_superglue.cc)
# target_link_libraries(test_superpoint_superglue ${PROJECT_NAME})

# add_executable(test_superpoint_only test_superpoint_only.cc)
# target_link_libraries(test_superpoint_only ${PROJECT_NAME})

# add_executable(test_debug_matching test_debug_matching.cc)
# target_link_libraries(test_debug_matching ${PROJECT_NAME})

# add_executable(test_superpoint_cosine_matching test_superpoint_cosine_matching.cc)
# target_link_libraries(test_superpoint_cosine_matching ${PROJECT_NAME})

add_executable(test_simple_visualization test_simple_visualization.cc)
target_link_libraries(test_simple_visualization ${PROJECT_NAME})

# Simple test for debugging examples
add_executable(test_example_simple test_example_simple.cc)
target_link_libraries(test_example_simple spdlog::spdlog)

# Vocabulary generation tool (needs updating)
# add_executable(generate_sp_vocabulary tools/generate_sp_vocabulary.cc)
# target_link_libraries(generate_sp_vocabulary ${PROJECT_NAME})
