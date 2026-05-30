cmake_minimum_required(VERSION 3.20)
project(SuperSLAM)

# Include FetchContent module
include(FetchContent)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Build options
option(BUILD_EXAMPLES "Build SuperSLAM example binaries" ON)
# The TensorRT/CUDA engines (SuperPoint, LightGlue, EigenPlaces) and the
# SuperSLAM facade are split into a separate library from the GPU-free SLAM core
# so the core and its unit tests build without CUDA/TensorRT (host CI, the
# Docker test stage). Auto-on when CUDA is available; force off for core-only
# builds.
option(SUPERSLAM_BUILD_TRT
       "Build the TensorRT/CUDA engines and the SuperSLAM facade" ON)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# GPU architectures. Default covers Ampere (86), Ada (89) and Blackwell (120,
# e.g. RTX PRO / RTX 50-series). Override with -DCMAKE_CUDA_ARCHITECTURES=... ;
# build engines for the matching arch (TensorRT engines are arch-specific).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89 120)
endif()
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom CUDA kernels (e.g. on-device descriptor gather) need nvcc; enable the
# CUDA language only for the TRT build so the GPU-free core still configures
# without a toolkit.
if(SUPERSLAM_BUILD_TRT)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Build all targets (incl. fetched deps like spdlog, gtsam, rerun_sdk,
# googletest) as position-independent. libSuperSLAM is a shared object, so every
# static archive linked into it must be -fPIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Choose the type of build." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# Enable export of compile commands for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Target CPU ISA. `native` is fastest for local/bare-host builds but bakes the
# build host's CPU into the binary -- wrong for a portable/shipped Docker image.
# Override with -DSUPERSLAM_MARCH=x86-64-v3 (or another baseline) for
# distributable images.
set(SUPERSLAM_MARCH
    "native"
    CACHE STRING
          "Value for -march (native for local; x86-64-v3 for portable images)")
set(CMAKE_C_FLAGS
    "${CMAKE_C_FLAGS} -w -Wno-unused-variable -march=${SUPERSLAM_MARCH} -g")
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -w -Wno-unused-variable -march=${SUPERSLAM_MARCH} -g")

# Add debug and backtrace support
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -rdynamic")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -rdynamic")
endif()

# Opt-in sanitizers for debugging memory and undefined-behavior bugs. Pass a
# comma list, for example -DSUPERSLAM_SANITIZE=address,undefined for ASan and
# UBSan.
set(SUPERSLAM_SANITIZE
    ""
    CACHE STRING "Comma list of sanitizers (address, undefined, thread, ...)")
if(SUPERSLAM_SANITIZE)
  set(_san_flags "-fsanitize=${SUPERSLAM_SANITIZE} -fno-omit-frame-pointer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_san_flags}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_san_flags}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_san_flags}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${_san_flags}")
endif()

# Add custom CMake modules path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Find dependencies. OpenCV + Eigen are needed by the GPU-free core; CUDA +
# yaml-cpp are only needed by the TensorRT engines / facade.
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
if(SUPERSLAM_BUILD_TRT)
  find_package(CUDA REQUIRED)
  find_package(yaml-cpp REQUIRED)
endif()

# Dependencies are dual-mode: prefer a prebuilt install (the Docker base image
# builds GTSAM once into /usr/local, shared by the trt10/trt11 variants), else
# FetchContent (bare-host builds). The fetched list is assembled and built in
# one MakeAvailable call.
set(FETCH_LIST "")

# GTSAM 4.2 (the heavy one — prebuilt in the Docker base image when available).
find_package(GTSAM 4.2 QUIET)
if(NOT GTSAM_FOUND)
  set(GTSAM_BUILD_EXAMPLES_ALWAYS
      OFF
      CACHE BOOL "")
  set(GTSAM_BUILD_TESTS
      OFF
      CACHE BOOL "")
  set(GTSAM_BUILD_DOCS
      OFF
      CACHE BOOL "")
  set(GTSAM_BUILD_UNSTABLE
      ON
      CACHE BOOL "")
  set(GTSAM_USE_SYSTEM_EIGEN
      ON
      CACHE BOOL "")
  FetchContent_Declare(
    GTSAM
    GIT_REPOSITORY https://github.com/borglab/gtsam.git
    GIT_TAG 4.2.0
    GIT_SHALLOW TRUE)
  list(APPEND FETCH_LIST GTSAM)
else()
  message(STATUS "Using prebuilt GTSAM ${GTSAM_VERSION}")
endif()

# spdlog (apt-installable; prefer the system package).
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
    GIT_SHALLOW TRUE)
  list(APPEND FETCH_LIST spdlog)
else()
  message(STATUS "Using prebuilt spdlog")
endif()

# Rerun C++ SDK. Pin 0.22.1: `latest` drifts and breaks the API
# (SeriesLine/Scalar were renamed to plural in 0.23). Prefer a prebuilt SDK at
# RERUN_SDK_ROOT (Docker base).
if(DEFINED ENV{RERUN_SDK_ROOT})
  find_package(rerun_sdk REQUIRED PATHS $ENV{RERUN_SDK_ROOT} NO_DEFAULT_PATH)
  message(STATUS "Using prebuilt rerun_sdk at $ENV{RERUN_SDK_ROOT}")
else()
  FetchContent_Declare(
    rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/0.22.1/rerun_cpp_sdk.zip
  )
  list(APPEND FETCH_LIST rerun_sdk)
endif()

# GoogleTest + GoogleMock (cheap; always fetched).
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  GIT_SHALLOW TRUE)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
list(APPEND FETCH_LIST googletest)

if(FETCH_LIST)
  FetchContent_MakeAvailable(${FETCH_LIST})
endif()

# PIC for fetched static libs linked into the shared core (no-op for imported
# targets).
if(TARGET spdlog)
  set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
if(rerun_sdk_SOURCE_DIR AND TARGET rerun_sdk)
  set_target_properties(rerun_sdk PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Sources split into the GPU-free SLAM core and the TensorRT engines + facade.
# TRT_SRC needs CUDA/TensorRT; everything else is core (GTSAM + OpenCV only).
set(TRT_SRC
    ${PROJECT_SOURCE_DIR}/src/SuperPoint.cc
    ${PROJECT_SOURCE_DIR}/src/LightGlue.cc
    ${PROJECT_SOURCE_DIR}/src/EigenPlaces.cc
    ${PROJECT_SOURCE_DIR}/src/DescriptorGather.cu
    ${PROJECT_SOURCE_DIR}/src/DescriptorPool.cc
    ${PROJECT_SOURCE_DIR}/src/SuperSLAM.cc)

file(GLOB_RECURSE SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp"
     "${PROJECT_SOURCE_DIR}/src/*.cc")
# SuperPointLT needs torch (unavailable); the TRT sources go in the TRT library.
list(FILTER SRC_FILES EXCLUDE REGEX ".*SuperPointLT\\.cc$")
set(CORE_SRC ${SRC_FILES})
list(REMOVE_ITEM CORE_SRC ${TRT_SRC})

# --- GPU-free core library (GTSAM + OpenCV; unit-testable without
# CUDA/TensorRT) ---
add_library(${PROJECT_NAME}_core SHARED ${CORE_SRC})
# Lowercase library file names (Unix convention): libsuperslam_core.so /
# libsuperslam.so. The CMake target names stay ${PROJECT_NAME}* so link
# references are unchanged.
set_target_properties(${PROJECT_NAME}_core PROPERTIES OUTPUT_NAME
                                                      superslam_core)
target_include_directories(
  ${PROJECT_NAME}_core
  PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include
         ${PROJECT_SOURCE_DIR}/thirdparty ${EIGEN3_INCLUDE_DIR}
         ${rerun_sdk_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
  ${PROJECT_NAME}_core
  PUBLIC spdlog::spdlog
         rerun_sdk
         gtsam
         gtsam_unstable
         Threads::Threads
         ${OpenCV_LIBRARIES}
         ${EIGEN3_LIBS})

# --- Full library: core + TensorRT engines + facade (needs CUDA/TensorRT) ---
if(SUPERSLAM_BUILD_TRT)
  add_library(${PROJECT_NAME} SHARED ${TRT_SRC})
  set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME superslam)
  target_include_directories(
    ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include
           ${YAML_CPP_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} /usr/include)
  target_link_libraries(
    ${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_core nvinfer nvonnxparser yaml-cpp
                           ${CUDA_LIBRARIES})
endif()

# Build examples. One binary per dataset (kitti, euroc); the Rerun viewer is on
# by default and toggled with --no-viewer. All binaries land in examples/.
if(BUILD_EXAMPLES AND SUPERSLAM_BUILD_TRT)
  set(EXAMPLES_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/examples)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXAMPLES_OUTPUT_DIR})

  add_executable(kitti examples/stereo/kitti.cc)
  target_link_libraries(kitti ${PROJECT_NAME})
  target_include_directories(kitti PRIVATE ${EXAMPLES_OUTPUT_DIR})

  add_executable(euroc examples/stereo/euroc.cc)
  target_link_libraries(euroc ${PROJECT_NAME})
  target_include_directories(euroc PRIVATE ${EXAMPLES_OUTPUT_DIR})

  add_executable(benchmark examples/stereo/benchmark.cc)
  target_link_libraries(benchmark ${PROJECT_NAME})
  target_include_directories(benchmark PRIVATE ${EXAMPLES_OUTPUT_DIR})

  add_executable(rgbd_tum examples/rgbd/tum.cc)
  target_link_libraries(rgbd_tum ${PROJECT_NAME})
  target_include_directories(rgbd_tum PRIVATE ${EXAMPLES_OUTPUT_DIR})

  add_executable(tartan examples/stereo/tartan.cc)
  target_link_libraries(tartan ${PROJECT_NAME})
  target_include_directories(tartan PRIVATE ${EXAMPLES_OUTPUT_DIR})
endif()

# Test executables build into the build tree, not the source root.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_pose_optimization_factors
               tests/test_pose_optimization_factors.cc)
target_link_libraries(test_pose_optimization_factors ${PROJECT_NAME}_core
                      GTest::gtest_main GTest::gmock_main)

include(CTest)
add_test(NAME test_pose_optimization_factors
         COMMAND test_pose_optimization_factors)
set_tests_properties(test_pose_optimization_factors PROPERTIES LABELS
                                                               "unit;cpu")

add_executable(test_window_smoother tests/test_window_smoother.cc)
target_link_libraries(test_window_smoother ${PROJECT_NAME}_core
                      GTest::gtest_main GTest::gmock_main)
add_test(NAME test_window_smoother COMMAND test_window_smoother)
set_tests_properties(test_window_smoother PROPERTIES LABELS "unit;cpu")

add_executable(test_frame_tracker tests/test_frame_tracker.cc)
target_link_libraries(test_frame_tracker ${PROJECT_NAME}_core GTest::gtest_main
                      GTest::gmock_main)
add_test(NAME test_frame_tracker COMMAND test_frame_tracker)
set_tests_properties(test_frame_tracker PROPERTIES LABELS "unit;cpu")

add_executable(test_keyframe_gate tests/test_keyframe_gate.cc)
target_link_libraries(test_keyframe_gate ${PROJECT_NAME}_core GTest::gtest_main
                      GTest::gmock_main)
add_test(NAME test_keyframe_gate COMMAND test_keyframe_gate)
set_tests_properties(test_keyframe_gate PROPERTIES LABELS "unit;cpu")

add_executable(test_stereo_frame tests/test_stereo_frame.cc)
target_link_libraries(test_stereo_frame ${PROJECT_NAME}_core GTest::gtest_main
                      GTest::gmock_main)
add_test(NAME test_stereo_frame COMMAND test_stereo_frame)
set_tests_properties(test_stereo_frame PROPERTIES LABELS "unit;cpu")

add_executable(test_stereo_frontend tests/test_stereo_frontend.cc)
target_link_libraries(test_stereo_frontend ${PROJECT_NAME}_core
                      GTest::gtest_main GTest::gmock_main)
add_test(NAME test_stereo_frontend COMMAND test_stereo_frontend)
set_tests_properties(test_stereo_frontend PROPERTIES LABELS "unit;cpu")

add_executable(test_vo_convention tests/test_vo_convention.cc)
target_link_libraries(test_vo_convention ${PROJECT_NAME}_core GTest::gtest_main
                      GTest::gmock_main)
add_test(NAME test_vo_convention COMMAND test_vo_convention)
set_tests_properties(test_vo_convention PROPERTIES LABELS "unit;cpu")

add_executable(test_global_pose_graph tests/test_global_pose_graph.cc)
target_link_libraries(test_global_pose_graph ${PROJECT_NAME}_core
                      GTest::gtest_main GTest::gmock_main)
add_test(NAME test_global_pose_graph COMMAND test_global_pose_graph)
set_tests_properties(test_global_pose_graph PROPERTIES LABELS "unit;cpu")

add_executable(test_keyframe_database tests/test_keyframe_database.cc)
target_link_libraries(test_keyframe_database ${PROJECT_NAME}_core
                      GTest::gtest_main GTest::gmock_main)
add_test(NAME test_keyframe_database COMMAND test_keyframe_database)
set_tests_properties(test_keyframe_database PROPERTIES LABELS "unit;cpu")

add_executable(test_place_recognizer tests/test_place_recognizer.cc)
target_link_libraries(test_place_recognizer ${PROJECT_NAME}_core
                      GTest::gtest_main GTest::gmock_main)
add_test(NAME test_place_recognizer COMMAND test_place_recognizer)
set_tests_properties(test_place_recognizer PROPERTIES LABELS "unit;cpu")

add_executable(test_loop_closer tests/test_loop_closer.cc)
target_link_libraries(test_loop_closer ${PROJECT_NAME}_core GTest::gtest_main
                      GTest::gmock_main)
add_test(NAME test_loop_closer COMMAND test_loop_closer)
set_tests_properties(test_loop_closer PROPERTIES LABELS "unit;cpu")

add_executable(test_vo_loop_closure tests/test_vo_loop_closure.cc)
target_link_libraries(test_vo_loop_closure ${PROJECT_NAME}_core
                      GTest::gtest_main GTest::gmock_main)
add_test(NAME test_vo_loop_closure COMMAND test_vo_loop_closure)
set_tests_properties(test_vo_loop_closure PROPERTIES LABELS "unit;cpu")

add_executable(test_sparse_map tests/test_sparse_map.cc)
target_link_libraries(test_sparse_map ${PROJECT_NAME}_core GTest::gtest_main
                      GTest::gmock_main)
add_test(NAME test_sparse_map COMMAND test_sparse_map)
set_tests_properties(test_sparse_map PROPERTIES LABELS "unit;cpu")

add_executable(test_descriptor_pool tests/test_descriptor_pool.cc)
target_link_libraries(test_descriptor_pool ${PROJECT_NAME}_core
                      GTest::gtest_main GTest::gmock_main)
add_test(NAME test_descriptor_pool COMMAND test_descriptor_pool)
set_tests_properties(test_descriptor_pool PROPERTIES LABELS "unit;cpu")
