cmake_minimum_required(VERSION 3.20)
project(SuperSLAM)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Enable export of compile commands for tools like clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w -Wno-unused-variable -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -Wno-unused-variable -march=native")

# Add custom CMake modules path
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Set paths for third-party libraries
set(Torch_DIR "${PROJECT_SOURCE_DIR}/thirdparty/libtorch/share/cmake/Torch") # Update this path
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
set(Eigen3_DIR "/usr/lib/cmake/eigen3")
set(Pangolin_DIR "${PROJECT_SOURCE_DIR}/thirdparty/Pangolin/cmake/Pangolin")

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(Pangolin REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

# Add subdirectories for third-party libraries
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/DBoW3)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/DBoW2)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/Pangolin)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/g2o)
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/tensorrtbuffer)

# Define library output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Create the main library target using wildcard logic
file(GLOB SRC_FILES "src/*.cpp" "src/*.cc")

add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# Include directories for the main library
target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty
    ${PROJECT_SOURCE_DIR}/thirdparty/libtorch/include
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    /usr/include/x86_64-linux-gnu # TensorRT include directory
)

# Link directories for the main library
target_link_directories(${PROJECT_NAME} PUBLIC /usr/lib/x86_64-linux-gnu) # TensorRT library directory

# Link dependencies for the main library
target_link_libraries(${PROJECT_NAME} PUBLIC
    nvinfer
    nvonnxparser
    yaml-cpp
    tensorrtbuffer
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/lib/libDBoW3.so
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o.so
)

# Build examples
set(EXAMPLES_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/examples)

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXAMPLES_OUTPUT_DIR}/RGB-D)
add_executable(rgbd_tum examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

# Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXAMPLES_OUTPUT_DIR}/stereo)
add_executable(stereo_kitti examples/stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc examples/stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

# Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXAMPLES_OUTPUT_DIR}/monocular)
add_executable(mono_tum examples/monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti examples/monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc examples/monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})
