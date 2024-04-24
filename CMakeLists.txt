cmake_minimum_required(VERSION 3.2)
project(SuperSLAM)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# # PACKAGES
# If cmake cannot find opencv or eigen3, try to set the path manually as following.
set(Torch_DIR "/pytorch/torch/share/cmake/Torch") # libtorch
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4/") # opencv
# set(Eigen3_DIR "/usr/lib/cmake/eigen3")    # eigen3
find_package(OpenCV 4.5.4 QUIET REQUIRED)

if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)

    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

# Find all the fun packages
find_package(Eigen3 3.1.0 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
find_package(Torch REQUIRED)

# # COMPILERS
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++17, C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

# if(COMPILER_SUPPORTS_CXX17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
# add_definitions(-DCOMPILEDWITHC17)
# message(STATUS "Using flag -std=c++17.")
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCOMPILEDWITHC14)
    message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# # SOURCE CODE
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}

    # ${PROJECT_SOURCE_DIR}/thirdparty/libtorch/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    src/Random.cpp
    src/Timestamp.cpp

    src/System.cc
    src/Tracking.cc
    src/LocalMapping.cc
    src/LoopClosing.cc

    src/SPextractor.cc
    src/SPmatcher.cc
    src/SuperPoint.cc

    src/FrameDrawer.cc
    src/Converter.cc
    src/MapPoint.cc
    src/KeyFrame.cc
    src/Map.cc
    src/MapDrawer.cc
    src/Optimizer.cc
    src/PnPsolver.cc
    src/Frame.cc
    src/KeyFrameDatabase.cc
    src/Sim3Solver.cc
    src/Initializer.cc
    src/Viewer.cc
)

target_link_libraries(${PROJECT_NAME}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/lib/libDBoW3.so
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o.so
)
