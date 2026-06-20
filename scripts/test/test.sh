#!/bin/bash
# Build the project (Debug) and run the CPU unit tests via CTest.
# Configure and build into build-core/ at the repo root, then run the tests
# labelled "unit" or "cpu" (the GPU and integration tests are skipped).
#
# Usage:  scripts/test/test.sh
set -e

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="${ROOT_DIR}/build-core"

# GPU-free core only (-DSUPERSLAM_BUILD_TRT=OFF); runs on any host without CUDA or TensorRT.
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Debug -DSUPERSLAM_BUILD_TRT=OFF
cmake --build "$BUILD_DIR" -- -j"$(getconf _NPROCESSORS_ONLN)"
ctest --test-dir "$BUILD_DIR" -L "unit|cpu"
