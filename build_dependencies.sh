#!/bin/bash

# Function to configure and build a third-party library with Ninja
build_library_with_ninja() {
    local library_name="$1"
    local build_type="$2"

    echo "Configuring and building thirdparty/$library_name with Ninja ..."
    cd "thirdparty/$library_name"
    mkdir -p .build
    cd .build
    cmake .. -GNinja -DCMAKE_BUILD_TYPE="$build_type"
    ninja -j"$(nproc)"
    cd ../../..
}

# Install prerequisites for Pangolin (if needed)
echo "Installing prerequisites for Pangolin ..."
thirdparty/Pangolin/scripts/install_prerequisites.sh recommended

# Build each library with Ninja in parallel
(
    build_library_with_ninja "yaml-cpp" "Release" &
    build_library_with_ninja "eigen" "Release" &
    build_library_with_ninja "Pangolin" "Release" &
    build_library_with_ninja "g2o" "Release" &
    build_library_with_ninja "opencv" "Release" &
    build_library_with_ninja "DBoW3" "Release"
)

# Wait for all background processes to complete
wait

echo "All third-party libraries have been configured and built with Ninja."
