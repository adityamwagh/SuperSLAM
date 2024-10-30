echo "Building SuperSLAM"
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j"$(nproc)"
