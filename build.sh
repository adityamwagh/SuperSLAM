#!/bin/bash

set -e

echo "-- Extracting the loop closure vocabulary to $(pwd)/vocabulary"
cd vocabulary
tar -xf ORBvoc.txt.tar.gz
echo "-- The vocabulary has been extracted to $(pwd)"
cd ..

echo "-- Building libSuperSLAM.so"
mkdir -p build
cd build
cmake .. -GNinja
ninja
echo "-- Finished building libSuperSLAM.so"
echo "-- The library has been built in ./lib/libSuperSLAM.so"
