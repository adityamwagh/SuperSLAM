echo "Building ROS nodes"

cd ros/ORB_SLAM2
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j"$(nproc)"