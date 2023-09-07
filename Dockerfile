# Use an appropriate base image, for example, Ubuntu
FROM yuefan2022/tensorrt-ubuntu20.04-cuda11.6

# Install Ninja
RUN apt-get ninja-build build-essential

# Install Ceres Solver
RUN apt-get update && apt-get install -y \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    && git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    cd ../../ && \
    rm -rf ceres-solver

# Copy source to the Docker Container
COPY . SuperSLAM/

# Set the default command to run when the container starts
CMD ["/bin/bash"]
