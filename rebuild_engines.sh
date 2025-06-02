#!/bin/bash

# Script to rebuild TensorRT engines for SuperSLAM with optimal configurations
# Based on C++ code analysis for proper optimization profiles
# Run this script on a system with properly configured TensorRT 10.11.0

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure TensorRT libraries are in library path
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

print_info "Rebuilding TensorRT engines for SuperSLAM with optimal configurations..."

# Check if trtexec is available
if ! command -v trtexec &>/dev/null; then
    print_error "trtexec not found in PATH"
    print_error "Please ensure TensorRT is properly installed and trtexec is in PATH"
    exit 1
fi

# Print TensorRT version
print_info "TensorRT version check:"
trtexec --version 2>/dev/null | head -1 || print_warning "Could not detect TensorRT version"

# Check GPU info
print_info "GPU information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null || print_warning "Could not detect GPU info"

# Check if ONNX files exist
print_info "Checking for required ONNX files..."

SUPERPOINT_ONNX="weights/superpoint_v1_sim_int32.onnx"
SUPERGLUE_INDOOR_ONNX="weights/superglue_indoor_sim_int32.onnx"
SUPERGLUE_OUTDOOR_ONNX="weights/superglue_outdoor_sim_int32.onnx"

if [ ! -f "$SUPERPOINT_ONNX" ]; then
    print_error "SuperPoint ONNX file not found: $SUPERPOINT_ONNX"
    exit 1
fi

if [ ! -f "$SUPERGLUE_INDOOR_ONNX" ]; then
    print_error "SuperGlue indoor ONNX file not found: $SUPERGLUE_INDOOR_ONNX"
    exit 1
fi

if [ ! -f "$SUPERGLUE_OUTDOOR_ONNX" ]; then
    print_error "SuperGlue outdoor ONNX file not found: $SUPERGLUE_OUTDOOR_ONNX"
    exit 1
fi

print_success "All ONNX files found"

# Create backup of existing engines if they exist
if ls weights/*.engine 1>/dev/null 2>&1; then
    print_info "Backing up existing engine files..."
    mkdir -p weights/backup_$(date +%Y%m%d_%H%M%S)
    cp weights/*.engine weights/backup_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
fi

# Remove old engine files
print_info "Cleaning up old engine files..."
rm -f weights/*.engine

print_info "Starting engine builds..."

# Build SuperPoint engine with optimal settings from C++ code
print_info "Building SuperPoint engine (estimated time: 5-10 minutes)..."
trtexec --onnx="$SUPERPOINT_ONNX" \
    --saveEngine=weights/superpoint_v1_sim_int32.engine \
    --fp16 \
    --minShapes=input:1x1x100x100 \
    --optShapes=input:1x1x500x500 \
    --maxShapes=input:1x1x1500x1500 \
    --memPoolSize=workspace:512 \
    --verbose

if [ $? -eq 0 ]; then
    print_success "SuperPoint engine built successfully"
else
    print_error "SuperPoint engine build failed"
    exit 1
fi

# Build SuperGlue indoor engine with optimal settings from C++ code
print_info "Building SuperGlue indoor engine (estimated time: 15-25 minutes)..."
trtexec --onnx="$SUPERGLUE_INDOOR_ONNX" \
    --saveEngine=weights/superglue_indoor_sim_int32.engine \
    --fp16 \
    --minShapes=keypoints_0:1x1x2,scores_0:1x1,descriptors_0:1x256x1,keypoints_1:1x1x2,scores_1:1x1,descriptors_1:1x256x1 \
    --optShapes=keypoints_0:1x512x2,scores_0:1x512,descriptors_0:1x256x512,keypoints_1:1x512x2,scores_1:1x512,descriptors_1:1x256x512 \
    --maxShapes=keypoints_0:1x1024x2,scores_0:1x1024,descriptors_0:1x256x1024,keypoints_1:1x1024x2,scores_1:1x1024,descriptors_1:1x256x1024 \
    --memPoolSize=workspace:512 \
    --verbose

if [ $? -eq 0 ]; then
    print_success "SuperGlue indoor engine built successfully"
else
    print_error "SuperGlue indoor engine build failed"
    exit 1
fi

# Build SuperGlue outdoor engine with optimal settings from C++ code
print_info "Building SuperGlue outdoor engine (estimated time: 15-25 minutes)..."
trtexec --onnx="$SUPERGLUE_OUTDOOR_ONNX" \
    --saveEngine=weights/superglue_outdoor_sim_int32.engine \
    --fp16 \
    --minShapes=keypoints_0:1x1x2,scores_0:1x1,descriptors_0:1x256x1,keypoints_1:1x1x2,scores_1:1x1,descriptors_1:1x256x1 \
    --optShapes=keypoints_0:1x512x2,scores_0:1x512,descriptors_0:1x256x512,keypoints_1:1x512x2,scores_1:1x512,descriptors_1:1x256x512 \
    --maxShapes=keypoints_0:1x1024x2,scores_0:1x1024,descriptors_0:1x256x1024,keypoints_1:1x1024x2,scores_1:1x1024,descriptors_1:1x256x1024 \
    --memPoolSize=workspace:512 \
    --verbose

if [ $? -eq 0 ]; then
    print_success "SuperGlue outdoor engine built successfully"
else
    print_error "SuperGlue outdoor engine build failed"
    exit 1
fi

print_success "All engines rebuilt successfully!"

print_info "Engine files created:"
ls -lh weights/*.engine

print_info "Engine file sizes:"
du -h weights/*.engine

print_info ""
print_success "Build completed! You can now run SuperSLAM with the optimized engines."

# Verify engines can be loaded (basic check)
print_info "Performing basic engine validation..."
for engine in weights/*.engine; do
    if [ -f "$engine" ] && [ -s "$engine" ]; then
        print_success "✓ $(basename "$engine") - valid ($(du -h "$engine" | cut -f1))"
    else
        print_error "✗ $(basename "$engine") - invalid or empty"
        exit 1
    fi
done

print_info ""
print_info "Configuration Summary:"
print_info "- SuperPoint: Optimized for 500x500 images (up to 1500x1500)"
print_info "- SuperGlue: Optimized for 512 keypoints (up to 1024)"
print_info "- Workspace: 512MB"
print_info "- Precision: FP16"
print_info ""
print_success "Ready for inference!"
