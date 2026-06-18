#!/bin/bash

# Rebuild the SuperSLAM TensorRT engines with optimal configurations.
# Run inside the superslam container. Engines are TRT-version and GPU specific.

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Colored output helpers
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Ensure TensorRT libraries are in library path
export LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
# trtexec ships at /usr/src/tensorrt/bin in the TensorRT image but isn't on PATH
export PATH=/usr/src/tensorrt/bin:$PATH

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

# Batch-dynamic SuperPoint ONNX (enables the batched stereo path: one {2,1,H,W} infer for
# L+R). Produced by utils/make_dynamic_batch_onnx.py (graph surgery on the deployed
# superpoint_dense_int32.onnx).
SUPERPOINT_ONNX="weights/superpoint_dense_dynamic_batch.onnx"
LIGHTGLUE_ONNX="weights/lightglue_superpoint.onnx"

if [ ! -f "$SUPERPOINT_ONNX" ]; then
    print_error "SuperPoint dynamic-batch ONNX not found: $SUPERPOINT_ONNX"
    print_error "Generate it (host, no system deps):"
    print_error "  uv run --with onnx python utils/make_dynamic_batch_onnx.py \\"
    print_error "      --in weights/superpoint_dense_int32.onnx --out $SUPERPOINT_ONNX"
    exit 1
fi

if [ ! -f "$LIGHTGLUE_ONNX" ]; then
    print_error "LightGlue ONNX file not found: $LIGHTGLUE_ONNX"
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

# Build the SuperPoint engine.
print_info "Building SuperPoint engine (estimated time: 5-10 minutes)..."
# Batch profile: batch 1-2 (mono extract + batched L+R stereo). H/W spans the datasets
# (KITTI 376x1241, EuRoC 480x752, TUM 480x640, TartanAir 640x640) up to 1080p (1080x1920) for
# real cameras. opt stays at the common 480x752 so runtime kernels are tuned for it. I/O formats:
# image input + scores output stay FP32 (host reads scores; preprocess uploads fp32); the dense
# `descriptors` grid is FP16, read directly by the on-device gather. Binding order is
# input=[input], outputs=[scores, descriptors].
trtexec --onnx="$SUPERPOINT_ONNX" \
    --saveEngine=weights/superpoint_dense_dynamic_batch_fp16.engine \
    --fp16 \
    --inputIOFormats=fp32:chw \
    --outputIOFormats=fp32:chw,fp16:chw \
    --minShapes=input:1x1x256x256 \
    --optShapes=input:2x1x480x752 \
    --maxShapes=input:2x1x1080x1920 \
    --memPoolSize=workspace:512 \
    --verbose

if [ $? -eq 0 ]; then
    print_success "SuperPoint engine built successfully"
else
    print_error "SuperPoint engine build failed"
    exit 1
fi

# Build the LightGlue engine.
print_info "Building LightGlue engine (estimated time: 10-20 minutes)..."
# I/O formats: kpts stay FP32 (small, host-normalized); desc0/desc1 are FP16, D2D-copied from
# the FP16 pool slots by the device match. matches0 stays INT32 (indices); mscores0 stays FP32
# (host reads it). Binding order: inputs [kpts0, desc0, kpts1, desc1], outputs [matches0, mscores0].
trtexec --onnx="$LIGHTGLUE_ONNX" \
    --saveEngine=weights/lightglue_superpoint_fp16.engine \
    --fp16 \
    --inputIOFormats=fp32:chw,fp16:chw,fp32:chw,fp16:chw \
    --outputIOFormats=int32:chw,fp32:chw \
    --minShapes=kpts0:1x1x2,desc0:1x1x256,kpts1:1x1x2,desc1:1x1x256 \
    --optShapes=kpts0:1x512x2,desc0:1x512x256,kpts1:1x512x2,desc1:1x512x256 \
    --maxShapes=kpts0:1x1024x2,desc0:1x1024x256,kpts1:1x1024x2,desc1:1x1024x256 \
    --memPoolSize=workspace:512 \
    --verbose

if [ $? -eq 0 ]; then
    print_success "LightGlue engine built successfully"
else
    print_error "LightGlue engine build failed"
    exit 1
fi

# Build EigenPlaces engine (loop-closure place recognition). Optional: only built when
# its ONNX is present (convert_eigenplaces_to_onnx.py). Fixed 512x512 RGB input.
EIGENPLACES_ONNX="weights/eigenplaces_resnet18_512.onnx"
if [ -f "$EIGENPLACES_ONNX" ]; then
    print_info "Building EigenPlaces engine (loop closure)..."
    trtexec --onnx="$EIGENPLACES_ONNX" \
        --saveEngine=weights/eigenplaces_resnet18_512_fp16.engine \
        --fp16 \
        --minShapes=image:1x3x512x512 \
        --optShapes=image:1x3x512x512 \
        --maxShapes=image:1x3x512x512 \
        --memPoolSize=workspace:512 \
        --verbose
    if [ $? -eq 0 ]; then
        print_success "EigenPlaces engine built successfully"
    else
        print_error "EigenPlaces engine build failed"
        exit 1
    fi
else
    print_warning "EigenPlaces ONNX not found ($EIGENPLACES_ONNX); skipping loop-closure engine."
    print_warning "Build it with: python utils/convert_eigenplaces_to_onnx.py --output_dir weights"
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
        print_success "OK $(basename "$engine") - valid ($(du -h "$engine" | cut -f1))"
    else
        print_error "X $(basename "$engine") - invalid or empty"
        exit 1
    fi
done

print_info ""
print_info "Configuration Summary:"
print_info "- SuperPoint: dynamic batch 1-2 (batched L+R stereo), opt 480x752, max 480x1280; FP16 desc I/O"
print_info "- LightGlue: 512 keypoints (up to 1024); FP16 desc I/O"
print_info "- Workspace: 512MB"
print_info "- Precision: FP16 (engine + descriptor I/O)"
print_info ""
print_success "Ready for inference!"
