# SuperSLAM - top-level entry point.
#
#   make help              # list every target
#
# Docker / C++ / engine steps go through `docker compose` (see compose.yaml).
# Python steps (dataset + weight/ONNX downloads) are run SEPARATELY with `uv run` - see the
# README; they are intentionally NOT make targets.
#
# Common knobs (override on the command line, e.g. `make run-superslam-kitti SEQUENCE=05`):
#   TENSORRT=10|11     which TensorRT image to use (default 10)
#   DATASETS=PATH      host datasets dir, mounted at /datasets (default ~/datasets)
#   SEQUENCE=00        KITTI sequence id
#   EUROC=MH_01_easy   EuRoC sequence dir
#   TUM=rgbd_dataset_freiburg2_xyz   TUM sequence dir

SHELL    := /bin/bash
TENSORRT ?= 10
DATASETS ?= $(HOME)/datasets
SEQUENCE ?= 00
EUROC    ?= MH_01_easy
TUM      ?= rgbd_dataset_freiburg2_xyz
IMAGE    := superslam:tensorrt$(TENSORRT)-ubuntu24.04
COMPOSE  := TENSORRT=$(TENSORRT) DATASETS=$(DATASETS) docker compose
RUN      := $(COMPOSE) run --rm superslam bash -lc

# Print a blue banner before each step (verbose, so a beginner sees what is happening).
banner = @printf '\n\033[1;34m==> %s\033[0m\n' "$(1)"

.DEFAULT_GOAL := help

.PHONY: help
help: ## List all targets
	@printf '\nSuperSLAM make targets  (TensorRT %s, image %s):\n\n' "$(TENSORRT)" "$(IMAGE)"
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-30s\033[0m %s\n", $$1, $$2}'
	@printf '\nPython steps (run with uv, NOT make):\n'
	@printf '  uv run --with requests python scripts/models/download_onnx_engine_<model>.py\n'
	@printf '  uv run python scripts/datasets/download_<dataset>.py --out $(DATASETS)/<dataset>\n'
	@printf '  (evaluate-* targets wrap uv --with evo for you)\n\n'

# ---- Build -----------------------------------------------------------------------------------

.PHONY: build-image-tensorrt10
build-image-tensorrt10: ## Build the Docker image (TensorRT 10, Ubuntu 24.04)
	$(call banner,Building image superslam:tensorrt10-ubuntu24.04)
	TENSORRT=10 DATASETS=$(DATASETS) docker compose build

.PHONY: build-image-tensorrt11
build-image-tensorrt11: ## Build the Docker image (TensorRT 11, Ubuntu 24.04)
	$(call banner,Building image superslam:tensorrt11-ubuntu24.04)
	TENSORRT=11 DATASETS=$(DATASETS) docker compose build

.PHONY: build-engines-tensorrt10
build-engines-tensorrt10: ## Build TensorRT 10 engines for this GPU from the weights ONNX
	$(call banner,Building TensorRT 10 engines into weights)
	TENSORRT=10 DATASETS=$(DATASETS) docker compose run --rm superslam bash -lc 'bash scripts/rebuild_engines.sh'

.PHONY: build-engines-tensorrt11
build-engines-tensorrt11: ## Build TensorRT 11 engines for this GPU from the weights ONNX
	$(call banner,Building TensorRT 11 engines into weights)
	TENSORRT=11 DATASETS=$(DATASETS) docker compose run --rm superslam bash -lc 'bash scripts/rebuild_engines.sh'

.PHONY: build-superslam
build-superslam: ## Compile the SuperSLAM library + examples in the container
	$(call banner,Compiling SuperSLAM into build-tensorrt/)
	$(RUN) 'cmake -S . -B build-tensorrt -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-tensorrt -j"$$(nproc)"'

.PHONY: shell
shell: ## Open an interactive shell in the container
	$(call banner,Shell in $(IMAGE))
	$(COMPOSE) run --rm superslam bash

.PHONY: test-superslam
test-superslam: ## Build + run the GPU-free core unit tests (host, no GPU)
	$(call banner,Core unit tests (GPU-free))
	bash scripts/test/test.sh

# ---- Run -------------------------------------------------------------------------------------

.PHONY: run-superslam-kitti
run-superslam-kitti: ## Run KITTI (SEQUENCE=$(SEQUENCE)) writes results/kitti/$(SEQUENCE).txt
	$(call banner,Running KITTI sequence $(SEQUENCE))
	@mkdir -p results/kitti
	$(RUN) 'case $(SEQUENCE) in 00|01|02) Y=KITTI00-02;; 03) Y=KITTI03;; *) Y=KITTI04-12;; esac; SUPERSLAM_ENABLE_LOOP=1 ./examples/kitti examples/stereo/$$Y.yaml /datasets/kitti/dataset/sequences/$(SEQUENCE) --no-viewer && cp -f CameraTrajectory_kitti.txt results/kitti/$(SEQUENCE).txt'

.PHONY: run-superslam-euroc
run-superslam-euroc: ## Run EuRoC (EUROC=$(EUROC)) writes results/euroc/$(EUROC).txt
	$(call banner,Running EuRoC sequence $(EUROC))
	@mkdir -p results/euroc
	$(RUN) 'SUPERSLAM_ENABLE_LOOP=1 ./examples/euroc examples/stereo/EuRoC.yaml /datasets/euroc/$(EUROC) --no-viewer && cp -f CameraTrajectory_euroc.txt results/euroc/$(EUROC).txt'

.PHONY: run-superslam-tum
run-superslam-tum: ## Run TUM RGB-D (TUM=$(TUM)) writes results/tum/$(TUM).txt
	$(call banner,Running TUM sequence $(TUM))
	@mkdir -p results/tum
	$(RUN) 'case $(TUM) in *freiburg1*) Y=TUM1;; *freiburg2*) Y=TUM2;; *) Y=TUM3;; esac; SUPERSLAM_ENABLE_LOOP=1 ./examples/rgbd_tum examples/rgbd/$$Y.yaml /datasets/tum/$(TUM) --no-viewer && cp -f CameraTrajectory_tum.txt results/tum/$(TUM).txt'

# ---- Evaluate (wraps `uv run --with evo`) ----------------------------------------------------

.PHONY: evaluate-superslam-kitti
evaluate-superslam-kitti: ## Evaluate KITTI ATE/RPE vs ground truth (uv + evo)
	$(call banner,Evaluating KITTI (uv --with evo))
	uv run --with evo python scripts/benchmarks/evaluate_kitti.py --traj-dir results/kitti --data-dir $(DATASETS)/kitti/dataset

.PHONY: evaluate-superslam-euroc
evaluate-superslam-euroc: ## Evaluate EuRoC ATE/RPE vs ground truth (uv + evo)
	$(call banner,Evaluating EuRoC (uv --with evo))
	uv run --with evo python scripts/benchmarks/evaluate_euroc.py --traj-dir results/euroc --data-dir $(DATASETS)/euroc

.PHONY: evaluate-superslam-tum
evaluate-superslam-tum: ## Evaluate TUM ATE/RPE vs ground truth (uv + evo)
	$(call banner,Evaluating TUM (uv --with evo))
	uv run --with evo python scripts/benchmarks/evaluate_tum.py --traj-dir results/tum --data-dir $(DATASETS)/tum

# ---- Clean (idempotent) ----------------------------------------------------------------------

.PHONY: clean-engines
clean-engines: ## Remove built TensorRT engines (weights/*.engine)
	$(call banner,Removing weights/*.engine)
	rm -f weights/*.engine

.PHONY: clean-weights
clean-weights: ## Remove generated/downloaded ONNX (keeps the committed source ONNX)
	$(call banner,Removing generated ONNX (keeping superpoint_dense_int32.onnx))
	rm -f weights/superpoint_dense_dynamic_batch.onnx \
	      weights/lightglue_superpoint.onnx weights/lightglue_superpoint.onnx.data \
	      weights/eigenplaces_resnet18_512.onnx weights/eigenplaces_resnet18_512.onnx.data

.PHONY: clean-build
clean-build: ## Remove build directories
	$(call banner,Removing build directories)
	rm -rf build build-core build-tensorrt build-rel

.PHONY: clean-results
clean-results: ## Remove evaluation outputs (results/*)
	$(call banner,Removing results/*)
	rm -rf results/*

.PHONY: clean-images
clean-images: ## Remove the SuperSLAM Docker images
	$(call banner,Removing SuperSLAM docker images)
	-docker image rm superslam:tensorrt10-ubuntu24.04 superslam:tensorrt11-ubuntu24.04 2>/dev/null || true

.PHONY: clean-all
clean-all: clean-engines clean-weights clean-build clean-results ## Clean everything except docker images
	$(call banner,Clean complete (images kept; use clean-images to remove them))
