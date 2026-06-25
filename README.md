# SuperSLAM: Accurate Real-Time SLAM with Deep Learned Features

> Alpha software.

SuperSLAM is a real-time stereo and RGB-D visual SLAM system. SuperPoint detects features and
LightGlue matches them on a TensorRT FP16 backend, with a GTSAM optimization core and pose-graph
loop closure.

## Results

Run on an NVIDIA RTX PRO 1000 (Blackwell, laptop, 8 GB) with an Intel Core Ultra 5 235H, TensorRT
10.11, CUDA 12.9, FP16. `fps` is the per-frame tracking rate (front-end plus window smoother); every
sequence runs above its camera rate (KITTI 10 Hz, EuRoC 20 Hz, TUM 30 Hz). ATE is SE3-aligned RMSE.
Per-sequence trajectory plots are in [PLOTS.md](PLOTS.md).

### KITTI (stereo, window 10)

<table width="100%">
<thead><tr><th align="left">seq</th><th>ATE RMSE (m)</th><th>ATE mean (m)</th><th>RPE RMSE (m)</th><th>t_rel (%)</th><th>r_rel (deg/m)</th><th>fps</th></tr></thead>
<tbody>
<tr><td align="left">00</td><td>1.582</td><td>1.238</td><td>0.398</td><td>0.76</td><td>0.0034</td><td>79</td></tr>
<tr><td align="left">01</td><td>375.893</td><td>340.771</td><td>2.607</td><td>46.10</td><td>0.0322</td><td>48</td></tr>
<tr><td align="left">02</td><td>7.908</td><td>5.945</td><td>0.395</td><td>1.02</td><td>0.0034</td><td>71</td></tr>
<tr><td align="left">03</td><td>2.291</td><td>1.687</td><td>0.290</td><td>4.66</td><td>0.0153</td><td>79</td></tr>
<tr><td align="left">04</td><td>0.646</td><td>0.575</td><td>0.121</td><td>0.72</td><td>0.0012</td><td>72</td></tr>
<tr><td align="left">05</td><td>4.187</td><td>2.719</td><td>0.438</td><td>1.19</td><td>0.0046</td><td>71</td></tr>
<tr><td align="left">06</td><td>2.222</td><td>1.954</td><td>0.468</td><td>1.56</td><td>0.0079</td><td>68</td></tr>
<tr><td align="left">07</td><td>15.857</td><td>8.421</td><td>3.935</td><td>6.51</td><td>0.0356</td><td>77</td></tr>
<tr><td align="left">08</td><td>6.946</td><td>5.897</td><td>0.286</td><td>1.34</td><td>0.0047</td><td>73</td></tr>
<tr><td align="left">09</td><td>3.991</td><td>3.626</td><td>0.095</td><td>1.80</td><td>0.0065</td><td>71</td></tr>
<tr><td align="left">10</td><td>2.113</td><td>1.745</td><td>0.089</td><td>0.79</td><td>0.0063</td><td>78</td></tr>
</tbody>
</table>

Sequence 01 is the highway: distant features give no parallax, scale drifts, and the run diverges (a
known failure for feature SLAM).

### EuRoC (stereo)

<table width="100%">
<thead><tr><th align="left">seq</th><th>ATE RMSE (m)</th><th>ATE mean (m)</th><th>ATE max (m)</th><th>RPE RMSE (m)</th><th>fps</th></tr></thead>
<tbody>
<tr><td align="left">MH_01_easy</td><td>0.059</td><td>0.048</td><td>0.225</td><td>0.910</td><td>145</td></tr>
<tr><td align="left">MH_02_easy</td><td>0.067</td><td>0.062</td><td>0.133</td><td>0.848</td><td>87</td></tr>
<tr><td align="left">MH_03_medium</td><td>0.104</td><td>0.092</td><td>0.257</td><td>1.155</td><td>23</td></tr>
<tr><td align="left">MH_04_difficult</td><td>0.253</td><td>0.243</td><td>0.416</td><td>1.187</td><td>44</td></tr>
<tr><td align="left">MH_05_difficult</td><td>0.090</td><td>0.070</td><td>0.608</td><td>1.126</td><td>158</td></tr>
<tr><td align="left">V1_01_easy</td><td>0.102</td><td>0.094</td><td>0.184</td><td>0.913</td><td>182</td></tr>
<tr><td align="left">V1_02_medium</td><td>0.068</td><td>0.062</td><td>0.161</td><td>0.998</td><td>158</td></tr>
<tr><td align="left">V1_03_difficult</td><td>0.126</td><td>0.103</td><td>0.585</td><td>0.890</td><td>173</td></tr>
<tr><td align="left">V2_01_easy</td><td>0.093</td><td>0.079</td><td>0.300</td><td>0.697</td><td>162</td></tr>
<tr><td align="left">V2_02_medium</td><td>0.129</td><td>0.110</td><td>0.405</td><td>0.892</td><td>122</td></tr>
<tr><td align="left">V2_03_difficult</td><td colspan="4">diverged</td><td>186</td></tr>
</tbody>
</table>

### TUM RGB-D

#### Standard

<table width="100%">
<thead><tr><th align="left">seq</th><th>ATE RMSE (m)</th><th>ATE mean (m)</th><th>ATE max (m)</th><th>RPE RMSE (m)</th><th>fps</th></tr></thead>
<tbody>
<tr><td align="left">fr1_desk</td><td>0.079</td><td>0.067</td><td>0.171</td><td>0.068</td><td>64</td></tr>
<tr><td align="left">fr2_xyz</td><td>0.013</td><td>0.012</td><td>0.045</td><td>0.025</td><td>141</td></tr>
<tr><td align="left">fr3_long_office_household</td><td>0.044</td><td>0.041</td><td>0.104</td><td>0.027</td><td>179</td></tr>
</tbody>
</table>

#### Dynamic (moving people)

<table width="100%">
<thead><tr><th align="left">seq</th><th>ATE RMSE (m)</th><th>ATE mean (m)</th><th>ATE max (m)</th><th>RPE RMSE (m)</th><th>fps</th></tr></thead>
<tbody>
<tr><td align="left">fr3_sitting_static</td><td>0.030</td><td>0.022</td><td>0.080</td><td>0.017</td><td>119</td></tr>
<tr><td align="left">fr3_sitting_xyz</td><td>0.060</td><td>0.053</td><td>0.171</td><td>0.074</td><td>112</td></tr>
<tr><td align="left">fr3_sitting_rpy</td><td>0.070</td><td>0.042</td><td>0.381</td><td>0.096</td><td>111</td></tr>
<tr><td align="left">fr3_sitting_halfsphere</td><td>0.088</td><td>0.066</td><td>0.308</td><td>0.128</td><td>107</td></tr>
<tr><td align="left">fr3_walking_static</td><td>0.266</td><td>0.183</td><td>0.893</td><td>0.494</td><td>87</td></tr>
<tr><td align="left">fr3_walking_halfsphere</td><td>0.436</td><td>0.360</td><td>1.232</td><td>0.521</td><td>91</td></tr>
<tr><td align="left">fr3_walking_xyz</td><td>0.525</td><td>0.374</td><td>1.492</td><td>0.494</td><td>77</td></tr>
<tr><td align="left">fr3_walking_rpy</td><td>0.682</td><td>0.570</td><td>1.337</td><td>0.550</td><td>88</td></tr>
</tbody>
</table>

Low-motion (sitting) sequences hold; high-motion (walking) sequences degrade, as for any feature
SLAM without explicit dynamic-object handling.

### TartanAir (stereo)

<table width="100%">
<thead><tr><th align="left">seq</th><th>ATE RMSE (m)</th><th>ATE mean (m)</th><th>RPE RMSE (m)</th><th>fps</th></tr></thead>
<tbody>
<tr><td align="left">P000</td><td>0.225</td><td>0.185</td><td>1.425</td><td>165</td></tr>
<tr><td align="left">P001</td><td>0.575</td><td>0.551</td><td>1.300</td><td>154</td></tr>
<tr><td align="left">P002</td><td>0.217</td><td>0.133</td><td>1.524</td><td>147</td></tr>
<tr><td align="left">P003</td><td>0.068</td><td>0.061</td><td>1.698</td><td>145</td></tr>
<tr><td align="left">P004</td><td>0.098</td><td>0.075</td><td>1.698</td><td>141</td></tr>
<tr><td align="left">P005</td><td>0.605</td><td>0.436</td><td>1.247</td><td>139</td></tr>
<tr><td align="left">P006</td><td>0.813</td><td>0.637</td><td>1.937</td><td>157</td></tr>
</tbody>
</table>

The KITTI segment metric (t_rel) is undefined here: TartanAir trajectories are too short for the
100-800 m segments.

### TartanGround (stereo)

<table width="100%">
<thead><tr><th align="left">seq</th><th>ATE RMSE (m)</th><th>ATE mean (m)</th><th>RPE RMSE (m)</th><th>t_rel (%)</th><th>fps</th></tr></thead>
<tbody>
<tr><td align="left">P0000</td><td>2.582</td><td>1.934</td><td>1.505</td><td>20.58</td><td>98</td></tr>
<tr><td align="left">P0001</td><td>0.290</td><td>0.238</td><td>1.408</td><td>22.36</td><td>115</td></tr>
<tr><td align="left">P0002</td><td>1.346</td><td>1.134</td><td>1.484</td><td>24.82</td><td>109</td></tr>
<tr><td align="left">P0003</td><td>0.400</td><td>0.254</td><td>1.444</td><td>17.73</td><td>127</td></tr>
<tr><td align="left">P0004</td><td>0.969</td><td>0.821</td><td>1.552</td><td>16.98</td><td>96</td></tr>
</tbody>
</table>

## Quick start

Install the command-line tools (Ubuntu):

```bash
sudo apt install git make docker.io nvidia-container-toolkit gh
curl -LsSf https://astral.sh/uv/install.sh | sh        # uv, the Python tool runner
```

Then, from a fresh clone on a machine with an NVIDIA GPU:

```bash
make build-image-tensorrt10                                        # container (Ubuntu 24.04, TensorRT 10)
uv run python scripts/models/download_onnx_engine_superpoint.py    # prebuilt ONNX models (~90 MB)
uv run python scripts/models/download_onnx_engine_lightglue.py
uv run python scripts/models/download_onnx_engine_eigenplaces.py
make build-engines-tensorrt10                                      # TensorRT engines for this GPU
make build-superslam                                               # compile SuperSLAM

uv run python scripts/datasets/download_kitti.py --out ~/datasets/kitti
make run-superslam-kitti                                           # writes results/kitti/00.txt
make evaluate-superslam-kitti                                      # ATE and RPE
```

`make help` lists every target. Every Python step runs through uv, which builds a throwaway
environment, so nothing is installed globally.

<details>
<summary>conda or pipx, Docker Compose version, bare-metal dependencies</summary>

uv via conda or pipx:

```bash
conda install -c conda-forge git make gh uv
pipx install uv
```

Docker Compose v2 2.30 or newer is required for the `gpus:` key in `compose.yaml`. Check with
`docker compose version`. On an older Compose, edit `compose.yaml`: comment out `gpus: all` and
uncomment the `deploy:` block.

A bare-metal build (no Docker) also needs CUDA 12.9, TensorRT 10.11, cuDNN 9, and the C++ libraries.
One script installs them:

```bash
bash scripts/setup/install_dependencies.sh        # Ubuntu 22.04 or 24.04
```

</details>

## Usage

```bash
# Datasets, downloaded into ~/datasets/<name>
uv run python scripts/datasets/download_kitti.py --out ~/datasets/kitti
uv run python scripts/datasets/download_euroc.py --out ~/datasets/euroc --area machine_hall
uv run python scripts/datasets/download_tum.py   --out ~/datasets/tum --seq fr2_xyz

# Run, writing the trajectory into results/<dataset>/
make run-superslam-kitti SEQUENCE=00
make run-superslam-euroc EUROC=MH_01_easy
make run-superslam-tum   TUM=rgbd_dataset_freiburg2_xyz

# Evaluate: ATE and RPE
make evaluate-superslam-kitti
make evaluate-superslam-euroc
make evaluate-superslam-tum
```

Per-dataset tuning knobs (window size, loop thresholds, and others) live in the dataset YAML under
`examples`, in the `Backend`, `Tracking`, and `loop` sections. Each knob also reads a `SUPERSLAM_*`
environment variable; precedence is the environment variable, then the YAML value, then the default.

The GTSAM optimization core builds and tests without a GPU:

```bash
make test-superslam
```

<details>
<summary>TensorRT 10 or 11, other GPUs, bare-metal build, cleaning</summary>

**TensorRT 10 or 11.** TensorRT 10 is the default and tested path. For 11:

```bash
make build-image-tensorrt11
make build-engines-tensorrt11
```

An engine built for one TensorRT version does not load under the other, so rebuild the engines after
switching. TensorRT 10 or newer is required for LightGlue (opset-18 ONNX).

**Other GPUs.** The code targets Ampere (sm_86), Ada (sm_89), and Blackwell (sm_120). To move to a
different NVIDIA GPU, rebuild the engines (they are GPU-architecture specific):

```bash
make build-engines-tensorrt10 && make build-superslam
```

**Bare-metal (no Docker).**

```bash
bash scripts/setup/install_dependencies.sh
source ~/.bashrc
cmake -S . -B build-tensorrt -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
cmake --build build-tensorrt -j"$(nproc)"
bash scripts/rebuild_engines.sh
./examples/kitti examples/stereo/KITTI00-02.yaml ~/datasets/kitti/dataset/sequences/00 --no-viewer
```

**Cleaning.** `make clean-engines`, `clean-weights`, `clean-build`, `clean-results`, `clean-images`,
or `clean-all`. Every target is safe to re-run.

</details>

<details>
<summary>The .onnx.data sidecar, and publishing models</summary>

**`.onnx.data`.** Some models ship as a pair: `model.onnx` holds the graph, and `model.onnx.data`
holds the large weight tensors. LightGlue and EigenPlaces are stored this way, so both files must
sit in `weights/` together. The download scripts fetch the sidecar automatically. The SuperPoint
ONNX has inline weights and no `.data` file.

**Publishing the prebuilt ONNX (maintainers).** The download scripts pull from a GitHub Release:

```bash
gh release create weights-v1 weights/superpoint_dense_dynamic_batch.onnx \
  weights/lightglue_superpoint.onnx weights/lightglue_superpoint.onnx.data \
  weights/eigenplaces_resnet18_512.onnx weights/eigenplaces_resnet18_512.onnx.data
```

To regenerate the ONNX from source, see `scripts/models/download_weights_*.py` and the converters in
`utils`.

</details>

## Citation

```bibtex
@software{wagh_superslam,
  author = {Wagh, Aditya},
  title  = {SuperSLAM: Accurate Real-Time SLAM with Deep Learned Features},
  year   = {2026},
  url    = {https://github.com/adityamwagh/SuperSLAM}
}
```

## License

Licensed under the [LGPL](LICENSE). For questions or issues, open an issue on GitHub.

## Acknowledgements

Code and ideas come from these projects:

- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [LightGlue](https://github.com/cvg/LightGlue)
- [EigenPlaces](https://github.com/gmberton/EigenPlaces)
- [GTSAM](https://github.com/borglab/gtsam)
- [TensorRT](https://github.com/NVIDIA/TensorRT)
- [AirVO](https://github.com/xukuanHIT/AirVO)
- [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2)
- [SuperPoint-SuperGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT)

## Star History

<a href="https://www.star-history.com/#adityamwagh/SuperSLAM&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=adityamwagh/SuperSLAM&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=adityamwagh/SuperSLAM&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=adityamwagh/SuperSLAM&type=Date" />
 </picture>
</a>
