# TensorRT Setup on Jetson Orin

[简体中文](README.tensorrt.zh-CN.md)

`unitree-controller --inference-backend tensorrt` loads ONNX Runtime's
`TensorrtExecutionProvider` first and `CUDAExecutionProvider` second. The
controller checks at startup that the TensorRT provider initializes.
`trtexec` verifies that the system TensorRT installation can run the model.
The controller also needs a GPU-enabled ONNX Runtime build that can load the
provider.

Run the commands below from the `unitree-deploy` repository root.

## 1. Check JetPack and runtime libraries

```bash
uname -m
head -n 1 /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core
ldconfig -p | grep -E 'lib(cuda|cudart|cudnn|nvinfer|nvonnxparser)\.so'
```

`R35` with `REVISION: 3.1` corresponds to JetPack 5.1.1, which ships with
CUDA 11.4, cuDNN 8.6, and TensorRT 8.5.2. The CUDA `.so` filename shows its
major version. Run `/usr/local/cuda/bin/nvcc --version` to check the toolchain
before building. [NVIDIA JetPack 5.1.1 release information](https://developer.nvidia.com/embedded/jetpack-sdk-511)

Run the system TensorRT tool to check the model offline. `PASSED` means
TensorRT built and ran the model on this device. If `trtexec` is missing,
install `libnvinfer-bin` from the current JetPack package repository at the same
version as `libnvinfer8`. [NVIDIA JetPack 5.1.1 package list](https://docs.nvidia.com/jetson/jetpack/5.1.1/install-jetpack/index.html)

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=ckpt/g1/vanilla_ppo_flat/policy_6000_w_symm_w_muon.onnx
```

## 2. Get a GPU wheel

If you already have a prebuilt wheel, check that it targets Jetson Orin
(`aarch64`, `sm_87`), includes the TensorRT provider, and matches the device's
CUDA, cuDNN, and TensorRT versions. The project supports Python 3.11 (`cp311`)
and 3.12 (`cp312`). For a compatible wheel with ONNX Runtime 1.24.4 or later,
use the original checkpoint and proceed to [Install and verify](#4-install-and-verify).
[NVIDIA Jetson wheel guide](https://github.com/NVIDIA/skills/blob/main/skills/jetson-package/SKILL.md#gpu-python-wheels-on-jetson) · [ONNX Runtime compatibility table](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)

### Build from source on JetPack 5.1.1

This setup uses **Python 3.11, ONNX Runtime 1.16.3, and NumPy 1.26.4**. It
builds the CUDA and TensorRT providers against the JetPack 5 system libraries.
Use ORT 1.16.3 in the dedicated deployment environment.
[AWS JetPack 5 build script](https://raw.githubusercontent.com/awslabs/DefectDetectionApplication/main/src/backend/edge_ml1_p_camera_management/install_onnxruntime_gpu.sh)

Install GCC/G++ 10 and development packages that match the system libraries on
the Orin. Skip the installation command if the packages are already installed.
[NVIDIA JetPack 5.1.1 package list](https://docs.nvidia.com/jetson/jetpack/5.1.1/install-jetpack/index.html)

```bash
sudo apt install build-essential gcc-10 g++-10 git curl cuda-toolkit-11-4 \
  libcudnn8-dev libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev
/usr/local/cuda/bin/nvcc --version
```

Build in a separate environment. The last command prints the absolute path to
the wheel for the installation step.

```bash
ort_build_dir="$(mktemp -d "$HOME/unitree-ort-jp5.XXXXXX")"
(
  set -euo pipefail
  uv venv --seed --python 3.11 "$ort_build_dir/venv"
  ort_build_python="$ort_build_dir/venv/bin/python"
  uv pip install --python "$ort_build_python" \
    'cmake>=3.26,<4' 'numpy==1.26.4' packaging setuptools wheel
  export PATH="$ort_build_dir/venv/bin:$PATH"

  git clone --depth 1 --branch v1.16.3 \
    https://github.com/microsoft/onnxruntime.git "$ort_build_dir/onnxruntime"
  eigen_commit=e7248b26a1ed53fa030c5c459f7ea095dfd276ac
  mkdir "$ort_build_dir/eigen"
  curl -fsSL "https://codeload.github.com/eigen-mirror/eigen/tar.gz/$eigen_commit" |
    tar -xz --strip-components=1 -C "$ort_build_dir/eigen"

  cd "$ort_build_dir/onnxruntime"
  CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 "$ort_build_python" tools/ci_build/build.py \
    --build_dir build/Linux --config Release --update --build --build_wheel --skip_tests \
    --parallel 2 --nvcc_threads 1 \
    --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
    --use_tensorrt --tensorrt_home /usr \
    --use_preinstalled_eigen --eigen_path "$ort_build_dir/eigen" \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 \
    CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 onnxruntime_BUILD_UNIT_TESTS=OFF

  find "$PWD/build/Linux/Release/dist" -maxdepth 1 \
    -name 'onnxruntime_gpu-1.16.3-cp311-*.whl' -print
)
```

## 3. Prepare the checkpoint for ORT 1.16.3

The repository's `vanilla_ppo_flat` model uses ONNX IR 10 and opset 18.
ORT 1.16.3 supports opset 18 and can load models up to IR 9. The conversion
below applies to this model's `Gemm` and `Elu` graph. It writes a new checkpoint
directory and preserves the weights, operators, opset, and policy settings.
Check compatibility separately for other checkpoints.
[ONNX Runtime IR and opset compatibility](https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support)

```bash
uv run --no-project --python 3.11 --with 'onnx==1.17.0' python - <<'PY'
from pathlib import Path
from shutil import copyfile

import onnx

source = Path("ckpt/g1/vanilla_ppo_flat")
target = source.with_name("vanilla_ppo_flat_jp5")
model_name = "policy_6000_w_symm_w_muon.onnx"
model = onnx.load(source / model_name)

assert model.ir_version == 10
assert [(item.domain, item.version) for item in model.opset_import] == [("", 18)]
assert {(node.domain, node.op_type) for node in model.graph.node} == {
    ("", "Gemm"), ("", "Elu"),
}
assert not model.functions and not model.training_info and not model.graph.sparse_initializer
assert all(not node.overload for node in model.graph.node)

graph = model.graph
for entry in (graph, *graph.node, *graph.input, *graph.output, *graph.value_info, *graph.initializer):
    assert all(item.key == "namespace" or item.key.startswith("pkg.torch.") for item in entry.metadata_props)
    entry.ClearField("metadata_props")
model.ir_version = 9
onnx.checker.check_model(model)

target.mkdir()
onnx.save(model, target / model_name)
copyfile(source / "policy.yaml", target / "policy.yaml")
print(target)
PY
```

After installing the compatible wheel, point the controller at
`ckpt/g1/vanilla_ppo_flat_jp5`.

## 4. Install and verify

Install the GPU wheel in a separate runtime environment. The project lockfile
declares the CPU `onnxruntime` package, so skip that package when syncing the
project dependencies. Set `wheel_path` to the wheel's absolute path. For a
`cp312` wheel, use `--python 3.12` in the sync command.
[uv project environment configuration](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path)

```bash
gpu_env="$HOME/.venvs/unitree-deploy-tensorrt"
wheel_path=/path/to/onnxruntime_gpu.whl
UV_PROJECT_ENVIRONMENT="$gpu_env" uv sync --python 3.11 --no-install-package onnxruntime

# ORT 1.16.3 built for JetPack 5.1.1
uv pip install --python "$gpu_env/bin/python" "$wheel_path" 'numpy==1.26.4'

# For a newer wheel that meets the project requirements, use the command below.
# uv pip install --python "$gpu_env/bin/python" "$wheel_path"
```

Install `unitree_sdk2_python` into the same environment as described in the
[project setup instructions](README.md#-setup).

Load the actual checkpoint and run one inference using the repository's
checkpoint loader. This check runs offline. For a newer ONNX Runtime build,
set `ckpt_dir` to `ckpt/g1/vanilla_ppo_flat`.

```bash
ckpt_dir=ckpt/g1/vanilla_ppo_flat_jp5
"$gpu_env/bin/python" - "$ckpt_dir/policy.yaml" <<'PY'
import sys
from pathlib import Path

import numpy as np
from unitree_deploy.runtime.controllers.checkpoints import build_ckpt_profile

profile = build_ckpt_profile(
    "default",
    Path(sys.argv[1]),
    providers=("TensorrtExecutionProvider", "CUDAExecutionProvider"),
)
session = profile.policy.session
actions = session.run(None, {"obs": np.zeros((1, 480), dtype=np.float32)})[0]
assert actions.shape == (1, 29) and np.isfinite(actions).all()
print(session.get_providers())
print(actions.shape)
PY
```

The output should include `TensorrtExecutionProvider` and `(1, 29)`. Then
start the controller with your DDS network interface. The example uses `eth0`.

```bash
"$gpu_env/bin/unitree-controller" \
  --mode real --net eth0 --ckpt "$ckpt_dir" --inference-backend tensorrt
```

Call the controller directly from this environment, or set
`UV_PROJECT_ENVIRONMENT="$gpu_env"` and use `uv run --no-sync`. A regular
`uv sync` or `uv run` targeting the GPU environment will sync the CPU package
back into it. Before using `tensorrt-fp16` on hardware, compare its actions with
the CPU backend on the same observations.
[uv sync documentation](https://docs.astral.sh/uv/concepts/projects/sync/)

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Available providers: AzureExecutionProvider, CPUExecutionProvider` | Print `onnxruntime.__file__` and `onnxruntime.get_available_providers()` with the same Python that runs the controller. Check that it loads the GPU wheel. |
| `TensorrtExecutionProvider did not initialize` | Check the preceding ORT logs. If a library is missing, run `ldd` on `onnxruntime/capi/libonnxruntime_providers_tensorrt.so` from the installed wheel and check the missing library's version against JetPack and the wheel. |
| `Unsupported model IR version: 10, max supported IR version: 9` | ORT 1.16.3 loaded the original model. Point `--ckpt` at the generated `vanilla_ppo_flat_jp5` directory. |
| `is not a supported wheel on this platform` | Compare `uname -m` and the target environment's `python --version` with the wheel's architecture and CPython tags. |

To switch back to CPU temporarily, run the controller from the original
project environment with the original checkpoint and `--inference-backend cpu`.
