# Jetson Orin TensorRT 配置

[English](README.tensorrt.md)

`unitree-controller --inference-backend tensorrt` 通过 ONNX Runtime 加载
`TensorrtExecutionProvider`，并把 `CUDAExecutionProvider` 作为次选。
控制器启动时会检查 TensorRT provider 是否成功初始化。
`trtexec` 可以单独验证系统 TensorRT 能否执行模型。控制器还需要能加载该 provider 的 GPU 版 ONNX Runtime。

以下命令在 `unitree-deploy` 仓库根目录运行。

## 1. 确认 JetPack 和运行库

```bash
uname -m
head -n 1 /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core
ldconfig -p | grep -E 'lib(cuda|cudart|cudnn|nvinfer|nvonnxparser)\.so'
```

`R35`、`REVISION: 3.1` 对应 JetPack 5.1.1，原配 CUDA 11.4、
cuDNN 8.6 和 TensorRT 8.5.2。CUDA 的 `.so` 文件名只能看出主版本，编译前可运行
`/usr/local/cuda/bin/nvcc --version` 确认工具链版本。 [NVIDIA JetPack 5.1.1 版本说明](https://developer.nvidia.com/embedded/jetpack-sdk-511)

可先用系统的 TensorRT 做一次离线验证。输出 `PASSED` 表示模型已在这台机器上构建并执行。
如果缺少 `trtexec`，可从当前 JetPack 软件源安装与 `libnvinfer8` 同版本的 `libnvinfer-bin`。
[NVIDIA JetPack 5.1.1 软件包列表](https://docs.nvidia.com/jetson/jetpack/5.1.1/install-jetpack/index.html)

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=ckpt/g1/vanilla_ppo_flat/policy_6000_w_symm_w_muon.onnx
```

## 2. 获取 GPU wheel

已有预编译包时，确认它面向 Jetson Orin（`aarch64`、`sm_87`），包含 TensorRT
provider，并与目标机的 CUDA、cuDNN 和 TensorRT 版本匹配。当前项目支持 Python 3.11
和 3.12，分别对应 `cp311` 和 `cp312`。使用 ONNX Runtime 1.24.4 或更高版本时，可直接用原
checkpoint，跳到[安装和验证](#4-安装和验证)。 [Jetson wheel 指南](https://github.com/NVIDIA/skills/blob/main/skills/jetson-package/SKILL.md#gpu-python-wheels-on-jetson) · [ONNX Runtime 兼容表](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)

### JetPack 5.1.1 从源码构建

这条路径使用 **Python 3.11、ONNX Runtime 1.16.3、NumPy 1.26.4**。CUDA 和 TensorRT
provider 一起构建，链接 JetPack 5 的系统库。ORT 1.16.3 是此部署环境单独使用的版本。
[AWS JetPack 5 构建脚本](https://raw.githubusercontent.com/awslabs/DefectDetectionApplication/main/src/backend/edge_ml1_p_camera_management/install_onnxruntime_gpu.sh)

在 Orin 上准备 GCC/G++ 10 和与系统运行库匹配的开发包。如果设备已经装齐，可跳过安装。
[NVIDIA JetPack 5.1.1 软件包列表](https://docs.nvidia.com/jetson/jetpack/5.1.1/install-jetpack/index.html)

```bash
sudo apt install build-essential gcc-10 g++-10 git curl cuda-toolkit-11-4 \
  libcudnn8-dev libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev
/usr/local/cuda/bin/nvcc --version
```

下面的命令使用独立的构建环境。编译结束后会打印 wheel 的绝对路径，安装时使用该路径。

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

## 3. 为 ORT 1.16.3 生成兼容 checkpoint

仓库里的 `vanilla_ppo_flat` 模型使用 ONNX IR 10 和 opset 18。ORT 1.16.3
支持 opset 18，最高支持 IR 9。下面的命令只处理这份由 `Gemm` 和 `Elu`
组成的模型，生成一个新目录，保留权重、算子、opset 和策略配置。其他 checkpoint
需要独立确认兼容性。 [ONNX Runtime IR 和 opset 兼容表](https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support)

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

装好兼容 wheel 后，用 `ckpt/g1/vanilla_ppo_flat_jp5` 作为控制器的 checkpoint。

## 4. 安装和验证

用独立的运行环境安装 GPU wheel。项目的 `uv.lock` 声明了 CPU 版 `onnxruntime`，
因此在同步项目依赖时显式跳过它。将 `wheel_path` 换成 wheel 的绝对路径。
如用 `cp312` wheel，将 `--python 3.11` 改为 `--python 3.12`。
[uv 项目环境配置](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path)

```bash
gpu_env="$HOME/.venvs/unitree-deploy-tensorrt"
wheel_path=/path/to/onnxruntime_gpu.whl
UV_PROJECT_ENVIRONMENT="$gpu_env" uv sync --python 3.11 --no-install-package onnxruntime

# JetPack 5.1.1 构建的 ORT 1.16.3
uv pip install --python "$gpu_env/bin/python" "$wheel_path" 'numpy==1.26.4'

# 对于满足当前项目要求的新版 wheel，上面的安装命令改用这一行：
# uv pip install --python "$gpu_env/bin/python" "$wheel_path"
```

仍需按[项目安装说明](README.zh-CN.md#-安装)把 `unitree_sdk2_python` 安装到这个环境。

先用实际 checkpoint 验证 TensorRT session 和一次推理。下面的命令使用仓库的
checkpoint 加载入口，不启动控制器。使用新版 ORT 时，把 `ckpt_dir` 设成
`ckpt/g1/vanilla_ppo_flat`。

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

输出应包含 `TensorrtExecutionProvider` 和 `(1, 29)`。然后用原来的 DDS
网卡启动控制器，示例中用 `eth0`：

```bash
"$gpu_env/bin/unitree-controller" \
  --mode real --net eth0 --ckpt "$ckpt_dir" --inference-backend tensorrt
```

直接调用这个环境的控制器，或设置 `UV_PROJECT_ENVIRONMENT="$gpu_env"` 后用
`uv run --no-sync`。对 GPU 环境运行普通 `uv sync` 或 `uv run` 会重新同步 CPU 包。
如果要启用 `tensorrt-fp16`，先用同一组观测对比 CPU 和 FP16 的动作输出。
[uv 同步说明](https://docs.astral.sh/uv/concepts/projects/sync/)

## 常见报错

| 现象 | 检查方法 |
| --- | --- |
| `Available providers: AzureExecutionProvider, CPUExecutionProvider` | 用运行控制器的同一个 Python 打印 `onnxruntime.__file__` 和 `onnxruntime.get_available_providers()`，确认它加载的是 GPU wheel。 |
| `TensorrtExecutionProvider did not initialize` | 查看前面的 ORT 日志。若有缺失库，对 wheel 中的 `onnxruntime/capi/libonnxruntime_providers_tensorrt.so` 运行 `ldd`，按缺失库的版本核对 JetPack 和 wheel。 |
| `Unsupported model IR version: 10, max supported IR version: 9` | ORT 1.16.3 加载到了原模型。将 `--ckpt` 指向生成的 `vanilla_ppo_flat_jp5`。 |
| `is not a supported wheel on this platform` | 核对 `uname -m`、环境内的 `python --version` 和 wheel 文件名的架构及 CPython 标签。 |

需要临时用 CPU 时，从原有项目环境启动控制器，使用原 checkpoint 和
`--inference-backend cpu`。
