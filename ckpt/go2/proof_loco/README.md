# Go2 Proof-Loco 感知策略

本目录包含 `proof-loco-website/mujoco_wasm` WebDemo 的 actor 和深度编码器 ONNX，以及对应的观测、循环状态和关节目标计算。策略运行于 50 Hz。Actor 读入 450 维本体观测和 64 维深度特征，编码器读入两帧相隔约 100 ms 的 `23 × 44` 深度图。

## MuJoCo 仿真

在仓库根目录分别运行两个命令。

```bash
unitree-sim-bridge --robot go2 --net lo --terrain rough \
  --sensor ckpt/go2/proof_loco/mjc_depth_camera_sensor.yaml \
  --sensor ckpt/go2/height_scan/mjc_height_scan_sensor.yaml
unitree-controller --mode sim --net lo --ckpt ckpt/go2/proof_loco
```

在仿真中按空格开始，按 Enter 进入默认姿态，再按反斜杠键启动策略。相机位姿和仿真采样频率在 `mjc_depth_camera_sensor.yaml` 中配置。上述命令使用 Unitree Deploy 的 Go2 模型和 rough 场景。

MuJoCo viewer 中，按 `G` 可切换深度相机视锥线框。在线框对应的 `mjc_depth_camera_sensor.yaml` 和采样点对应的 `mjc_height_scan_sensor.yaml` 中，可分别设置 `params.debug_rgba`（红、绿、蓝、不透明度，每项取 0 到 1）。左下角显示实时深度图（近处为黑，远处为白）。加载 [height scan 传感器](../height_scan/README.md) 时也会单独发布高度网格。

## 真机深度源

RealSense 采集和 controller 在同一台机器运行。按实际相机填写 `realsense.yaml` 的序列号和流配置，安装 `pyrealsense2`。使用该策略前，应在机器人上核对相机安装位姿及深度坐标系，并确认启用的是原始深度流。

```bash
unitree-realsense --visualizer-yaml ckpt/go2/proof_loco/realsense.yaml
unitree-controller --mode real --net eth0 --ckpt ckpt/go2/proof_loco
```

将 `eth0` 换成连接 Go2 的网卡。采集进程的 `camera_name` 必须和 `policy.yaml` 中的 `source_params.camera_name` 一致。深度源启动后可通过正常状态机进入策略。相机断流或帧超过 `max_age` 时，controller 回到阻尼状态。

## 更换观测或数据源

- `policy.yaml` 的 `observations` 定义本体观测。独立的 `depth_observation` 定义深度观测类型、相机时间上的历史间隔、视场、裁剪、滤波和归一化范围。修改观测布局后，需要让它与相应的 ONNX 输入一致。
- `depth_observation.params.source_class` 接受 `模块:类`，`source_params` 作为构造参数。自定义来源可以放在 checkpoint 中，例如 ROS 或厂商 SDK 的适配类。实现非阻塞的 `read()` 和 `close()`。每次 `read()` 返回 `RealSenseFrame` 或 `None`，使用 `depth` 传递二维米制光轴深度，使用 `timestamp_sec` 传递采集时间，同一帧保持相同时间戳。`depth_intrinsics` 可提供 `(fx, fy, cx, cy)`，省略时按目标视场和源图尺寸估算。若提供 `published_at`，应使用 controller 所在机器的单调时钟。
- 自定义仿真深度相机可继承 `unitree_deploy.runtime.simulation.sensors.depth_camera.MujocoDepthSensor`，并在传感器 YAML 中将 `sensor_class` 指向子类，例如 `custom_sensor:CustomDepthSensor`。`params` 传给子类的构造方法。实现新类型时可继承 `unitree_deploy.runtime.simulation.sensors.base_sensor.MujocoSensor`，复用采样调度并实现 `configure_model(spec)`、`_capture_and_publish(model, data, captured_at=...)` 和 `close()`。发布成功时返回 `True`，需要在下一轮重试时返回 `False`。
