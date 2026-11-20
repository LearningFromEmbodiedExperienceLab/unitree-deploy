# Go2 Height Scan

MuJoCo 在机器人脚下和前方按网格向下发射射线，忽略机器人和其他动态物体。默认网格为 `[15, 11]`，以底座为原点，只随底座 yaw 旋转。X 向前，Y 向左，观测按 X 外层、Y 内层展开。每格为 `地面 Z - 底座 Z`，单位米。网格范围、射线高度和频率在 `mjc_height_scan_sensor.yaml` 配置。

## 启动仿真

```bash
unitree-sim-bridge --robot go2 --net lo --terrain rough \
  --sensor ckpt/go2/height_scan/mjc_height_scan_sensor.yaml
```

MuJoCo viewer 中的点表示扫描射线命中地形的位置，颜色可通过 `params.debug_rgba` 配置。同时需要深度图时，在同一个命令中再加上 `--sensor ckpt/go2/proof_loco/mjc_depth_camera_sensor.yaml`。两个传感器各自发布帧。可用下面的命令查看当前扫描。

```bash
python -c 'from unitree_deploy.runtime.transport.height_scan import HeightScanReader; print(HeightScanReader("go2_height_scan").read())'
```

## 接入策略

把 `height_scan_observation.py` 放到策略的 `policy.yaml` 旁边。在 YAML 的 `observation_types` 和 `observations` 中加入以下配置，按训练时的顺序放置观测项。仿真扫描可在传感器 YAML 中调整 `params`，也可以继承 `unitree_deploy.runtime.simulation.sensors.height_scan.MujocoHeightScanSensor`，再把 `sensor_class` 指向子类。

```yaml
observation_types:
  height_scan: "height_scan_observation:HeightScanObservation"
observations:
  - type: height_scan
    history_len: 1
    clip: [-1.0, 1.0]
    params:
      shape: [15, 11]
      stream_name: go2_height_scan
      max_age: 0.5
      fill_miss: -1.0
```

对于拼接输入，每个历史帧增加 165 个值。历史按策略步堆叠，未命中使用 `fill_miss`，之后依次应用 `params.offset`、`scale` 和 `clip`。使用需要该输入的 ONNX 权重和训练时的形状、单位及排列。仓库里的 Proof-Loco WebDemo 模型使用深度图。

## 真机数据源

真机高程图重建待实现。LiDAR、深度融合等重建方法都可通过下面的网格发布接口接入。

同机的 LiDAR 或深度建图进程可以从 `unitree_deploy.runtime.transport.height_scan` 导入 `HeightScanPublisher`，然后调用 `HeightScanPublisher("go2_height_scan")` 创建数据流，通过 `publish(grid, captured_at=stamp)` 发布对齐后的网格。网格使用与仿真相同的 `[X, Y]`、坐标系和单位，未命中填 `NaN`。时间戳用 controller 所在主机的单调时钟。进程退出时调用 `close()`。

已有其他传输方式时，可以在 checkpoint 里继承 `HeightScanObservation`，构造时传入 `source`。其非阻塞 `read()` 返回 `(grid, captured_at)` 或 `None`。通过 `observation_types` 注册自己的类。扫描缺失、超时或形状不符时，控制器退回阻尼。
