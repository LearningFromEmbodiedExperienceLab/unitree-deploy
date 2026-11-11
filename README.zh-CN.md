# Unitree Deploy

[English](README.md)

面向 Unitree 机器人的轻量级部署代码库。

本项目把 ONNX 策略、观测构造、DDS 通信、MuJoCo 模型和控制器状态机整理成一套可复用流程，适合先在仿真中验证策略，再用相同控制路径部署到 Unitree 真机。

## ✨ 特性

- 同一套 controller runtime 支持 `sim` 和 `real` 模式。
- 通过可配置的 `yaml` 文件加载 ONNX policy。
- 支持多 policy 切换。
- 提供模板生成器，方便自定义部署需求，例如自定义 observation。

## 📦 项目结构

```text
src/unitree_deploy/
├── cli/              # 命令行入口
├── runtime/
│   ├── controllers/  # policy/Z1 控制主循环与 checkpoint 加载
│   ├── state_machine/
│   │   ├── machine.py   # 状态生命周期与切换调度
│   │   ├── config.py    # YAML、事件、目标姿态与增益解析
│   │   ├── registry.py  # 状态构建与切换目标校验
│   │   ├── filters.py   # 共用的目标平滑
│   │   └── states/      # 姿态、policy 与手臂过渡状态实现
│   ├── simulation/   # MuJoCo bridge 与键盘输入
│   ├── transport/    # Unitree DDS 适配与遥控输入解码
│   ├── trajectory/   # 轨迹录制与回放应用
│   └── timing.py     # 共用的循环计时
├── policy/
│   ├── base_policy.py  # 通用 ONNX 封装与 checkpoint 加载
│   └── motion_tracking/  # 动作跟踪策略与参考数据
├── obs/              # 共用观测项与策略专用观测项
├── visualization/    # 机器人状态可视化
├── utils/            # 通用工具
└── robot_model/      # 内置 MuJoCo 机器人与地形资源

ckpt/
├── g1/               # G1 示例策略与 multi-ckpt 配置
└── go2/              # Go2 示例策略
```

## 🚀 安装

```bash
uv sync
source .venv/bin/activate
```

如果需要 mjviser 浏览器可视化依赖，用于实时机器人状态可视化：

```bash
uv sync --extra viewer
```

`unitree_sdk2_python` 需要单独安装到同一个 Python 环境中。

## 🕹️ 快速开始

启动 MuJoCo 到 DDS 的仿真桥：

```bash
unitree-sim-bridge --robot g1
```

MuJoCo Viewer 窗口内显示移动、policy 和仿真控制的按键提示。
按 `?`（`Shift` + `/`）可显示或隐藏提示。
默认屏蔽原生键盘快捷键。在 Viewer 中按 `F12` 可恢复原生快捷键，同时禁用全局 sim2sim 控制键，避免终端输入触发仿真命令；再按一次恢复 sim2sim 控制并屏蔽原生快捷键。
按 `T` 可切换相机是否跟随机器人。相机跟随功能要求
机器人 `visualizer.yaml` 中的 `viewer.camera.track_body` 已配置。

velocity command policy 运行时显示 mjlab 风格的速度箭头：蓝色/青色表示目标/实际线速度，
绿色/浅绿色表示目标/实际 yaw 角速度。默认按 `G` 统一切换速度箭头和 motion ghost；
显示状态在 policy 切换后保留，并同步应用于 MP4 导出。已有 `motion_ghost.toggle_key` 配置会作为两者的共同按键。

速度箭头的位置和大小可在各机器人 `visualizer.yaml` 的现有 `viewer` 配置下调整：

```yaml
viewer:
  policy_debug_viz:
    velocity:
      offset: [0.0, 0.0, 0.3]  # base-link 坐标系偏移，单位米；Go2 使用此高度
      scale: 0.5              # 目标和实际速度共用的箭头长度比例
      width: 0.015            # 箭头宽度，单位米
```

未配置时默认 `offset: [0.0, 0.0, 0.1]`、`scale: 0.5`、`width: 0.015`。
offset 不随 scale 缩放；修改后重启 bridge 生效，视频导出使用相同配置。

按 `=` 加速、`-` 减速，依次切换 `0.5x`、`1x`、`2x`、`3x`、`4x`、`5x`、`6x`、`7x`、`8x` 物理仿真速度，默认 `1x`，达到边界后不再变化。
物理时间步长保持不变，当前倍率会显示在 bridge 终端状态中；暂停期间也可选择倍率，按 `space` 恢复后生效。
独立 policy controller 仍使用配置的运行频率，因此这些按键会改变物理时间与 controller 更新的比例。
`Space` 会同时暂停物理、policy 更新和 motion reference，恢复时从原来的帧继续。
更新后请同时重启 bridge 和 controller，以启用同步暂停。

启动仿真 controller：

```bash
# ckpt参数如果输入为路径，默认加载其中的policy.yaml文件
unitree-controller --mode sim --ckpt ckpt/g1/vanilla_ppo_flat

# 也可以直接指定 policy YAML 文件。
unitree-controller --mode sim --ckpt ckpt/g1/vanilla_ppo_flat/policy_torso_base.yaml
```

使用 multi-policy 配置：

```bash
unitree-controller --mode sim --multi-ckpt ckpt/g1/multi_ckpt.yaml
```

启动 viser 可视化：

```bash
unitree-visualizer --mode sim --robot g1
```

### 同机共享 RealSense

采集进程通过 Linux 共享内存向 Viser 和 policy 提供最新一帧。

```bash
uv pip install pyrealsense2
unitree-realsense --robot g1
unitree-visualizer --mode real --robot g1 --net eth0 --camera
```

两端使用 `visualizer.yaml` 中相同的相机名。Policy 可通过
`unitree_deploy.runtime.transport.realsense.RealSenseReader` 读取。

回放已保存的轨迹，默认使用 Viser 前端：

```bash
unitree-trajectory-replay export/runs/g1_walk/20260629-153012/trajectory.npz
```

也可以切到原生 MuJoCo Viewer：

```bash
unitree-trajectory-replay \
  --viewer mujoco \
  export/runs/g1_walk/20260629-153012/trajectory.npz
```

MuJoCo Viewer 前端支持 `space` 播放/暂停，左右方向键逐帧步进，`r` 回到开头，`+/-` 调速，`l` 切换循环播放。

浏览器里的 policy 展示已经移到独立的 `web_policy` 项目：

```bash
cd /home/syw/.gitrepos/web_policy
web-policy --robot g1 --ckpt /home/syw/.gitrepos/unitree-deploy/ckpt/g1/vanilla_ppo_flat
```

打开 `http://127.0.0.1:8000` 查看。

真机运行时需要显式指定 DDS 网卡：

```bash
unitree-controller --mode real --net <interface> --ckpt ckpt/g1/vanilla_ppo_flat
```

### Z1 阻抗控制器

Z1 控制器复用 locomotion controller 的 `A`/`Start`/`X` 状态机操作：

- `A`：从机械臂当前实测姿态移动到 `default_qpos`。
- `Start`：默认姿态移动完成后进入阻抗模式。
- `X`：回到 damping。

real 模式会自动启动固定配置的 `z1_udp_service`，并连接
`rt/z1/lowstate` 和 `rt/z1/lowcmd`：

```bash
unitree-z1-controller \
  --mode real \
  --net <interface> \
  --ckpt ckpt/b2z1/b2z1_manip
```

sim 模式不会启动 UDP bridge。B2Z1 sim2sim 先启动 MuJoCo bridge：

```bash
unitree-sim-bridge --robot b2z1 --net lo
```

如果仿真中的 B2 底盘也需要主动控制，再启动 locomotion controller：

```bash
unitree-controller \
  --mode sim \
  --robot b2z1 \
  --net lo \
  --ckpt ckpt/b2z1/b2z1_loco
```

然后启动 Z1 controller：

```bash
unitree-z1-controller \
  --mode sim \
  --net lo \
  --ckpt ckpt/b2z1/b2z1_manip
```

控制器启动后保持 damping，不会自动移动机械臂。bridge 将
`rt/lowcmd`/`rt/lowstate` 映射到 12 个 B2 actuator，将
`rt/z1/lowcmd`/`rt/z1/lowstate` 映射到 7 个 Z1 actuator。两个 controller
共用同一份仿真遥控器，因此同时运行时，`A`、`Start` 和 `X` 会作用于两个状态机。

## Pico 人形遥操 / MimicLite

在两个终端启动 G1 仿真和控制器：

```bash
unitree-sim-bridge --robot g1 --net lo
unitree-controller --mode sim --robot g1 --net lo --ckpt ckpt/g1/mimic_lite
```

按 `Enter` 移到默认姿态。参考源就绪后按 `\`，先做手臂过渡，
完成后按 `b` 确认进入参考首帧保持。看到就绪提示后再按 `b` 播放。`g` 切换参考 ghost，`x` 回到 damping。
空格暂停仿真和文件动作，`-` 和 `=` 同步调整播放倍速。

在第三个终端选择参考源：

```bash
# any4hdmi G1 动作（保留数据集的 manifest.json）
unitree-motion-source --config ckpt/g1/mimic_lite/policy.yaml --motion-path /path/to/clip.npz
# 或 Pico（启动 XRoboToolkit PC Service 并开启全身追踪）
unitree-motion-source --config ckpt/g1/mimic_lite/policy.yaml --human-height 1.75 --retarget gmr
```

动作源默认启动 G1 参考姿态的 mjviser 预览，并打印浏览器地址。
在 `unitree-deploy` 仓库根目录运行，将 `viewer` 可选依赖及 Any4HDMI 共享查看器
安装到同一个环境中：

```bash
uv sync --extra viewer
uv pip install --no-deps -e /mnt/datasets/any4hdmi
```

将 `/mnt/datasets/any4hdmi` 替换为你的 Any4HDMI 仓库路径。`-e` 使用可编辑安装，
修改源码后无需重新安装，只需重启 `unitree-motion-source`。
`--no-deps` 避免安装 Any4HDMI 的训练依赖。

预览独立于 DDS 发布；无界面发布可加 `--no-visualizer`。
仿真时，预览通过 `rt/policy_debug_viz` 显示控制器缓冲并对齐后的参考姿态，
与 sim-bridge ghost 一致；控制器参考未激活或超过 0.5 秒未更新时，恢复显示动作源姿态。

也可直接给控制器传 `--motion-path`。播完或 DDS 参考中断时，GMT 继续跟踪最后有效帧。
启动新的发布进程后，控制器会自动准备手臂，完成后按 `B` 确认保持新首帧，就绪后再按 `B` 播放。
Pico：站直完成标定，用 Pico `X` 切换实时输入；需安装 `xrobotoolkit_sdk` 和选用的后端
（支持 XRobot 的 GMR、FRAX 或 RoboKit）。

真机将控制器改为 `--mode real`，两个进程用 `--net` 指定各自的 DDS 网卡。
宇树手柄用 `A` → `Start` → `B`（确认）→ `B`（播放），`X` 回到 damping。

## 🧩 部署目录

一个 policy deployment 目录通常包含：

```text
ckpt/<robot>/<policy>/
├── policy.yaml             # policy、observation、关节顺序和增益配置
├── policy.onnx             # 导出的 ONNX，也可以在 YAML 中改成其他相对路径
├── custom_observations.py  # [可选] 自定义 observation
└── custom_policy.py        # [可选] 自定义推理或 action 后处理
```

交互式生成模板：

```bash
unitree-plugin-template
```

也可以脚本化生成：

```bash
unitree-plugin-template ckpt --robot g1 --name my_policy
```

`policy.yaml` 中最关键的字段：

```yaml
policy_path: "policy.onnx"
imu_source: pelvis  # G1 可选 pelvis 或 torso
obs_joint_order: [...]
action_joint_order: [...]
sdk_joint_order: [...]
```

`obs_joint_order`、`action_joint_order` 和 `sdk_joint_order` 会按关节名自动推导重排索引，不需要手写 index。

`observation_types` 中的内置类可以直接写类名，自定义类使用 `模块名:类名`，原有的完整导入路径仍然支持：

```yaml
observation_types:
  base_ang_vel: BaseAngularVelocityObservation
  joint_pos_rel: JointPositionObservation
  prev_action: PreviousActionObservation
  custom_obs: "custom_observations:CustomObservation"
```

G1 policy 必须按训练时使用的 IMU 设置 `imu_source`。`pelvis` 读取
`LowState.imu_state`，`torso` 订阅 `rt/secondary_imu`；G1 仿真会同时发布两路。
其他机器人必须省略 `imu_source`，始终读取各自的 `LowState.imu_state`。

## 🔁 多策略切换

需要在运行时切换多个 policy 时，使用 `--multi-ckpt`：

```yaml
default: vanilla_ppo_flat

ckpts:
  vanilla_ppo_flat: "./vanilla_ppo_flat"
  unitree_rl_lab_flat: "./unitree_rl_lab_flat"

switch:
  enabled: true
  button: B
  order: [vanilla_ppo_flat, unitree_rl_lab_flat]
  only_when: [run_policy]
  on_switch: null
```

仿真中按 `b` 切换 policy；真机遥控器默认使用 `B`。同一个 manifest 下的 policy 必须共享 `sdk_joint_order` 和 `policy_step_dt`，但 observation、action、gain 和 ONNX 文件可以不同。

## ⌨️ 默认控制

默认状态机使用这些遥控器按键：

- `A`: 移动到默认关节位置。
- `Start`: 运行当前 policy。
- `B`: 在启用 multi-ckpt 时切换 policy。
- `X`: 回到 damping。

仿真中的键盘映射是：`enter` 对应 `A`，`\` 对应 `Start`，`b` 对应 `B`，`x` 对应 `X`。
