# Unitree Deploy

[简体中文](README.zh-CN.md)

Lightweight deployment codebase for Unitree robots.

This repository wraps ONNX policies, robot observations, DDS topics, MuJoCo
models, and simple controller state machines into a reusable deployment flow.
It is useful for validating policies in simulation first, then running the same
controller path on supported Unitree hardware.

## ✨ Highlights

- Shared controller runtime for `sim` and `real` modes.
- ONNX policy loading with configurable `yaml` file.
- Multi-policy switching support.
- G1 motion tracking with a configurable reference ghost.
- Live MuJoCo camera export to MP4, including the reference ghost and simulation speed.
- Template generator for custom deployment demands (such as customized obs).

## 📦 Project Layout

```text
src/unitree_deploy/
├── cli/              # console entry points
├── runtime/
│   ├── controllers/  # policy/Z1 control loops and checkpoint loading
│   ├── state_machine/
│   │   ├── machine.py   # state lifecycle and transition dispatch
│   │   ├── config.py    # YAML, events, targets, and gains
│   │   ├── registry.py  # state construction and transition validation
│   │   ├── filters.py   # shared target smoothing
│   │   └── states/      # pose, policy, and arm-transition implementations
│   ├── simulation/   # MuJoCo bridge and keyboard input
│   ├── transport/    # Unitree DDS adapters and remote decoding
│   ├── trajectory/   # recording and replay applications
│   └── timing.py     # shared loop pacing
├── policy/
│   ├── base_policy.py  # shared ONNX wrapper and checkpoint loader
│   └── motion_tracking/  # tracking policies and references
├── obs/              # shared and policy-specific observation terms
├── visualization/    # robot state visualizer
├── utils/            # shared helpers
└── robot_model/      # bundled MuJoCo robot and terrain assets

ckpt/
├── g1/               # example G1 policies and multi-ckpt config
└── go2/              # example Go2 policies
```

## 🚀 Setup

```bash
uv sync
source .venv/bin/activate
```

For the mjviser browser viewer extras (realtime robot state visualization):

```bash
uv sync --extra viewer
```

`unitree_sdk2_python` is expected to be installed separately in the same Python
environment.

## 🕹️ Quick Start

- Run the MuJoCo-to-DDS bridge:

    ```bash
    unitree-sim-bridge --robot g1
    ```

    An in-window key guide lists movement, policy, and simulation controls.
    Press `?` (`Shift` + `/`) to show or hide it.
    Press `T` in the MuJoCo Viewer to toggle camera tracking.
    Press `F12` in the viewer to toggle sim2sim keyboard controls and mujoco viewer's native shortcuts.
    Camera tracking requires `viewer.camera.track_body` to be configured in the robot's `visualizer.yaml`.
    Scene lights follow the robot's position independently of camera tracking.
    The current controller state and simulation speed remain visible at the top left
    when keyboard hints are hidden. B2Z1 also shows the independent Z1 controller state.

    Velocity-command policies show target/actual linear velocity in blue/cyan and
    target/actual yaw rate in green/light green, following mjlab's arrow convention.
    Press `G` to toggle both velocity arrows and the motion reference ghost.
    Visibility persists across policy switches and applies to MP4 export as well.

    Adjust velocity arrow placement and size under the existing `viewer` section
    in each robot's `visualizer.yaml`:

    ```yaml
    viewer:
      policy_debug_viz:
        velocity:
          offset: [0.0, 0.0, 0.3]  # base-link frame, meters; raised for Go2
          scale: 0.5              # shared length scale for target/actual arrows
          width: 0.015            # meters
    ```

    Defaults are `offset: [0.0, 0.0, 0.1]`, `scale: 0.5`, and `width: 0.015`.
    The offset is independent of scale. Restart the bridge after editing;
    MP4 export uses the same settings.

    Press `=` to go faster or `-` to go slower through physics speeds `0.5x`,
    `1x`, `2x`, `3x`, `4x`, `5x`, `6x`, `7x`, and `8x` (default: `1x`), stopping at either end.
    The physics timestep stays fixed; the selected speed
    appears in the bridge's terminal status and also applies after resuming
    with `space`. In simulation, the policy controller follows the same speed
    setting so physics and motion-reference playback advance together.
    `Space` pauses physics, policy updates, and motion-reference playback together;
    resume continues from the same frame. Restart both bridge and controller
    after updating to use synchronized pause.

- Run a controller against the simulator:

    ```bash
    # By default, it loads `policy.yaml` in the provides ckpt path
    unitree-controller --mode sim --ckpt ckpt/g1/vanilla_ppo_flat

    # Or specify a policy YAML file directly.
    unitree-controller --mode sim --ckpt ckpt/g1/vanilla_ppo_flat/policy_torso_base.yaml
    ```

- Run with a multi-policy manifest:

    ```bash
    unitree-controller --mode sim --multi-ckpt ckpt/g1/multi_ckpt.yaml
    ```

- Run the bundled G1 motion-tracking profile:

    ```bash
    unitree-controller --mode sim --robot g1 \
      --multi-ckpt ckpt/g1/tracking_lafan_dance1_subject3/multi_ckpt.yaml
    ```

    With the G1 bridge running, press `Enter` to move to the default pose, then
    `\` to start locomotion. Press `B` to prepare the arms and enter tracking.
    At the end of the clip, the robot holds the final reference frame; press
    `B` again to return to locomotion. `X` enters damping.

    Configure the reference ghost in the tracking profile's `multi_ckpt.yaml`:

    ```yaml
    motion_ghost:
      enabled: true             # initial visibility of policy debug viz
      toggle_key: g             # shared by the ghost and velocity arrows
      color: [0.35, 0.65, 1.0]  # RGB values from 0 to 1
      opacity: 0.25             # 0 = invisible, 1 = opaque
    ```

    These are the default color and opacity. The ghost follows the controller's
    aligned reference pose and draws only the robot's visual meshes, with no
    collision or dynamics. Press `G` to toggle policy debug viz without changing motion
    playback. Ghost settings apply only in simulation.

    The MuJoCo overlay shows simulation speed and keyboard hints without motion
    details. Restart the controller after changing the YAML settings.

- Start the viser visualizer:

    ```bash
    unitree-visualizer --mode sim --robot g1
    ```

- Export the live MuJoCo camera to MP4 (requires `ffmpeg` on `PATH`):

    ```bash
    unitree-sim-bridge --robot g1 --video export/tracking.mp4 \
      --video-width 1280 --video-height 720 --video-fps 25 \
      --video-start 5 --video-seconds 12
    ```

    The video follows the interactive camera and includes the visible reference
    ghost and simulation speed. Start and duration use simulation time, so pauses
    add no duplicate frames and changing simulation speed does not change video
    playback speed. Rendering and encoding can slow the live simulation.

    | Option | Default | Meaning |
    | --- | --- | --- |
    | `--video` | Disabled | Output MP4 path; existing files are not overwritten. |
    | `--video-width` | `960` | Width in pixels; a positive even integer. |
    | `--video-height` | `540` | Height in pixels; a positive even integer. |
    | `--video-fps` | `25` | Positive frame rate, up to the simulation rate. |
    | `--video-start` | `0` | Start time in simulation seconds. |
    | `--video-seconds` | Until viewer closes | Recording duration in simulation seconds. |

    A simulator reset ends the recording. Closing the viewer finalizes the MP4.

- Record a MuJoCo/DDS trajectory for rendering or analysis:

    ```bash
    unitree-trajectory-recorder \
      --robot g1 \
      --out export/runs/g1_walk \
      --record-hz 30
    ```

    Run it in a separate terminal while the robot is running:

    Press `o` in the recorder process to start recording, then press `o` again to
    stop and save. Each recording is written under a timestamped subdirectory such
    as `export/runs/g1_walk/20260629-153012/`, containing `trajectory.npz`,
    `metadata.json`, and `scene.json`.

- Replay a saved trajectory. The default replay frontend is Viser:

    ```bash
    unitree-trajectory-replay export/runs/g1_walk/20260629-153012/trajectory.npz
    ```

    To replay in the native MuJoCo Viewer instead:

    ```bash
    unitree-trajectory-replay \
      --viewer mujoco \
      export/runs/g1_walk/20260629-153012/trajectory.npz
    ```

    The replay UI starts paused and follows the robot by default. It includes
    pause, follow, playback speed, frame scrubbing, and one-step forward/back
    controls. In MuJoCo Viewer, use `space` to play/pause, left/right arrows to
    step, `r` to restart, `+/-` to change speed, and `l` to toggle looping.

- For browser-based policy presentation, use the separate [`policy-web-viewer`](https://github.com/syw-robotics/policy-web-viewer) project:

    Check it out at [here](https://syw-robotics.github.io/policy-web-viewer/)

- For real hardware, pass the DDS network interface explicitly:

    ```bash
    unitree-controller --mode real --net <interface> --ckpt ckpt/g1/vanilla_ppo_flat
    ```

### Share RealSense on the same machine

Capture shares the latest camera frame with Viser and policies via Linux shared memory.

```bash
uv pip install pyrealsense2
unitree-realsense --robot g1
unitree-visualizer --mode real --robot g1 --net eth0 --camera
```

Use the same camera name from `visualizer.yaml` in both consumers. Policies can
read it through `unitree_deploy.runtime.transport.realsense.RealSenseReader`.

### Z1 Impedance Controller

The Z1 controller uses the same `A`/`Start`/`X` state-machine flow as the
locomotion controller:

- `A`: move from the measured arm pose to `default_qpos`.
- `Start`: enter impedance mode after the default-pose move completes.
- `X`: return to damping.

Real mode automatically starts the fixed `z1_udp_service` bridge and uses
`rt/z1/lowstate` and `rt/z1/lowcmd`:

```bash
unitree-z1-controller \
  --mode real \
  --net <interface> \
  --ckpt ckpt/b2z1/b2z1_manip
```

Simulation mode never starts the UDP bridge. For B2Z1 sim2sim, start the
MuJoCo bridge first:

```bash
unitree-sim-bridge --robot b2z1 --net lo
```

Run the B2 locomotion controller when the simulated base also needs active
control:

```bash
unitree-controller \
  --mode sim \
  --robot b2z1 \
  --net lo \
  --ckpt ckpt/b2z1/b2z1_loco
```

Then run the Z1 controller:

```bash
unitree-z1-controller \
  --mode sim \
  --net lo \
  --ckpt ckpt/b2z1/b2z1_manip
```

The controller starts in damping and never moves the arm automatically.
The bridge maps `rt/lowcmd`/`rt/lowstate` to the 12 B2 actuators and
`rt/z1/lowcmd`/`rt/z1/lowstate` to the 7 Z1 actuators. Both controllers consume
the same simulated wireless remote, so `A`, `Start`, and `X` affect both state
machines when they run together.

## Pico humanoid teleoperation / MimicLite

Start G1 simulation and the controller in separate terminals:

```bash
unitree-sim-bridge --robot g1 --net lo
unitree-controller --mode sim --robot g1 --net lo --ckpt ckpt/g1/mimic_lite
```

`Enter` to move to the default pose. Once the reference source is ready, press `\` to prepare.
After initial pose is ready, press `b` play.

Press `g` to toggle the reference ghost and `x` for damping.

`-` and `=` change playback speed.

Choose a motion reference source in a third terminal:

```bash
# Recorded any4hdmi G1 clip (keep the dataset's manifest.json)
unitree-motion-source --config ckpt/g1/mimic_lite/policy.yaml --motion-path /path/to/clip.npz

# Or Pico (XRoboToolkit PC Service + full-body tracking)
unitree-motion-source --config ckpt/g1/mimic_lite/policy.yaml --human-height 1.75 --retarget gmr
```

The motion source starts an mjviser preview of the G1 reference and prints its browser URL.
Install the `viewer` extras; use `--no-visualizer` for headless publishing.

For local files you can also pass `--motion-path` directly to the controller.
GMT keeps tracking the last valid frame after playback or a lost DDS source.
Start the next publisher to prepare its first frame, then press `B` to play.

For Pico, stand upright for calibration and use Pico `X` to toggle live input.
Install `xrobotoolkit_sdk` and the selected backend (XRobot-enabled GMR, FRAX, or RoboKit).

On hardware, use `--mode real` and set the DDS interface with `--net` on both
processes. The Unitree remote uses `A` → `Start` → `B`, with `X` for damping.

## 🧩 Deployment Folders

A policy deployment folder usually looks like this:

```text
ckpt/<robot>/<policy>/
├── policy.yaml             # policy, observation, joint order, and gain config
├── policy.onnx             # exported ONNX policy, or another relative path
├── custom_observations.py  # [optional] custom observation terms
└── custom_policy.py        # [optional] custom inference/action logic
```

- Generate a starter folder interactively:

    ```bash
    unitree-plugin-template
    ```

    Or script it:

    ```bash
    unitree-plugin-template ckpt --robot g1 --name my_policy
    ```

- Key `policy.yaml` fields:

```yaml
policy_path: "policy.onnx"
imu_source: pelvis  # G1: pelvis or torso
obs_joint_order: [...]
action_joint_order: [...]
sdk_joint_order: [...]
```

`obs_joint_order`, `action_joint_order`, and `sdk_joint_order` are matched by
joint name. Reorder indices are derived automatically, so they do not need to be
written by hand.

In `observation_types`, built-in classes accept bare names; custom classes use
`module:Class`. Existing full import paths remain supported:

```yaml
observation_types:
  base_ang_vel: BaseAngularVelocityObservation
  joint_pos_rel: JointPositionObservation
  prev_action: PreviousActionObservation
  custom_obs: "custom_observations:CustomObservation"
```

G1 policies must set `imu_source` to match training. `pelvis` reads
`LowState.imu_state`; `torso` subscribes to `rt/secondary_imu`. The simulator
publishes both sources for G1. Other robots must omit `imu_source` and always
use their own `LowState.imu_state`.

## 🔁 Multi-Policy Switching

Use `--multi-ckpt` when several policies should be switchable at runtime:

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

In simulation, press `b` to switch policies. On hardware, use the remote `B`
button. Policies in one manifest must share `sdk_joint_order` and
`policy_step_dt`; observations, actions, gains, and ONNX files may differ.

## ⌨️ Default Controls

The default state machine uses these remote buttons:

- `A`: move to default joint positions.
- `Start`: run the active policy.
- `B`: switch policy when multi-ckpt switching is enabled.
- `X`: return to damping.

In simulation, the mapped keys are `enter` for `A`, `\` for `Start`, `b` for
`B`, and `x` for `X`.

## ✅ TODO

- [ ] Max torque clipping.
- [x] G1 motion tracking policy support.
- [x] Pico humanoid teleoperation over DDS with MimicLite.
- [ ] Check viser usability for odometry and RealSense hardware deployment.
- [ ] Refine Z1 controller code
