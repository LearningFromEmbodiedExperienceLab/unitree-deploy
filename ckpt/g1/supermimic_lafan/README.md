# SuperMimic general motion tracker

Copy this whole directory to ckpt/g1/<policy>/ in unitree-deploy.
policy.onnx contains the normalized EMA actor, with policy (sensor history) and goal inputs.
No motion clip or frame lookup is embedded; the same model tracks runtime references.
It uses the shared `MotionTrackingPolicy` entry point and `motion_tracking_obs` module.
`motion_include_endpoint: false` matches training by stopping local clips at the last
whole control step; both direct loading and the DDS file publisher use this setting.

For a local any4hdmi clip, add `motion_path: /path/to/clip.npz` to policy.yaml.
For DDS references, leave motion_path unset and run the motion source in another terminal:

```sh
unitree-motion-source --config ckpt/g1/<policy>/policy.yaml --net lo --motion-path /path/to/clip.npz
```

The existing motion source also supports live retargeting. The controller buffers the ten-frame
future horizon before inference. Future goals and reference-residual joint targets are built
from that runtime reference, preserving training coordinates and velocity scaling.

Run the standalone GMT controller:

```sh
unitree-sim-bridge --robot g1
unitree-controller --mode sim --robot g1 --net lo --ckpt ckpt/g1/<policy>
```

A moves to the default pose, Start prepares the arms, B confirms first-frame hold, another B plays once ready, X damps.
The optional multi_ckpt.yaml selects this same GMT policy and enables its simulation ghost.
Choose another motion at runtime without exporting the actor again.
