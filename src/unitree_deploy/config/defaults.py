from __future__ import annotations

from pathlib import Path


DEFAULT_NET = "lo"
DEFAULT_MODE = "sim"

LOWCMD_TOPIC = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"
SECONDARY_IMU_TOPIC = "rt/secondary_imu"
ODOM_TOPIC = "rt/odommodestate"
SIM_PAUSE_TOPIC = "rt/sim_pause"
SIM_SPEED_TOPIC = "rt/sim_speed"
CONTROLLER_STATE_TOPIC = "rt/controller_state"

Z1_LOWCMD_TOPIC = "rt/z1/lowcmd"
Z1_LOWSTATE_TOPIC = "rt/z1/lowstate"
Z1_CONTROLLER_STATE_TOPIC = "rt/z1/controller_state"
Z1_UDP_SERVICE_PATH = str(
    Path(__file__).resolve().parents[4] / "z1_controller/deploy/x86_64/z1_udp_service"
)
Z1_UDP_NAMESPACE = "z1"
Z1_ROBOT_IP = "192.168.123.110"
Z1_LOCAL_PORT = 8881

SIM_HZ = 500
STATE_HZ = 200
RENDER_HZ = 30

BASE_HEIGHT = 1.0
BASE_QUAT = (1.0, 0.0, 0.0, 0.0)  # MuJoCo uses wxyz; identity faces +X.

GYRO_SENSOR_NAMES = ("imu_ang_vel", "imu_gyro")
ACC_SENSOR_NAMES = ("imu_lin_acc", "imu_acc")

# ---------- DDS Topics ----------
GO_ROBOTS = frozenset({"b2", "go2", "b2z1"})  # robots that use unitree_go_msg_dds__LowCmd_
HG_ROBOTS = frozenset({"g1"})  # robots that use unitree_hg_msg_dds__LowCmd_
HG_MODE_PR = 0
HG_MODE_MACHINE = 0

# ---------- Band Parameters ----------
BAND_SITES = ("left_gantry_attach_point", "right_gantry_attach_point")
BAND_CLEARANCE = 0.00
BAND_STIFFNESS = 600.0
BAND_DAMPING = 60.0
BAND_STEP = 0.1
BAND_MIN_Z = 0.8
BAND_MAX_Z = 1.6
BAND_MAX_FORCE = 400.0

# ---------- Remote Joystick Parameters ----------
WIRELESS_REMOTE_BUTTON_BITS = {
    "Start": (2, 2), # enter rl policy
    "A": (3, 0),  # move to default pos
    "B": (3, 1),  # switch rl policy
    "X": (3, 2),  # damping
}
SIM_REMOTE_BUTTON_KEYS = {
    "enter": WIRELESS_REMOTE_BUTTON_BITS["A"],
    "b": WIRELESS_REMOTE_BUTTON_BITS["B"],
    "x": WIRELESS_REMOTE_BUTTON_BITS["X"],
    "\\": WIRELESS_REMOTE_BUTTON_BITS["Start"],
}
def sim_key_for_button(button: str) -> str:
    target_bit = WIRELESS_REMOTE_BUTTON_BITS[button]
    for key, bit in SIM_REMOTE_BUTTON_KEYS.items():
        if bit == target_bit:
            return key
    raise KeyError(f"no sim keyboard key configured for virtual button {button!r}")

# ---------- State Machine Parameters ----------
MOVE_TO_DEFAULT_TIME = 2.0
DAMPING_STATE = "damped"
MOVE_TO_DEFAULT_STATE = "move_to_default_qpos"
RUN_POLICY_STATE = "run_policy"
