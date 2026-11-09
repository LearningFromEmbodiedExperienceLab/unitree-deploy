"""Shared XRobot to G1 calibration for the native retarget adapters."""

# XRobot calibration from EGalahad/GMR's xrobot_to_g1.json (235b0558).
# Columns: G1 joint frame, human body, position/rotation weights, position scale,
# and human-to-robot local rotation in wxyz. Feet track the toe frame below.
_TORSO = (-.5, .5, -.5, -.5)
XROBOT_G1_TARGETS = (
    ("yaw_joint", "Pelvis", 10, 5, .9, _TORSO),
    ("left_hip_yaw_joint", "Left_Hip", 10, 5, .9, _TORSO),
    ("left_knee_joint", "Left_Knee", 10, 5, .9, _TORSO),
    ("left_ankle_roll_joint", "Left_Foot", 100, 10, .9, _TORSO),
    ("right_hip_yaw_joint", "Right_Hip", 10, 5, .9, _TORSO),
    ("right_knee_joint", "Right_Knee", 10, 5, .9, _TORSO),
    ("right_ankle_roll_joint", "Right_Foot", 100, 10, .9, _TORSO),
    ("waist_pitch_joint", "Spine3", 0, 10, .9, _TORSO),
    ("left_shoulder_yaw_joint", "Left_Shoulder", 0, 10, .8, (2**-.5, 0, 2**-.5, 0)),
    ("left_elbow_joint", "Left_Elbow", 0, 10, .8, (0, 0, 1, 0)),
    ("left_wrist_yaw_joint", "Left_Wrist", 0, 10, .8, (0, 0, 1, 0)),
    ("right_shoulder_yaw_joint", "Right_Shoulder", 0, 10, .8, (0, 2**-.5, 0, -2**-.5)),
    ("right_elbow_joint", "Right_Elbow", 0, 10, .8, (0, 1, 0, 0)),
    ("right_wrist_yaw_joint", "Right_Wrist", 0, 10, .8, (0, 1, 0, 0)),
)
