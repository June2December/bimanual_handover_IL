import numpy as np
from isaacsim.core.utils.types import ArticulationAction


def get_pos(xform_prim):
    positions, _ = xform_prim.get_world_poses()
    return np.array(positions[0], dtype=np.float64)


def arm6_deg_to_rad(arm_deg):
    return np.deg2rad(np.array(arm_deg, dtype=np.float64))


def full_q_from_arm6(arm_deg):
    return np.concatenate([arm6_deg_to_rad(arm_deg), np.zeros(6, dtype=np.float64)])


def apply_arm_hold_with_grip(robot, arm_hold_q, finger_idx, grip_val):
    """
    arm_hold_q: close/release 시작 시점의 전체 joint position
    finger_idx만 바꾸고 나머지 joint는 그대로 유지
    """
    full = arm_hold_q.copy()
    full[finger_idx] = grip_val
    robot.apply_action(ArticulationAction(joint_positions=full))