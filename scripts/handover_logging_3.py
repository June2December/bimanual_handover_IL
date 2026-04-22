from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

import csv
import time
import numpy as np
import omni.usd
import isaacsim.robot_motion.motion_generation as mg

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction

from rotation_utils import get_pose_and_rot6d, quat_2_rot6d
from gripper_utils import grip_to_state, grip_to_cmd
from episode_checks_utils import check_episode_termination
from robot_utils import get_pos, arm6_deg_to_rad, full_q_from_arm6, apply_arm_hold_with_grip
from target_utils import sample_cylinder_pos, get_right_target_poses
from phase_utils import PHASE_NAMES, phase_to_onehot, phase_feature_names

# =========================
# RMPFlow Controller
# =========================
class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name, robot_articulation, physics_dt=1.0 / 60.0):
        config = mg.interface_config_loader.load_supported_motion_policy_config(
            "UR10", "RMPflow"
        )

        rmp_flow = mg.lula.motion_policies.RmpFlow(**config)
        articulation_policy = mg.ArticulationMotionPolicy(
            robot_articulation, rmp_flow, physics_dt
        )

        super().__init__(name=name, articulation_motion_policy=articulation_policy)

        pos, ori = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=pos,
            robot_orientation=ori,
        )

# =========================
# Global Settings
# =========================
usd_path = "/home/june/bimanual_handover_IL/bimanual_scene.usd"

left_robot_path = "/World/ur_left"
right_robot_path = "/World/ur_right"

left_tcp_path = "/World/ur_left/wrist_3_link/flange/tool0"
right_tcp_path = "/World/ur_right/wrist_3_link/flange/tool0"

cylinder_path = "/World/Cylinder"

physics_dt = 1.0 / 60.0

LEFT_HOME_DEG = [-45, -90, 90, 90, 90, 90]
RIGHT_HOME_DEG = [-135, -90, -90, 90, -90, 90]

RIGHT_WAIT_DEG = [-125, -62, -130, 180, -90, 80]
RIGHT_RECEIVE_DEG = [-121, -97, -139, 237, -90, 90]

LEFT_HOME = full_q_from_arm6(LEFT_HOME_DEG)
RIGHT_HOME = full_q_from_arm6(RIGHT_HOME_DEG)
RIGHT_WAIT = arm6_deg_to_rad(RIGHT_WAIT_DEG)
RIGHT_RECEIVE = arm6_deg_to_rad(RIGHT_RECEIVE_DEG)

RIGHT_WAIT_FULL = full_q_from_arm6(RIGHT_WAIT_DEG)
RIGHT_RECEIVE_FULL = full_q_from_arm6(RIGHT_RECEIVE_DEG)

LEFT_TOP_ORI = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0], dtype=np.float32))
LEFT_TOP_ROT6D = quat_2_rot6d(np.array(LEFT_TOP_ORI, dtype=np.float32))

GRIP_OPEN = 0.0
LEFT_GRIP_CLOSE = 0.43
RIGHT_GRIP_CLOSE = 0.43

LEFT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.4]
RIGHT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.4]
LEFT_OPEN_SEQ = [0.35, 0.25, 0.1, 0.0]

LEFT_PRE_OFFSET = np.array([0.0, 0.0, 0.50], dtype=np.float64)
LEFT_GRASP_OFFSET = np.array([0.0, 0.0, 0.22], dtype=np.float64)

LEFT_LIFT_Z = 0.50
LEFT_TRANSFER_XY = np.array([0.50, 0.00], dtype=np.float64)

LEFT_POST_RELEASE_UP_DZ = 0.05
LEFT_HOME_TOL = 0.08

GRIP_STEP_HOLD = 10
SETTLE_STEPS = 10
END_HOLD_STEPS = 60
MAX_STEPS = 1200
PRINT_EVERY = 30

NUM_EPISODES = 100

LOG_DIR = "/home/june/bimanual_handover_IL/data/handover_logging_3"
os.makedirs(LOG_DIR, exist_ok=True)

# =========================
# Open Scene Once
# =========================
omni.usd.get_context().open_stage(usd_path)
for _ in range(60):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Stage open failed")

if not stage.GetPrimAtPath(left_robot_path).IsValid():
    raise RuntimeError(f"Invalid left robot path: {left_robot_path}")
if not stage.GetPrimAtPath(right_robot_path).IsValid():
    raise RuntimeError(f"Invalid right robot path: {right_robot_path}")
if not stage.GetPrimAtPath(cylinder_path).IsValid():
    raise RuntimeError(f"Invalid cylinder path: {cylinder_path}")

# =========================
# World / Prim Handles
# =========================
world = World(stage_units_in_meters=1.0)
world.reset()

left_tcp = XFormPrim(left_tcp_path)
right_tcp = XFormPrim(right_tcp_path)
cyl = XFormPrim(cylinder_path)

left_tcp.initialize()
right_tcp.initialize()
cyl.initialize()

# =========================
# Phase command builder
# =========================
def build_phase_outputs(
    phase,
    left_pre_target,
    left_grasp_target,
    left_lift_target,
    left_transfer_target,
    left_post_release_up_target,
    left_home_tcp_target,
    right_wait_tcp_pos,
    right_wait_rot6d,
    right_receive_tcp_pos,
    right_receive_rot6d,
    sub_idx,
):
    # left
    if phase == 0:
        left_target = left_pre_target
        left_grip_cmd = GRIP_OPEN
    elif phase == 1:
        left_target = left_grasp_target
        left_grip_cmd = GRIP_OPEN
    elif phase == 2:
        left_target = left_grasp_target
        left_grip_cmd = GRIP_OPEN
    elif phase == 3:
        left_target = left_grasp_target
        left_grip_cmd = LEFT_CLOSE_SEQ[min(sub_idx, len(LEFT_CLOSE_SEQ)-1)]
    elif phase == 4:
        left_target = left_lift_target
        left_grip_cmd = LEFT_GRIP_CLOSE
    elif phase == 5:
        left_target = left_transfer_target
        left_grip_cmd = LEFT_GRIP_CLOSE
    elif phase == 6:
        left_target = left_transfer_target
        left_grip_cmd = LEFT_GRIP_CLOSE
    elif phase == 7:
        left_target = left_transfer_target
        left_grip_cmd = LEFT_GRIP_CLOSE
    elif phase == 8:
        left_target = left_transfer_target
        left_grip_cmd = LEFT_OPEN_SEQ[min(sub_idx, len(LEFT_OPEN_SEQ)-1)]
    elif phase == 9:
        left_target = left_post_release_up_target
        left_grip_cmd = GRIP_OPEN
    elif phase == 10:
        left_target = left_home_tcp_target
        left_grip_cmd = GRIP_OPEN
    elif phase == 11:
        left_target = left_home_tcp_target
        left_grip_cmd = GRIP_OPEN
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # right
    if phase in [0, 1, 2, 3, 4, 5]:
        action_right_pos = np.array(right_wait_tcp_pos, dtype=np.float32)
        action_right_rot6d = np.array(right_wait_rot6d, dtype=np.float32)
    else:
        action_right_pos = np.array(right_receive_tcp_pos, dtype=np.float32)
        action_right_rot6d = np.array(right_receive_rot6d, dtype=np.float32)

    if phase in [0, 1, 2, 3, 4, 5, 6]:
        right_grip_cmd = GRIP_OPEN
    elif phase == 7:
        right_grip_cmd = RIGHT_CLOSE_SEQ[min(sub_idx, len(RIGHT_CLOSE_SEQ)-1)]
    elif phase in [8, 9, 10, 11]:
        right_grip_cmd = RIGHT_GRIP_CLOSE
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return (
        np.array(left_target, dtype=np.float32),
        float(left_grip_cmd),
        action_right_pos,
        action_right_rot6d,
        float(right_grip_cmd),
    )

# =========================
# One Episode
# =========================
def handover_episode(csv_path, cyl_spawn_pos):
    left_robot = SingleArticulation(left_robot_path, name="left_robot")
    right_robot = SingleArticulation(right_robot_path, name="right_robot")

    left_robot.initialize()
    right_robot.initialize()

    left_robot.set_joints_default_state(LEFT_HOME, np.zeros_like(LEFT_HOME))
    right_robot.set_joints_default_state(RIGHT_HOME, np.zeros_like(RIGHT_HOME))

    world.reset()

    left_robot.initialize()
    right_robot.initialize()

    left_finger_idx = left_robot.dof_names.index("finger_joint")
    right_finger_idx = right_robot.dof_names.index("finger_joint")

    left_robot.set_joint_positions(LEFT_HOME)
    right_robot.set_joint_positions(RIGHT_HOME)

    curr_pos, curr_ori = cyl.get_world_poses()
    cyl.set_world_poses(
        positions=np.array([cyl_spawn_pos], dtype=np.float64),
        orientations=curr_ori,
    )

    for _ in range(30):
        world.step(render=True)

    (
        right_wait_tcp_pos,
        right_wait_rot6d,
        right_receive_tcp_pos,
        right_receive_rot6d,
    ) = get_right_target_poses(
        world=world,
        right_robot=right_robot,
        right_tcp=right_tcp,
        right_wait_full=RIGHT_WAIT_FULL,
        right_receive_full=RIGHT_RECEIVE_FULL,
        right_home=RIGHT_HOME,
        get_pose_and_rot6d=get_pose_and_rot6d,
    )

    left_ctrl = RMPFlowController("left_ctrl", left_robot, physics_dt)

    cyl_start_pos = get_pos(cyl)
    left_pre_target = cyl_start_pos + LEFT_PRE_OFFSET
    left_grasp_target = cyl_start_pos + LEFT_GRASP_OFFSET

    left_lift_target = None
    left_transfer_target = None
    left_post_release_up_target = None
    left_home_tcp_target = left_pre_target

    phase = 0
    sub_idx = 0
    sub_hold = 0
    settle_count = 0
    end_hold = 0

    left_grasped = 0.0
    right_grasped = 0.0

    left_arm_hold_q_close = None
    left_arm_hold_q_release = None

    progress_hist = []
    fail_reason = None

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    phase_cols = phase_feature_names()

    csv_writer.writerow([
        "step",
        "phase",
        "phase_name",

        "left_pos_x", "left_pos_y", "left_pos_z",
        "left_rot6d_0", "left_rot6d_1", "left_rot6d_2",
        "left_rot6d_3", "left_rot6d_4", "left_rot6d_5",

        "right_pos_x", "right_pos_y", "right_pos_z",
        "right_rot6d_0", "right_rot6d_1", "right_rot6d_2",
        "right_rot6d_3", "right_rot6d_4", "right_rot6d_5",

        "obj_pos_x", "obj_pos_y", "obj_pos_z",
        "obj_rot6d_0", "obj_rot6d_1", "obj_rot6d_2",
        "obj_rot6d_3", "obj_rot6d_4", "obj_rot6d_5",

        "left_grip_state",
        "right_grip_state",

        *phase_cols,

        "action_left_pos_x", "action_left_pos_y", "action_left_pos_z",
        "action_left_rot6d_0", "action_left_rot6d_1", "action_left_rot6d_2",
        "action_left_rot6d_3", "action_left_rot6d_4", "action_left_rot6d_5",
        "action_left_grip_cmd",

        "action_right_pos_x", "action_right_pos_y", "action_right_pos_z",
        "action_right_rot6d_0", "action_right_rot6d_1", "action_right_rot6d_2",
        "action_right_rot6d_3", "action_right_rot6d_4", "action_right_rot6d_5",
        "action_right_grip_cmd",
    ])

    print("\n[INFO] Episode start")
    print("csv_path                 =", csv_path)
    print("cyl_start_pos            =", np.round(cyl_start_pos, 4).tolist())
    print("right_wait_tcp_pos       =", np.round(right_wait_tcp_pos, 4).tolist())
    print("right_receive_tcp_pos    =", np.round(right_receive_tcp_pos, 4).tolist())

    success = False

    try:
        for step in range(MAX_STEPS):
            left_tcp_pos, left_rot6d = get_pose_and_rot6d(left_tcp)
            right_tcp_pos, right_rot6d = get_pose_and_rot6d(right_tcp)
            obj_pos, obj_rot6d = get_pose_and_rot6d(cyl)
            right_q = right_robot.get_joint_positions()[:6]

            # =========================================
            # 1) PHASE TRANSITION FIRST
            # =========================================
            if phase == 0:
                if np.linalg.norm(left_tcp_pos - left_pre_target) < 0.03:
                    phase = 1

            elif phase == 1:
                if np.linalg.norm(left_tcp_pos - left_grasp_target) < 0.035:
                    phase = 2
                    settle_count = 0

            elif phase == 2:
                settle_count += 1
                if settle_count >= SETTLE_STEPS:
                    phase = 3
                    sub_idx = 0
                    sub_hold = 0
                    left_arm_hold_q_close = left_robot.get_joint_positions().copy()

            elif phase == 3:
                sub_hold += 1
                if sub_hold >= GRIP_STEP_HOLD:
                    sub_hold = 0
                    sub_idx += 1
                    if sub_idx >= len(LEFT_CLOSE_SEQ):
                        left_grasped = 1.0
                        left_lift_target = np.array(
                            [left_tcp_pos[0], left_tcp_pos[1], LEFT_LIFT_Z],
                            dtype=np.float64,
                        )
                        phase = 4

            elif phase == 4:
                if left_lift_target is not None and np.linalg.norm(left_tcp_pos - left_lift_target) < 0.035:
                    dx = LEFT_TRANSFER_XY[0] - left_tcp_pos[0]
                    dy = LEFT_TRANSFER_XY[1] - left_tcp_pos[1]
                    left_transfer_target = np.array(
                        [left_tcp_pos[0] + dx, left_tcp_pos[1] + dy, left_tcp_pos[2]],
                        dtype=np.float64,
                    )
                    phase = 5

            elif phase == 5:
                if left_transfer_target is not None and np.linalg.norm(left_tcp_pos - left_transfer_target) < 0.04:
                    phase = 6

            elif phase == 6:
                if np.all(np.abs(right_q - RIGHT_RECEIVE) < np.deg2rad(2.0)):
                    phase = 7
                    sub_idx = 0
                    sub_hold = 0

            elif phase == 7:
                sub_hold += 1
                if sub_hold >= GRIP_STEP_HOLD:
                    sub_hold = 0
                    sub_idx += 1
                    if sub_idx >= len(RIGHT_CLOSE_SEQ):
                        right_grasped = 1.0
                        phase = 8
                        sub_idx = 0
                        sub_hold = 0
                        left_arm_hold_q_release = left_robot.get_joint_positions().copy()

            elif phase == 8:
                sub_hold += 1
                if sub_hold >= GRIP_STEP_HOLD:
                    sub_hold = 0
                    sub_idx += 1
                    if sub_idx >= len(LEFT_OPEN_SEQ):
                        left_grasped = 0.0
                        left_post_release_up_target = np.array(
                            [
                                left_tcp_pos[0],
                                left_tcp_pos[1],
                                left_tcp_pos[2] + LEFT_POST_RELEASE_UP_DZ,
                            ],
                            dtype=np.float64,
                        )
                        phase = 9

            elif phase == 9:
                if left_post_release_up_target is not None and np.linalg.norm(left_tcp_pos - left_post_release_up_target) < 0.03:
                    phase = 10

            elif phase == 10:
                if np.linalg.norm(left_tcp_pos - left_home_tcp_target) < LEFT_HOME_TOL:
                    phase = 11
                    end_hold = 0

            elif phase == 11:
                end_hold += 1
                if end_hold >= END_HOLD_STEPS:
                    success = True
                    break

            # =========================================
            # 2) BUILD TARGET / GRIP FROM UPDATED PHASE
            # =========================================
            (
                action_left_pos,
                left_grip_cmd,
                action_right_pos,
                action_right_rot6d,
                right_grip_cmd,
            ) = build_phase_outputs(
                phase=phase,
                left_pre_target=left_pre_target,
                left_grasp_target=left_grasp_target,
                left_lift_target=left_lift_target if left_lift_target is not None else left_grasp_target,
                left_transfer_target=left_transfer_target if left_transfer_target is not None else left_grasp_target,
                left_post_release_up_target=left_post_release_up_target if left_post_release_up_target is not None else left_grasp_target,
                left_home_tcp_target=left_home_tcp_target,
                right_wait_tcp_pos=right_wait_tcp_pos,
                right_wait_rot6d=right_wait_rot6d,
                right_receive_tcp_pos=right_receive_tcp_pos,
                right_receive_rot6d=right_receive_rot6d,
                sub_idx=sub_idx,
            )

            action_left_rot6d = np.array(LEFT_TOP_ROT6D, dtype=np.float32)

            # =========================================
            # 3) TERMINATION CHECK WITH CURRENT LABELS
            # =========================================
            progress_metric = float(np.linalg.norm(left_tcp_pos - action_left_pos))
            progress_hist.append(progress_metric)

            should_stop, fail_reason = check_episode_termination(
                step=step,
                max_steps=MAX_STEPS,
                progress_hist=progress_hist,
                phase=phase,
                left_grasped=left_grasped,
                right_grasped=right_grasped,
                left_tcp_pos=left_tcp_pos,
                right_tcp_pos=right_tcp_pos,
                obj_pos=obj_pos,
            )
            if should_stop:
                success = False
                break

            phase_onehot = phase_to_onehot(phase)

            left_grip_state = grip_to_state(left_grip_cmd)
            right_grip_state = grip_to_state(right_grip_cmd)

            left_grip_cmd_disc = grip_to_cmd(left_grip_cmd)
            right_grip_cmd_disc = grip_to_cmd(right_grip_cmd)

            # =========================================
            # 4) CONTROL
            # =========================================
            if phase in [0, 1, 2, 4, 5, 6, 9, 10]:
                left_action = left_ctrl.forward(
                    target_end_effector_position=action_left_pos,
                    target_end_effector_orientation=LEFT_TOP_ORI,
                )
                left_robot.apply_action(left_action)

            elif phase == 3:
                apply_arm_hold_with_grip(
                    robot=left_robot,
                    arm_hold_q=left_arm_hold_q_close,
                    finger_idx=left_finger_idx,
                    grip_val=left_grip_cmd,
                )

            elif phase == 8:
                apply_arm_hold_with_grip(
                    robot=left_robot,
                    arm_hold_q=left_arm_hold_q_release,
                    finger_idx=left_finger_idx,
                    grip_val=left_grip_cmd,
                )

            else:
                left_full = left_robot.get_joint_positions().copy()
                left_full[left_finger_idx] = left_grip_cmd
                left_robot.apply_action(ArticulationAction(joint_positions=left_full))

            if phase in [0, 1, 2, 3, 4, 5]:
                right_arm_target = RIGHT_WAIT
            else:
                right_arm_target = RIGHT_RECEIVE

            right_full = right_robot.get_joint_positions().copy()
            right_full[:6] = right_arm_target
            right_full[right_finger_idx] = right_grip_cmd
            right_robot.apply_action(ArticulationAction(joint_positions=right_full))

            # =========================================
            # 5) LOG BEFORE STEP ADVANCES
            # =========================================
            csv_writer.writerow([
                step,
                phase,
                PHASE_NAMES[phase],

                float(left_tcp_pos[0]), float(left_tcp_pos[1]), float(left_tcp_pos[2]),
                float(left_rot6d[0]), float(left_rot6d[1]), float(left_rot6d[2]),
                float(left_rot6d[3]), float(left_rot6d[4]), float(left_rot6d[5]),

                float(right_tcp_pos[0]), float(right_tcp_pos[1]), float(right_tcp_pos[2]),
                float(right_rot6d[0]), float(right_rot6d[1]), float(right_rot6d[2]),
                float(right_rot6d[3]), float(right_rot6d[4]), float(right_rot6d[5]),

                float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]),
                float(obj_rot6d[0]), float(obj_rot6d[1]), float(obj_rot6d[2]),
                float(obj_rot6d[3]), float(obj_rot6d[4]), float(obj_rot6d[5]),

                float(left_grip_state),
                float(right_grip_state),

                *phase_onehot.tolist(),

                float(action_left_pos[0]), float(action_left_pos[1]), float(action_left_pos[2]),
                float(action_left_rot6d[0]), float(action_left_rot6d[1]), float(action_left_rot6d[2]),
                float(action_left_rot6d[3]), float(action_left_rot6d[4]), float(action_left_rot6d[5]),
                float(left_grip_cmd_disc),

                float(action_right_pos[0]), float(action_right_pos[1]), float(action_right_pos[2]),
                float(action_right_rot6d[0]), float(action_right_rot6d[1]), float(action_right_rot6d[2]),
                float(action_right_rot6d[3]), float(action_right_rot6d[4]), float(action_right_rot6d[5]),
                float(right_grip_cmd_disc),
            ])

            world.step(render=True)

            if step % PRINT_EVERY == 0:
                print(
                    f"[STEP {step:04d}] "
                    f"phase={phase}({PHASE_NAMES[phase]}) "
                    f"obj={np.round(obj_pos, 4).tolist()} "
                    f"left_grip={left_grip_cmd:.2f} "
                    f"right_grip={right_grip_cmd:.2f}"
                )

    except Exception as e:
        print(f"[ERROR] Episode failed: {e}")
        success = False

    finally:
        csv_file.close()

    if (not success) and os.path.exists(csv_path):
        os.remove(csv_path)
        if fail_reason is not None:
            print(f"[INFO] Deleted failed episode csv: {csv_path} ({fail_reason})")
        else:
            print(f"[INFO] Deleted failed episode csv: {csv_path}")

    return success

# =========================
# Episode Wrapper
# =========================
def run_one_episode(ep_idx):
    new_pos = sample_cylinder_pos()

    csv_path = os.path.join(
        LOG_DIR,
        f"handover_ep_{ep_idx:03d}_{int(time.time())}.csv"
    )

    success = handover_episode(csv_path=csv_path, cyl_spawn_pos=new_pos)
    return success

# =========================
# Main loop
# =========================
success_count = 0

for ep in range(NUM_EPISODES):
    print(f"\n================ EPISODE {ep} ================")
    success = run_one_episode(ep)

    if success:
        success_count += 1
        print(f"[EP {ep}] SUCCESS")
    else:
        print(f"[EP {ep}] FAIL")

print("\n====================")
print(f"TOTAL SUCCESS: {success_count}/{NUM_EPISODES}")
print("====================")

simulation_app.close()