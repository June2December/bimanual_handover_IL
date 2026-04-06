from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import csv
import time
import numpy as np
import omni.usd
import isaacsim.robot_motion.motion_generation as mg

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction


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
# Helper
# =========================
def get_pos(xform_prim):
    positions, _ = xform_prim.get_world_poses()
    return np.array(positions[0], dtype=np.float64)


def arm6_deg_to_rad(arm_deg):
    return np.deg2rad(np.array(arm_deg, dtype=np.float64))


def full_q_from_arm6(arm_deg):
    return np.concatenate([arm6_deg_to_rad(arm_deg), np.zeros(6, dtype=np.float64)])


def clip_delta(target, current, max_step=0.02):
    delta = np.array(target, dtype=np.float64) - np.array(current, dtype=np.float64)
    norm = np.linalg.norm(delta)
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if norm > max_step:
        delta = delta / norm * max_step
    return delta


def apply_arm_hold_with_grip(robot, arm_hold_q, finger_idx, grip_val):
    """
    arm_hold_q: close/release 시작 시점의 전체 joint position
    finger_idx만 바꾸고 나머지 joint는 그대로 유지
    """
    full = arm_hold_q.copy()
    full[finger_idx] = grip_val
    robot.apply_action(ArticulationAction(joint_positions=full))


def sample_cylinder_pos():
    return np.array([
        np.random.uniform(0.50, 0.58),
        0.20 + np.random.uniform(-0.03, 0.03),
        0.05,
    ], dtype=np.float64)


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

LEFT_TOP_ORI = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))

GRIP_OPEN = 0.0
LEFT_GRIP_CLOSE = 0.43
RIGHT_GRIP_CLOSE = 0.43

LEFT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.4]
RIGHT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.4]
LEFT_OPEN_SEQ = [0.35, 0.25, 0.1, 0.0]

LEFT_PRE_OFFSET = np.array([0.0, 0.0, 0.50])
LEFT_GRASP_OFFSET = np.array([0.0, 0.0, 0.22])

LEFT_LIFT_Z = 0.50
LEFT_TRANSFER_XY = np.array([0.50, 0.00], dtype=np.float64)

LEFT_POST_RELEASE_UP_DZ = 0.05
LEFT_HOME_TOL = 0.08

GRIP_STEP_HOLD = 30
SETTLE_STEPS = 30
END_HOLD_STEPS = 60
MAX_STEPS = 2000
PRINT_EVERY = 30

NUM_EPISODES = 30

LOG_DIR = "/home/june/bimanual_handover_IL/data/left_RMPFlow_right_joint_logged"
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


PHASE_NAMES = {
    0: "LEFT_APPROACH_PRE",
    1: "LEFT_APPROACH_GRASP",
    2: "LEFT_SETTLE",
    3: "LEFT_CLOSE",
    4: "LEFT_LIFT",
    5: "LEFT_MOVE_XY_TO_TRANSFER",
    6: "RIGHT_MOVE_TO_RECEIVE",
    7: "RIGHT_CLOSE",
    8: "LEFT_RELEASE",
    9: "LEFT_POST_RELEASE_UP",
    10: "LEFT_GO_HOME",
    11: "DONE",
}


# =========================
# One Episode
# =========================
def handover_episode(csv_path, cyl_spawn_pos):
    # ---------- robot recreate / initialize ----------
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

    # ---------- object reset ----------
    curr_pos, curr_ori = cyl.get_world_poses()
    cyl.set_world_poses(
        positions=np.array([cyl_spawn_pos], dtype=np.float64),
        orientations=curr_ori,
    )

    for _ in range(30):
        world.step(render=True)

    # ---------- right tcp target templates ----------
    # 화면에 안 보이게 target tcp만 읽고 바로 HOME으로 복원
    right_robot.set_joint_positions(RIGHT_WAIT_FULL)
    for _ in range(2):
        world.step(render=False)
    right_wait_tcp_target = get_pos(right_tcp)

    right_robot.set_joint_positions(RIGHT_RECEIVE_FULL)
    for _ in range(2):
        world.step(render=False)
    right_receive_tcp_target = get_pos(right_tcp)

    right_robot.set_joint_positions(RIGHT_HOME)
    for _ in range(2):
        world.step(render=False)

    # ---------- controllers ----------
    left_ctrl = RMPFlowController("left_ctrl", left_robot, physics_dt)

    # ---------- fixed targets ----------
    cyl_start_pos = get_pos(cyl)
    left_pre_target = cyl_start_pos + LEFT_PRE_OFFSET
    left_grasp_target = cyl_start_pos + LEFT_GRASP_OFFSET

    left_lift_target = None
    left_transfer_target = None
    left_post_release_up_target = None
    left_home_tcp_target = left_pre_target

    # ---------- phase vars ----------
    phase = 0
    sub_idx = 0
    sub_hold = 0
    settle_count = 0
    end_hold = 0

    left_target = left_pre_target
    left_grip_cmd = GRIP_OPEN
    right_grip_cmd = GRIP_OPEN

    left_grasped = 0.0
    right_grasped = 0.0

    left_arm_hold_q_close = None
    left_arm_hold_q_release = None

    # ---------- logger ----------
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step",
        "phase",
        "phase_name",
        "left_tcp_x", "left_tcp_y", "left_tcp_z",
        "right_tcp_x", "right_tcp_y", "right_tcp_z",
        "obj_x", "obj_y", "obj_z",
        "left_grasped",
        "right_grasped",
        "action_left_dx", "action_left_dy", "action_left_dz",
        "action_right_dx", "action_right_dy", "action_right_dz",
        "action_left_gripper",
        "action_right_gripper",
    ])

    print("\n[INFO] Episode start")
    print("csv_path                 =", csv_path)
    print("cyl_start_pos            =", np.round(cyl_start_pos, 4).tolist())
    print("right_wait_tcp_target    =", np.round(right_wait_tcp_target, 4).tolist())
    print("right_receive_tcp_target =", np.round(right_receive_tcp_target, 4).tolist())

    success = False

    try:
        for step in range(MAX_STEPS):
            left_tcp_pos = get_pos(left_tcp)
            right_tcp_pos = get_pos(right_tcp)
            obj_pos = get_pos(cyl)
            right_q = right_robot.get_joint_positions()[:6]

            # -------------------------
            # phase logic
            # -------------------------
            if phase == 0:
                left_target = left_pre_target
                left_grip_cmd = GRIP_OPEN
                right_grip_cmd = GRIP_OPEN

                if np.linalg.norm(left_tcp_pos - left_target) < 0.03:
                    phase = 1

            elif phase == 1:
                left_target = left_grasp_target
                left_grip_cmd = GRIP_OPEN
                right_grip_cmd = GRIP_OPEN

                if np.linalg.norm(left_tcp_pos - left_target) < 0.035:
                    phase = 2
                    settle_count = 0

            elif phase == 2:
                left_target = left_grasp_target
                left_grip_cmd = GRIP_OPEN
                right_grip_cmd = GRIP_OPEN
                settle_count += 1

                if settle_count >= SETTLE_STEPS:
                    phase = 3
                    sub_idx = 0
                    sub_hold = 0
                    left_arm_hold_q_close = left_robot.get_joint_positions().copy()

            elif phase == 3:
                left_target = left_grasp_target
                left_grip_cmd = LEFT_CLOSE_SEQ[sub_idx]
                right_grip_cmd = GRIP_OPEN

                sub_hold += 1
                if sub_hold >= GRIP_STEP_HOLD:
                    sub_hold = 0
                    sub_idx += 1

                    if sub_idx >= len(LEFT_CLOSE_SEQ):
                        left_grip_cmd = LEFT_GRIP_CLOSE
                        left_grasped = 1.0

                        left_lift_target = np.array(
                            [left_tcp_pos[0], left_tcp_pos[1], LEFT_LIFT_Z],
                            dtype=np.float64,
                        )

                        phase = 4

            elif phase == 4:
                left_target = left_lift_target
                left_grip_cmd = LEFT_GRIP_CLOSE
                right_grip_cmd = GRIP_OPEN

                if np.linalg.norm(left_tcp_pos - left_target) < 0.035:
                    dx = LEFT_TRANSFER_XY[0] - left_tcp_pos[0]
                    dy = LEFT_TRANSFER_XY[1] - left_tcp_pos[1]

                    left_transfer_target = np.array(
                        [left_tcp_pos[0] + dx, left_tcp_pos[1] + dy, left_tcp_pos[2]],
                        dtype=np.float64,
                    )

                    phase = 5

            elif phase == 5:
                left_target = left_transfer_target
                left_grip_cmd = LEFT_GRIP_CLOSE
                right_grip_cmd = GRIP_OPEN

                if np.linalg.norm(left_tcp_pos - left_target) < 0.04:
                    phase = 6

            elif phase == 6:
                left_target = left_transfer_target
                left_grip_cmd = LEFT_GRIP_CLOSE
                right_grip_cmd = GRIP_OPEN

                if np.all(np.abs(right_q - RIGHT_RECEIVE) < np.deg2rad(2.0)):
                    phase = 7
                    sub_idx = 0
                    sub_hold = 0

            elif phase == 7:
                left_target = left_transfer_target
                left_grip_cmd = LEFT_GRIP_CLOSE
                right_grip_cmd = RIGHT_CLOSE_SEQ[sub_idx]

                sub_hold += 1
                if sub_hold >= GRIP_STEP_HOLD:
                    sub_hold = 0
                    sub_idx += 1

                    if sub_idx >= len(RIGHT_CLOSE_SEQ):
                        right_grip_cmd = RIGHT_GRIP_CLOSE
                        right_grasped = 1.0
                        phase = 8
                        sub_idx = 0
                        sub_hold = 0
                        left_arm_hold_q_release = left_robot.get_joint_positions().copy()

            elif phase == 8:
                left_target = left_transfer_target
                right_grip_cmd = RIGHT_GRIP_CLOSE
                left_grip_cmd = LEFT_OPEN_SEQ[sub_idx]

                sub_hold += 1
                if sub_hold >= GRIP_STEP_HOLD:
                    sub_hold = 0
                    sub_idx += 1

                    if sub_idx >= len(LEFT_OPEN_SEQ):
                        left_grip_cmd = GRIP_OPEN
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
                left_target = left_post_release_up_target
                left_grip_cmd = GRIP_OPEN
                right_grip_cmd = RIGHT_GRIP_CLOSE

                if np.linalg.norm(left_tcp_pos - left_target) < 0.03:
                    phase = 10

            elif phase == 10:
                left_target = left_home_tcp_target
                left_grip_cmd = GRIP_OPEN
                right_grip_cmd = RIGHT_GRIP_CLOSE

                if np.linalg.norm(left_tcp_pos - left_target) < LEFT_HOME_TOL:
                    phase = 11
                    end_hold = 0

            elif phase == 11:
                left_target = left_home_tcp_target
                left_grip_cmd = GRIP_OPEN
                right_grip_cmd = RIGHT_GRIP_CLOSE

                end_hold += 1
                if end_hold >= END_HOLD_STEPS:
                    success = True
                    break

            # -------------------------
            # action label for logging
            # -------------------------
            action_left_delta = clip_delta(left_target, left_tcp_pos, max_step=0.02)

            if phase in [0, 1, 2, 3, 4, 5]:
                right_target_tcp = right_wait_tcp_target
            else:
                right_target_tcp = right_receive_tcp_target

            action_right_delta = clip_delta(right_target_tcp, right_tcp_pos, max_step=0.02)

            # -------------------------
            # left arm control
            # -------------------------
            if phase in [0, 1, 2, 4, 5, 6, 9, 10]:
                left_action = left_ctrl.forward(
                    target_end_effector_position=left_target,
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

            # -------------------------
            # right arm control
            # -------------------------
            if phase in [0, 1, 2, 3, 4, 5]:
                right_arm_target = RIGHT_WAIT
            else:
                right_arm_target = RIGHT_RECEIVE

            right_full = right_robot.get_joint_positions().copy()
            right_full[:6] = right_arm_target
            right_full[right_finger_idx] = right_grip_cmd
            right_robot.apply_action(ArticulationAction(joint_positions=right_full))

            world.step(render=True)

            # -------------------------
            # logging
            # -------------------------
            csv_writer.writerow([
                step,
                phase,
                PHASE_NAMES[phase],
                float(left_tcp_pos[0]), float(left_tcp_pos[1]), float(left_tcp_pos[2]),
                float(right_tcp_pos[0]), float(right_tcp_pos[1]), float(right_tcp_pos[2]),
                float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2]),
                float(left_grasped),
                float(right_grasped),
                float(action_left_delta[0]), float(action_left_delta[1]), float(action_left_delta[2]),
                float(action_right_delta[0]), float(action_right_delta[1]), float(action_right_delta[2]),
                float(left_grip_cmd),
                float(right_grip_cmd),
            ])

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