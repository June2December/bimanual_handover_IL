from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
import numpy as np
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


def apply_left_rmp_with_grip(robot, ctrl, target_pos, target_ori, finger_idx, grip_val):
    action = ctrl.forward(
        target_end_effector_position=target_pos,
        target_end_effector_orientation=target_ori,
    )

    curr = robot.get_joint_positions().copy()
    full = curr.copy()

    if action.joint_positions is not None:
        jp = np.array(action.joint_positions, dtype=np.float64)
        full[:len(jp)] = jp

    full[finger_idx] = grip_val
    robot.apply_action(ArticulationAction(joint_positions=full))


# =========================
# User setting
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
# [-121, -100, -138, 230, -90, 90]
# [-121, -100, -138, 238, -90, 90]
# [-121, -96, -140, 240, -90, 90]
# [-121, -100, -140, 240, -90, 90]

LEFT_HOME = full_q_from_arm6(LEFT_HOME_DEG)
RIGHT_HOME = full_q_from_arm6(RIGHT_HOME_DEG)
RIGHT_WAIT = arm6_deg_to_rad(RIGHT_WAIT_DEG)
RIGHT_RECEIVE = arm6_deg_to_rad(RIGHT_RECEIVE_DEG)

LEFT_TOP_ORI = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))

GRIP_OPEN = 0.0
LEFT_GRIP_CLOSE = 0.45
RIGHT_GRIP_CLOSE = 0.45

LEFT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.45]
RIGHT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.45]
LEFT_OPEN_SEQ = [0.4, 0.25, 0.1, 0.0]

LEFT_PRE_OFFSET = np.array([0.0, 0.0, 0.50])
LEFT_GRASP_OFFSET = np.array([0.0, 0.0, 0.20])

LEFT_LIFT_Z = 0.50
LEFT_TRANSFER_XY = np.array([0.50, 0.00], dtype=np.float64)

LEFT_POST_RELEASE_UP_DZ = 0.05
LEFT_HOME_TOL = 0.08

GRIP_STEP_HOLD = 20
SETTLE_STEPS = 20
END_HOLD_STEPS = 60
MAX_STEPS = 2500
PRINT_EVERY = 30


# =========================
# Open scene
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
# World
# =========================
world = World(stage_units_in_meters=1.0)
world.reset()


# =========================
# Robots
# =========================
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

for _ in range(30):
    world.step(render=True)


# =========================
# TCP / Object
# =========================
left_tcp = XFormPrim(left_tcp_path)
right_tcp = XFormPrim(right_tcp_path)
cyl = XFormPrim(cylinder_path)

left_tcp.initialize()
right_tcp.initialize()
cyl.initialize()


# =========================
# Controller
# =========================
left_ctrl = RMPFlowController("left_ctrl", left_robot, physics_dt)


# =========================
# Fixed targets: 시작 시 1회만 계산
# =========================
cyl_start_pos = get_pos(cyl)

left_pre_target = cyl_start_pos + LEFT_PRE_OFFSET
left_grasp_target = cyl_start_pos + LEFT_GRASP_OFFSET

left_lift_target = None
left_transfer_target = None
left_post_release_up_target = None
left_home_tcp_target = left_pre_target

print("\n[INFO] Fixed targets")
print("cyl_start_pos      =", np.round(cyl_start_pos, 4).tolist())
print("left_pre_target    =", np.round(left_pre_target, 4).tolist())
print("left_grasp_target  =", np.round(left_grasp_target, 4).tolist())
print("left_home_deg      =", LEFT_HOME_DEG)
print("right_home_deg     =", RIGHT_HOME_DEG)
print("right_wait_deg     =", RIGHT_WAIT_DEG)
print("right_receive_deg  =", RIGHT_RECEIVE_DEG)


# =========================
# Phase
# =========================
PHASE_NAMES = {
    0: "LEFT_APPROACH",
    1: "LEFT_GRASP",
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

phase = 0
sub_idx = 0
sub_hold = 0
settle_count = 0
end_hold = 0

left_target = left_pre_target
left_grip_cmd = GRIP_OPEN
right_grip_cmd = GRIP_OPEN

print("\n[INFO] Start staged handover test")
print("left finger idx  =", left_finger_idx)
print("right finger idx =", right_finger_idx)


# =========================
# Main loop
# =========================
for step in range(MAX_STEPS):
    left_tcp_pos = get_pos(left_tcp)
    right_q = right_robot.get_joint_positions()[:6]

    # -------------------------
    # phase logic
    # -------------------------
    if phase == 0:
        left_target = left_pre_target

        if np.linalg.norm(left_tcp_pos - left_target) < 0.03:
            phase = 1
            print(f"\n[PHASE 1] {PHASE_NAMES[phase]}")
            print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())
            print("left_target       =", np.round(left_target, 4).tolist())

    elif phase == 1:
        left_target = left_grasp_target

        if np.linalg.norm(left_tcp_pos - left_target) < 0.035:
            phase = 2
            settle_count = 0
            print(f"\n[PHASE 2] {PHASE_NAMES[phase]}")
            print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())
            print("left_target       =", np.round(left_target, 4).tolist())

    elif phase == 2:
        left_target = left_grasp_target
        settle_count += 1

        if settle_count >= SETTLE_STEPS:
            phase = 3
            sub_idx = 0
            sub_hold = 0
            print(f"\n[PHASE 3] {PHASE_NAMES[phase]}")
            print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())
            print("left_target       =", np.round(left_target, 4).tolist())

    elif phase == 3:
        left_target = left_grasp_target
        left_grip_cmd = LEFT_CLOSE_SEQ[sub_idx]

        sub_hold += 1
        if sub_hold >= GRIP_STEP_HOLD:
            sub_hold = 0
            sub_idx += 1

            if sub_idx >= len(LEFT_CLOSE_SEQ):
                left_grip_cmd = LEFT_GRIP_CLOSE

                # 상대이동 1회 생성: 현재 TCP 기준, z만 0.5로
                left_lift_target = np.array(
                    [left_tcp_pos[0], left_tcp_pos[1], LEFT_LIFT_Z],
                    dtype=np.float64,
                )

                phase = 4
                print(f"\n[PHASE 4] {PHASE_NAMES[phase]}")
                print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())
                print("left_lift_target  =", np.round(left_lift_target, 4).tolist())

    elif phase == 4:
        left_target = left_lift_target
        left_grip_cmd = LEFT_GRIP_CLOSE

        if np.linalg.norm(left_tcp_pos - left_target) < 0.035:
            # 상대이동 1회 생성: 현재 TCP 기준, XY만 (0.50, 0.00) 쪽으로
            dx = LEFT_TRANSFER_XY[0] - left_tcp_pos[0]
            dy = LEFT_TRANSFER_XY[1] - left_tcp_pos[1]

            left_transfer_target = np.array(
                [left_tcp_pos[0] + dx, left_tcp_pos[1] + dy, left_tcp_pos[2]],
                dtype=np.float64,
            )

            phase = 5
            print(f"\n[PHASE 5] {PHASE_NAMES[phase]}")
            print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())
            print("left_transfer_tgt =", np.round(left_transfer_target, 4).tolist())

    elif phase == 5:
        left_target = left_transfer_target
        left_grip_cmd = LEFT_GRIP_CLOSE

        if np.linalg.norm(left_tcp_pos - left_target) < 0.04:
            phase = 6
            print(f"\n[PHASE 6] {PHASE_NAMES[phase]}")
            print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())
            print("left_target       =", np.round(left_target, 4).tolist())

    elif phase == 6:
        left_target = left_transfer_target
        left_grip_cmd = LEFT_GRIP_CLOSE

        if np.all(np.abs(right_q - RIGHT_RECEIVE) < np.deg2rad(2.0)):
            phase = 7
            sub_idx = 0
            sub_hold = 0
            print(f"\n[PHASE 7] {PHASE_NAMES[phase]}")
            print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())

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
                phase = 8
                sub_idx = 0
                sub_hold = 0
                print(f"\n[PHASE 8] {PHASE_NAMES[phase]}")
                print("right_grip_cmd    =", right_grip_cmd)

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

                left_post_release_up_target = np.array(
                    [
                        left_tcp_pos[0],
                        left_tcp_pos[1],
                        left_tcp_pos[2] + LEFT_POST_RELEASE_UP_DZ,
                    ],
                    dtype=np.float64,
                )

                phase = 9
                print(f"\n[PHASE 9] {PHASE_NAMES[phase]}")
                print("left_grip_cmd     =", left_grip_cmd)
                print("left_post_up_tgt  =", np.round(left_post_release_up_target, 4).tolist())

    elif phase == 9:
        left_target = left_post_release_up_target
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE

        if np.linalg.norm(left_tcp_pos - left_target) < 0.03:
            phase = 10
            print(f"\n[PHASE 10] {PHASE_NAMES[phase]}")
            print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())
            print("left_home_tcp_tgt =", np.round(left_home_tcp_target, 4).tolist())

    elif phase == 10:
        left_target = left_home_tcp_target
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE

        if np.linalg.norm(left_tcp_pos - left_target) < LEFT_HOME_TOL:
            phase = 11
            end_hold = 0
            print(f"\n[PHASE 11] {PHASE_NAMES[phase]}")
            print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())

    elif phase == 11:
        left_target = left_home_tcp_target
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE

        end_hold += 1
        if end_hold >= END_HOLD_STEPS:
            print("\n[END] staged handover finished")
            print("final_left_tcp    =", np.round(get_pos(left_tcp), 4).tolist())
            print("final_right_tcp   =", np.round(get_pos(right_tcp), 4).tolist())
            print("final_cylinder    =", np.round(get_pos(cyl), 4).tolist())
            break

    # -------------------------
    # left arm
    # phase 3(close)만 RMPFlow + grip
    # 나머지 move phase는 raw RMPFlow
    # -------------------------
    if phase in [0, 1, 2, 4, 5, 6, 9, 10]:
        left_action = left_ctrl.forward(
            target_end_effector_position=left_target,
            target_end_effector_orientation=LEFT_TOP_ORI,
        )
        left_robot.apply_action(left_action)

    elif phase == 3:
        apply_left_rmp_with_grip(
            robot=left_robot,
            ctrl=left_ctrl,
            target_pos=left_target,
            target_ori=LEFT_TOP_ORI,
            finger_idx=left_finger_idx,
            grip_val=left_grip_cmd,
        )

    else:
        left_full = left_robot.get_joint_positions().copy()
        left_full[left_finger_idx] = left_grip_cmd
        left_robot.apply_action(ArticulationAction(joint_positions=left_full))

    # -------------------------
    # right arm
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

    if step % PRINT_EVERY == 0:
        print(f"\n[STEP {step}] phase={phase} ({PHASE_NAMES[phase]})")
        print("left_tcp_pos      =", np.round(left_tcp_pos, 4).tolist())
        print("left_target       =", np.round(left_target, 4).tolist())
        print("left_grip_cmd     =", float(left_grip_cmd))
        print("right_grip_cmd    =", float(right_grip_cmd))
        if left_lift_target is not None:
            print("left_lift_target  =", np.round(left_lift_target, 4).tolist())
        if left_transfer_target is not None:
            print("left_transfer_tgt =", np.round(left_transfer_target, 4).tolist())
        if left_post_release_up_target is not None:
            print("left_post_up_tgt  =", np.round(left_post_release_up_target, 4).tolist())


for _ in range(120):
    world.step(render=True)

simulation_app.close()