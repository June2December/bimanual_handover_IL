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


def get_pos(xform_prim):
    positions, _ = xform_prim.get_world_poses()
    return np.array(positions[0], dtype=np.float64)

# =========================
# 사용자 설정
# =========================
usd_path = "/home/june/bimanul_ws/bimanual_scene.usd"

left_robot_path = "/World/ur_left"
right_robot_path = "/World/ur_right"

left_tcp_path = "/World/ur_left/wrist_3_link/flange/tool0"
right_tcp_path = "/World/ur_right/wrist_3_link/flange/tool0"

cylinder_path = "/World/Cylinder"

physics_dt = 1.0 / 60.0

# ===== 초기 자세 =====
left_q = np.concatenate([
    np.deg2rad([45, -90, 90, 90, 90, 90]),  # arm
    np.zeros(6)                             # gripper
])

right_q = np.concatenate([
    np.deg2rad([-45, -90, 90, 90, 90, 90]),
    np.zeros(6)
])

left_dq = np.zeros_like(left_q)
right_dq = np.zeros_like(right_q)

target_offset = np.array([0.0, 0.0, 0.3])
target_orientation = euler_angles_to_quat(np.array([0.0, np.pi / 2, 0.0]))

# =========================
# scene 열기
# =========================
omni.usd.get_context().open_stage(usd_path)
for _ in range(60):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Stage open failed")

# =========================
# world
# =========================
world = World(stage_units_in_meters=1.0)
world.reset()

# =========================
# robots
# =========================
left_robot = SingleArticulation(left_robot_path, name="left_robot")
right_robot = SingleArticulation(right_robot_path, name="right_robot")

left_robot.initialize()
right_robot.initialize()

# 초기 상태 고정
left_robot.set_joints_default_state(left_q, left_dq)
right_robot.set_joints_default_state(right_q, right_dq)

world.reset()

left_robot.initialize()
right_robot.initialize()

# 6 index 가 gripper 용
finger_idx = left_robot.dof_names.index("finger_joint")
gripper_open = 0.0
gripper_close = 0.7
gripper_cmd = gripper_open

left_robot.set_joint_positions(left_q)
right_robot.set_joint_positions(right_q)

for _ in range(30):
    world.step(render=True)

# =========================
# TCP / object
# =========================
left_tcp = XFormPrim(left_tcp_path)
right_tcp = XFormPrim(right_tcp_path)
cyl = XFormPrim(cylinder_path)

left_tcp.initialize()
right_tcp.initialize()
cyl.initialize()

# =========================
# RMPFlow controllers
# =========================
left_ctrl = RMPFlowController("left_ctrl", left_robot, physics_dt)
right_ctrl = RMPFlowController("right_ctrl", right_robot, physics_dt)

# =========================
# target
# =========================
cyl_pos = get_pos(cyl)
target_pos = cyl_pos + target_offset

# =========================
# loop
# =========================
phase = 0
for step in range(1000):
    cyl_pos = get_pos(cyl)

    if phase == 0:
        target = cyl_pos + np.array([0, 0, 0.3])

        if np.linalg.norm(get_pos(left_tcp) - target) < 0.03:
            phase = 1

    elif phase == 1:
        target = cyl_pos + np.array([0, 0, 0.2])

        if np.linalg.norm(get_pos(left_tcp) - target) < 0.035:
            phase = 2

    elif phase == 2:
        # settle
        for _ in range(40):
            left_action = left_ctrl.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=target_orientation,
            )
            left_robot.apply_action(left_action)
            world.step(render=True)

        phase = 3
        continue

    elif phase == 3:
        # 천천히 close
        for grip in [0.1, 0.2, 0.3, 0.4]:
            gripper_target = left_robot.get_joint_positions().copy()
            gripper_target[finger_idx] = grip

            close_action = ArticulationAction(joint_positions=gripper_target)

            for _ in range(20):
                left_robot.apply_action(close_action)
                world.step(render=True)

        phase = 4
        continue

    elif phase == 4:
        # hold
        gripper_target = left_robot.get_joint_positions().copy()
        gripper_target[finger_idx] = 0.4
        hold_action = ArticulationAction(joint_positions=gripper_target)

        for _ in range(30):
            left_robot.apply_action(hold_action)
            world.step(render=True)

        phase = 5
        end_target = cyl_pos + np.array([0, 0, 0.40])
        continue

    elif phase == 5:
        # target = cyl_pos + np.array([0, 0, 0.30])
        target = end_target

    left_action = left_ctrl.forward(
        target_end_effector_position=target,
        target_end_effector_orientation=target_orientation,
    )

    left_robot.apply_action(left_action)

    world.step(render=True)

    if step % 30 == 0:
        print(f"[{step}]")
        print(" left tcp :", get_pos(left_tcp))
        print(" right tcp:", get_pos(right_tcp))
        print(" phase:", phase)


for _ in range(100):
    world.step(render=True)

simulation_app.close()