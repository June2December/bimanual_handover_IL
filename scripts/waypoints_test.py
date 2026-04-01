from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

# =========================
# 사용자 설정
# =========================
usd_path = "/home/june/bimanul_ws/bimanual_scene.usd"

left_robot_path = "/World/ur_left"
right_robot_path = "/World/ur_right"

# =========================
# joint 값
# =========================
LEFT_HOME_DEG  = [-45, -90,  90,  90,  90,  90]
RIGHT_HOME_DEG = [-135, -90, -90, 90, -90, 90]

RIGHT_WP1_DEG = [-135, -60, -120, 135, -90, 70]
RIGHT_WP2_DEG = [-125, -62, -130, 180, -90, 80]
RIGHT_WP3_DEG = [-121, -90, -140, 230, -90, 90]

def arm6_with_zeros6(deg6):
    return np.concatenate([np.deg2rad(deg6), np.zeros(6)])

LEFT_HOME  = arm6_with_zeros6(LEFT_HOME_DEG)
RIGHT_HOME = arm6_with_zeros6(RIGHT_HOME_DEG)

RIGHT_WAYPOINTS = [
    ("RIGHT_HOME", arm6_with_zeros6(RIGHT_HOME_DEG)),
    ("RIGHT_WP1",  arm6_with_zeros6(RIGHT_WP1_DEG)),
    ("RIGHT_WP2",  arm6_with_zeros6(RIGHT_WP2_DEG)),
    ("RIGHT_WP3",  arm6_with_zeros6(RIGHT_WP3_DEG)),
    ("RIGHT_BACK_HOME", arm6_with_zeros6(RIGHT_HOME_DEG)),
]

# =========================
# stage 열기
# =========================
omni.usd.get_context().open_stage(usd_path)
for _ in range(60):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Stage open failed")

# prim 존재 확인
left_prim = stage.GetPrimAtPath(left_robot_path)
right_prim = stage.GetPrimAtPath(right_robot_path)

print("left prim valid :", left_prim.IsValid())
print("right prim valid:", right_prim.IsValid())

if not left_prim.IsValid():
    raise RuntimeError(f"Invalid left robot path: {left_robot_path}")
if not right_prim.IsValid():
    raise RuntimeError(f"Invalid right robot path: {right_robot_path}")

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

left_robot.set_joints_default_state(LEFT_HOME, np.zeros_like(LEFT_HOME))
right_robot.set_joints_default_state(RIGHT_HOME, np.zeros_like(RIGHT_HOME))

world.reset()

left_robot.initialize()
right_robot.initialize()

left_robot.set_joint_positions(LEFT_HOME)
right_robot.set_joint_positions(RIGHT_HOME)

for _ in range(30):
    world.step(render=True)

print("\n[RIGHT DOF NAMES]")
for i, n in enumerate(right_robot.dof_names):
    print(i, n)

# =========================
# helper
# =========================
def reached(curr, target, tol_deg=2.0):
    tol = np.deg2rad(tol_deg)
    err = np.abs(curr[:6] - target[:6])
    return np.all(err < tol), np.rad2deg(err)

# =========================
# waypoint test
# =========================
phase = 0
hold_count = 0
HOLD_STEPS = 45

while simulation_app.is_running():
    name, target = RIGHT_WAYPOINTS[phase]

    # 왼팔은 home 고정
    left_robot.apply_action(ArticulationAction(joint_positions=LEFT_HOME))

    # 오른팔 현재 waypoint로 이동
    right_robot.apply_action(ArticulationAction(joint_positions=target))

    world.step(render=True)

    curr = right_robot.get_joint_positions()
    ok, err_deg = reached(curr, target)

    if ok:
        if hold_count == 0:
            print(f"\n[REACHED] {name}")
            print("target(deg):", np.round(np.rad2deg(target[:6]), 2).tolist())
            print("curr(deg):  ", np.round(np.rad2deg(curr[:6]), 2).tolist())

        hold_count += 1

        if hold_count >= HOLD_STEPS:
            phase += 1
            hold_count = 0

            if phase >= len(RIGHT_WAYPOINTS):
                print("\n[INFO] waypoint test finished")
                break

            print(f"\n[MOVE NEXT] {RIGHT_WAYPOINTS[phase][0]}")
    else:
        hold_count = 0

simulation_app.close()