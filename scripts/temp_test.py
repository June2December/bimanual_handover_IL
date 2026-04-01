from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


usd_path = "/home/june/bimanul_ws/bimanual_scene.usd"
robot_path = "/World/ur_right"

omni.usd.get_context().open_stage(usd_path)
for _ in range(60):
    simulation_app.update()

world = World(stage_units_in_meters=1.0)
world.reset()

robot = SingleArticulation(prim_path=robot_path, name="right_robot")
robot.initialize()

print("DOF names:", robot.dof_names)
print("Num DOF:", robot.num_dof)

q = robot.get_joint_positions().copy()
dq = np.zeros_like(q)

# 팔 자세 고정
q[:6] = np.deg2rad([-45, -70, 100, -120, -90, 0])

robot.set_joints_default_state(positions=q, velocities=dq)
world.reset()
robot.initialize()

q = robot.get_joint_positions().copy()
q[:6] = np.deg2rad([-45, -70, 100, -120, -90, 0])
robot.set_joint_positions(q)

for _ in range(60):
    world.step(render=True)

finger_idx = robot.dof_names.index("finger_joint")
print("finger_idx:", finger_idx)

def step_with_target(full_q, steps=120):
    action = ArticulationAction(joint_positions=full_q)
    for _ in range(steps):
        robot.apply_action(action)
        world.step(render=True)

# 1. open
target = robot.get_joint_positions().copy()
target[finger_idx] = 0.0
print("OPEN target:", target[finger_idx])
step_with_target(target, steps=120)

# 2. close sweep
for val in [0.1, 0.2, 0.35, 0.5, 0.7]:
    target = robot.get_joint_positions().copy()
    target[:6] = np.deg2rad([-45, -70, 100, -120, -90, 0])  # 팔 자세 유지
    target[finger_idx] = val
    print("CLOSE target:", val)
    step_with_target(target, steps=120)

# 3. 다시 open
target = robot.get_joint_positions().copy()
target[:6] = np.deg2rad([-45, -70, 100, -120, -90, 0])
target[finger_idx] = 0.0
print("RE-OPEN target:", target[finger_idx])
step_with_target(target, steps=180)

simulation_app.close()