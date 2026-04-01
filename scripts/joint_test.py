from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
import omni.usd
import numpy as np

usd_path = "/home/june/bimanul_ws/bimanual_scene.usd"
robot_path = "/World/ur_right"
tcp_path = "/World/ur_right/wrist_3_link/flange/tool0"

# 1. stage 열기
omni.usd.get_context().open_stage(usd_path)
for _ in range(50):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError(f"Stage failed to open: {usd_path}")

# 2. world 생성
world = World(stage_units_in_meters=1.0)
world.reset()

# 3. robot articulation 연결
robot = SingleArticulation(prim_path=robot_path, name="ur_right_robot")
robot.initialize()

# 4. tcp prim 연결
tcp_prim = XFormPrim(prim_paths_expr=tcp_path, name="ur_right_tcp")
tcp_prim.initialize()

# 5. 현재 tcp 좌표 읽기
tcp_before_all, _ = tcp_prim.get_world_poses()
tcp_before = np.array(tcp_before_all[0], dtype=float)
print("TCP before:", tcp_before)

# 6. 현재 joint 읽기
joint_positions = robot.get_joint_positions()
print("Current joints:", joint_positions)

# 7. joint 하나만 조금 수정
target_joints = np.array(joint_positions, dtype=float)
target_joints[0] += 0.1   # 약 5.7도

print("Target joints:", target_joints)

# 8. 적용
for _ in range(200):
    robot.set_joint_positions(target_joints)
    world.step(render=True)

# 9. 적용 후 tcp 좌표 읽기
tcp_after_all, _ = tcp_prim.get_world_poses()
tcp_after = np.array(tcp_after_all[0], dtype=float)
print("TCP after:", tcp_after)
print("TCP delta:", tcp_after - tcp_before)

simulation_app.close()