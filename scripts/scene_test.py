# 아이작심 파이썬에서 실행
# headless = False 는 창 띄운다는 뜻
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
from pxr import UsdGeom

# 1. USD 경로
usd_path = "/home/june/bimanul_ws/bimanual_scene.usd"

# 2. stage 열기
omni.usd.get_context().open_stage(usd_path)

# 3. 로딩 기다리기
for _ in range(50):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()

# 4. prim path
cylinder_path = "/World/Cylinder"
ee_path = "/World/ur_left/ee_link"

cylinder_prim = stage.GetPrimAtPath(cylinder_path)
ee_prim = stage.GetPrimAtPath(ee_path)

if not cylinder_prim.IsValid():
    raise ValueError("Cylinder path wrong")
if not ee_prim.IsValid():
    raise ValueError("EE path wrong")

# 5. world 좌표 계산
cyl_xform = UsdGeom.Xformable(cylinder_prim)
ee_xform = UsdGeom.Xformable(ee_prim)

cyl_world = cyl_xform.ComputeLocalToWorldTransform(0)
ee_world = ee_xform.ComputeLocalToWorldTransform(0)

cyl_pos = cyl_world.ExtractTranslation()
ee_pos = ee_world.ExtractTranslation()

print("Cylinder:", cyl_pos)
print("EE:", ee_pos)

dx = cyl_pos[0] - ee_pos[0]
dy = cyl_pos[1] - ee_pos[1]
dz = cyl_pos[2] - ee_pos[2]

print("Delta:", (dx, dy, dz))

# 6. 물체 집을때 완충지점 확인하자
target_pos = (
    cyl_pos[0],
    cyl_pos[1],
    cyl_pos[2] + 0.1
)
print("Target above object:", target_pos)

# 결과 확인 시간
for _ in range(100):
    simulation_app.update()

simulation_app.close()
