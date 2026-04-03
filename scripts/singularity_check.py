from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.usd
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


# =========================================================
# User setting
# =========================================================
USD_PATH = "/home/june/bimanual_handover_IL/bimanual_scene.usd"

# 분석할 로봇 하나만 선택
ROBOT_PRIM_PATH = "/World/ur_right"

# body_names 출력 후 실제 이름에 맞게 수정
# 예: "tool0", "flange", "ee_link" 등
EE_LINK_NAME = "tool0"

# UR arm 6축 이름
ARM_DOF_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# 테스트할 자세들 [deg]
TEST_POSES_DEG = [
    [-135, -90, -90,  90, -90,  90],   # 지금 세팅한 자세
    [-135, -90, -90,  90, -10,  90],
    [-135, -90, -90,  90,  -1,  90],
    [-135, -90, -90,  90,   0,  90],   # wrist singularity 후보
    [-135, -90, -90,  90,   1,  90],
    [-135, -90, -90,  90,  10,  90],
]

SETTLE_STEPS = 30
EPS = 1e-12


# =========================================================
# Helper
# =========================================================
def deg_to_rad_list(values_deg):
    return np.deg2rad(np.array(values_deg, dtype=np.float64))


def print_array(name, arr, precision=6):
    print(f"{name} = {np.array2string(np.asarray(arr), precision=precision, suppress_small=False)}")


def compute_metrics_from_jacobian(J):
    """
    J: (6, 6) expected for UR10 arm ee Jacobian
    """
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    sigma_min = float(np.min(S))
    sigma_max = float(np.max(S))
    cond = float(sigma_max / sigma_min) if sigma_min > EPS else np.inf

    JJT = J @ J.T
    det_val = float(np.linalg.det(JJT))
    manip = float(np.sqrt(max(det_val, 0.0)))

    return {
        "singular_values": S,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "condition_number": cond,
        "manipulability": manip,
    }


def get_arm_joint_indices(robot, arm_dof_names):
    indices = []
    for name in arm_dof_names:
        idx = robot.get_joint_index(name)
        indices.append(idx)
    return np.array(indices, dtype=np.int64)


def set_robot_arm_pose(robot, q_arm_rad, arm_joint_indices):
    """
    arm 6축만 원하는 값으로 넣고, 나머지 DOF는 현재값 유지
    """
    q_full = robot.get_joint_positions().copy()
    q_full[arm_joint_indices] = q_arm_rad
    robot.apply_action(ArticulationAction(joint_positions=q_full))


def get_ee_jacobian_6x6(robot, ee_link_name, arm_joint_indices):
    """
    fixed-base articulation 기준:
      jacobian shape = (num_bodies - 1, 6, num_dof)
    따라서 link_index에서 root body(0)를 제외한 jacobian row를 쓰기 위해 -1 필요
    """
    # body_names 확인용
    body_names = list(robot.body_names)

    if ee_link_name not in body_names:
        raise ValueError(
            f"EE_LINK_NAME '{ee_link_name}' not found.\n"
            f"Available body_names:\n{body_names}"
        )

    link_index = robot.get_link_index(ee_link_name)
    if link_index <= 0:
        raise ValueError(
            f"EE link '{ee_link_name}' has link_index={link_index}. "
            f"Fixed-base Jacobian row uses non-root links only, so root link은 사용할 수 없습니다."
        )

    jac_row = link_index - 1

    # SingleArticulation 내부 articulation view에서 jacobian 획득
    # shape: (num_articulations=1, num_bodies-1, 6, num_dof)
    jac_all = robot._articulation_view.get_jacobians()

    if hasattr(jac_all, "cpu"):  # torch tensor 대응
        jac_all = jac_all.cpu().numpy()
    else:
        jac_all = np.asarray(jac_all)

    # 첫 articulation 하나만 사용
    J_full = jac_all[0, jac_row, :, :]   # shape: (6, num_dof)

    # arm 6축만 사용
    J_arm = J_full[:, arm_joint_indices]  # shape: (6, 6)

    return J_arm, {
        "body_names": body_names,
        "link_index": int(link_index),
        "jacobian_row_index": int(jac_row),
    }


# =========================================================
# Open Stage
# =========================================================
omni.usd.get_context().open_stage(USD_PATH)
for _ in range(60):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Stage open failed")

if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
    raise RuntimeError(f"Invalid robot prim path: {ROBOT_PRIM_PATH}")


# =========================================================
# World / Robot
# =========================================================
world = World(stage_units_in_meters=1.0)
world.reset()

robot = SingleArticulation(ROBOT_PRIM_PATH, name="target_robot")
robot.initialize()

world.reset()
robot.initialize()

print("\n[INFO] Robot initialized")
print("robot prim path =", ROBOT_PRIM_PATH)
print("num_dof         =", robot.num_dof)
print("dof_names       =", list(robot.dof_names))
print("body_names      =", list(robot.body_names))

arm_joint_indices = get_arm_joint_indices(robot, ARM_DOF_NAMES)
print("arm_joint_indices =", arm_joint_indices.tolist())

for i, name in enumerate(ARM_DOF_NAMES):
    print(f"  {name:20s} -> {int(arm_joint_indices[i])}")

print(f"\n[INFO] EE_LINK_NAME = {EE_LINK_NAME}")


# =========================================================
# Main test loop
# =========================================================
for pose_deg in TEST_POSES_DEG:
    print("\n" + "=" * 80)
    print(f"[TEST] pose_deg = {pose_deg}")

    q_arm_rad = deg_to_rad_list(pose_deg)
    set_robot_arm_pose(robot, q_arm_rad, arm_joint_indices)

    for _ in range(SETTLE_STEPS):
        world.step(render=True)

    q_now = robot.get_joint_positions()
    q_arm_now_deg = np.rad2deg(q_now[arm_joint_indices])

    print_array("q_arm_now_deg", q_arm_now_deg, precision=3)

    try:
        J, info = get_ee_jacobian_6x6(robot, EE_LINK_NAME, arm_joint_indices)
    except Exception as e:
        print("\n[ERROR] Jacobian extraction failed")
        print(str(e))
        break

    metrics = compute_metrics_from_jacobian(J)

    print(f"ee_link_index        = {info['link_index']}")
    print(f"jacobian_row_index   = {info['jacobian_row_index']}")
    print(f"J shape              = {J.shape}")

    print_array("J", J, precision=5)
    print_array("singular_values", metrics["singular_values"], precision=8)
    print(f"sigma_min            = {metrics['sigma_min']:.10e}")
    print(f"sigma_max            = {metrics['sigma_max']:.10e}")
    print(f"condition_number     = {metrics['condition_number']:.10e}")
    print(f"manipulability       = {metrics['manipulability']:.10e}")

    if metrics["sigma_min"] < 1e-3:
        print("[WARN] sigma_min is very small -> singularity very near")
    elif metrics["sigma_min"] < 1e-2:
        print("[WARN] sigma_min is small -> singularity near")
    else:
        print("[INFO] sigma_min looks normal")

print("\n[INFO] Finished singularity check")

for _ in range(120):
    world.step(render=True)

simulation_app.close()