from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import csv
import omni.usd
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction


# =========================================================
# 사용자 설정
# =========================================================
USD_PATH = "/home/june/bimanual_handover_IL/poly_scene.usd"   # 실제 경로 확인
ROBOT_PRIM_PATH = "/World/ur_right"
EE_LINK_NAME = "wrist_3_link"

ARM_DOF_NAMES = [
    "shoulder_pan_joint",   # j1
    "shoulder_lift_joint",  # j2
    "elbow_joint",          # j3
    "wrist_1_joint",        # j4
    "wrist_2_joint",        # j5
    "wrist_3_joint",        # j6
]

# 시작 자세
START_POSE_DEG = [-90, -90, -90, 90, -90, 90]

# ---------------------------------------------------------
# 2D sweep: j4 + j5
# 시작자세 기준 delta
# j4:  90 -> 150
# j5: -90 ->  10
# ---------------------------------------------------------
J4_NUMBER = 4
J5_NUMBER = 5

J4_DELTA_DEG = list(range(0, 61, 10))     # 0,10,...,60
J5_DELTA_DEG = list(range(0, 101, 10))    # 0,10,...,100

SETTLE_STEPS = 30
EPS = 1e-12

CSV_DIR = "/home/june/bimanual_handover_IL/data/"
CSV_PATH = os.path.join(CSV_DIR, "singularity_j4_j5.csv")


# =========================================================
# Helper
# =========================================================
def deg_to_rad(values_deg):
    return np.deg2rad(np.array(values_deg, dtype=np.float64))


def rad_to_deg(values_rad):
    return np.rad2deg(np.array(values_rad, dtype=np.float64))


def joint_number_to_name(joint_number):
    if joint_number < 1 or joint_number > 6:
        raise ValueError("joint_number must be 1~6")
    return ARM_DOF_NAMES[joint_number - 1]


def get_arm_joint_indices(robot):
    return np.array([robot.get_dof_index(name) for name in ARM_DOF_NAMES], dtype=np.int64)


def set_arm_pose(robot, arm_joint_indices, q_arm_rad):
    q_full = robot.get_joint_positions().copy()
    q_full[arm_joint_indices] = q_arm_rad
    robot.apply_action(ArticulationAction(joint_positions=q_full))


def get_jacobian_6x6(robot, ee_link_name, arm_joint_indices):
    view = robot._articulation_view
    body_names = list(view.body_names)

    if ee_link_name not in body_names:
        raise ValueError(
            f"EE_LINK_NAME '{ee_link_name}' not found.\n"
            f"Available body_names: {body_names}"
        )

    link_index = view.get_link_index(ee_link_name)
    if link_index <= 0:
        raise ValueError(f"Invalid EE link index for {ee_link_name}: {link_index}")

    jac_row = link_index - 1

    jac_all = view.get_jacobians()
    if hasattr(jac_all, "cpu"):
        jac_all = jac_all.cpu().numpy()
    else:
        jac_all = np.asarray(jac_all)

    # expected shape: (1, num_bodies-1, 6, num_dof)
    J_full = jac_all[0, jac_row, :, :]           # (6, num_dof)
    J_arm = J_full[:, arm_joint_indices]         # (6, 6)
    return J_arm


def compute_metrics(J):
    _, S, _ = np.linalg.svd(J, full_matrices=False)

    sigma_min = float(np.min(S))
    sigma_max = float(np.max(S))
    cond = float(sigma_max / sigma_min) if sigma_min > EPS else np.inf

    det_val = float(np.linalg.det(J @ J.T))
    manip = float(np.sqrt(max(det_val, 0.0)))

    return S, sigma_min, sigma_max, cond, manip


# =========================================================
# 경로 확인
# =========================================================
print("USD exists:", os.path.exists(USD_PATH), USD_PATH)
if not os.path.exists(USD_PATH):
    raise FileNotFoundError(f"USD not found: {USD_PATH}")

os.makedirs(CSV_DIR, exist_ok=True)

J4_NAME = joint_number_to_name(J4_NUMBER)
J5_NAME = joint_number_to_name(J5_NUMBER)


# =========================================================
# Stage 열기
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
# World / Robot 초기화
# =========================================================
world = World(stage_units_in_meters=1.0)
world.reset()

robot = SingleArticulation(ROBOT_PRIM_PATH, name="target_robot")
robot.initialize()

world.reset()
robot.initialize()

view = robot._articulation_view
arm_joint_indices = get_arm_joint_indices(robot)

j4_local_idx = J4_NUMBER - 1
j5_local_idx = J5_NUMBER - 1

print("\n[INFO] Robot initialized")
print("robot prim path   =", ROBOT_PRIM_PATH)
print("dof_names         =", list(robot.dof_names))
print("body_names        =", list(view.body_names))
print("arm_joint_indices =", arm_joint_indices.tolist())
print("start_pose_deg    =", START_POSE_DEG)
print("j4_name           =", J4_NAME)
print("j5_name           =", J5_NAME)
print("j4_delta_deg      =", J4_DELTA_DEG)
print("j5_delta_deg      =", J5_DELTA_DEG)
print("csv_path          =", CSV_PATH)


# =========================================================
# 시작 자세 적용
# =========================================================
start_pose_rad = deg_to_rad(START_POSE_DEG)
set_arm_pose(robot, arm_joint_indices, start_pose_rad)

for _ in range(SETTLE_STEPS):
    world.step(render=True)


# =========================================================
# 2D Sweep 실행
# =========================================================
rows = []

base_j4_deg = START_POSE_DEG[j4_local_idx]
base_j5_deg = START_POSE_DEG[j5_local_idx]

for j4_delta in J4_DELTA_DEG:
    for j5_delta in J5_DELTA_DEG:
        target_j4_deg = base_j4_deg + j4_delta
        target_j5_deg = base_j5_deg + j5_delta

        pose_deg = START_POSE_DEG.copy()
        pose_deg[j4_local_idx] = target_j4_deg
        pose_deg[j5_local_idx] = target_j5_deg
        pose_rad = deg_to_rad(pose_deg)

        set_arm_pose(robot, arm_joint_indices, pose_rad)

        for _ in range(SETTLE_STEPS):
            world.step(render=True)

        q_now = robot.get_joint_positions()
        q_arm_now_deg = rad_to_deg(q_now[arm_joint_indices])

        J = get_jacobian_6x6(robot, EE_LINK_NAME, arm_joint_indices)
        S, sigma_min, sigma_max, cond, manip = compute_metrics(J)

        print(
            f"[SWEEP j4+j5] "
            f"j4={target_j4_deg:>7.2f} deg | "
            f"j5={target_j5_deg:>7.2f} deg | "
            f"sigma_min={sigma_min:.6e} | "
            f"cond={cond:.6e} | "
            f"manip={manip:.6e}"
        )

        rows.append({
            "sweep_joint_1_number": J4_NUMBER,
            "sweep_joint_1_name": J4_NAME,
            "sweep_joint_2_number": J5_NUMBER,
            "sweep_joint_2_name": J5_NAME,

            "j4_delta_deg": float(j4_delta),
            "j5_delta_deg": float(j5_delta),
            "target_j4_deg": float(target_j4_deg),
            "target_j5_deg": float(target_j5_deg),

            "q1_deg": float(q_arm_now_deg[0]),
            "q2_deg": float(q_arm_now_deg[1]),
            "q3_deg": float(q_arm_now_deg[2]),
            "q4_deg": float(q_arm_now_deg[3]),
            "q5_deg": float(q_arm_now_deg[4]),
            "q6_deg": float(q_arm_now_deg[5]),

            "sigma_1": float(S[0]),
            "sigma_2": float(S[1]),
            "sigma_3": float(S[2]),
            "sigma_4": float(S[3]),
            "sigma_5": float(S[4]),
            "sigma_6": float(S[5]),

            "sigma_min": float(sigma_min),
            "sigma_max": float(sigma_max),
            "condition_number": float(cond),
            "manipulability": float(manip),
        })


# =========================================================
# CSV 저장
# =========================================================
fieldnames = [
    "sweep_joint_1_number",
    "sweep_joint_1_name",
    "sweep_joint_2_number",
    "sweep_joint_2_name",

    "j4_delta_deg",
    "j5_delta_deg",
    "target_j4_deg",
    "target_j5_deg",

    "q1_deg", "q2_deg", "q3_deg", "q4_deg", "q5_deg", "q6_deg",

    "sigma_1", "sigma_2", "sigma_3", "sigma_4", "sigma_5", "sigma_6",
    "sigma_min", "sigma_max", "condition_number", "manipulability",
]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n[INFO] CSV saved: {CSV_PATH}")

for _ in range(120):
    world.step(render=True)

simulation_app.close()