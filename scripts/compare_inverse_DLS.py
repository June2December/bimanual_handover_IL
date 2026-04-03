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
# 설정
# =========================================================
USD_PATH = "/home/june/bimanual_handover_IL/poly_scene.usd"
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

# singularity 근처 자세
START_POSE_DEG = [-90, -90, -90, 90, 0, 90]

# 목표 TCP 속도 [vx, vy, vz, wx, wy, wz]
XDOT_DES = np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0], dtype=np.float64)

SETTLE_STEPS = 40
CONTROL_STEPS = 80
DT = 1.0 / 60.0
LAMBDA = 0.1
EPS = 1e-12

SAVE_DIR = "/home/june/bimanual_handover_IL/data/"
PINV_CSV = os.path.join(SAVE_DIR, "pinv_log.csv")
DLS_CSV = os.path.join(SAVE_DIR, "dls_log.csv")


# =========================================================
# helper
# =========================================================
def deg_to_rad(values_deg):
    return np.deg2rad(np.array(values_deg, dtype=np.float64))


def rad_to_deg(values_rad):
    return np.rad2deg(np.array(values_rad, dtype=np.float64))


def get_arm_joint_indices(robot):
    return np.array([robot.get_dof_index(name) for name in ARM_DOF_NAMES], dtype=np.int64)


def set_arm_pose(robot, arm_joint_indices, q_arm_rad):
    q_full = robot.get_joint_positions().copy()
    q_full[arm_joint_indices] = q_arm_rad
    robot.apply_action(ArticulationAction(joint_positions=q_full))


def get_arm_positions(robot, arm_joint_indices):
    q_full = robot.get_joint_positions().copy()
    return q_full[arm_joint_indices].astype(np.float64)


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

    # expected: (1, num_bodies-1, 6, num_dof)
    J_full = jac_all[0, jac_row, :, :]         # (6, num_dof)
    J_arm = J_full[:, arm_joint_indices]       # (6, 6)
    return J_arm.astype(np.float64)


def qdot_pinv(J, xdot):
    J_pinv = np.linalg.pinv(J)
    return J_pinv @ xdot


def qdot_dls(J, xdot, lambda_val):
    I = np.eye(J.shape[0], dtype=np.float64)
    return J.T @ np.linalg.inv(J @ J.T + (lambda_val ** 2) * I) @ xdot


def apply_qdot_step(robot, arm_joint_indices, qdot, dt):
    q_full = robot.get_joint_positions().copy()
    q_arm = q_full[arm_joint_indices].astype(np.float64)
    q_arm_next = q_arm + qdot * dt
    q_full[arm_joint_indices] = q_arm_next
    robot.apply_action(ArticulationAction(joint_positions=q_full))


def run_controller(world, robot, arm_joint_indices, controller_name, save_csv_path, lambda_val=None):
    print("\n" + "=" * 90)
    print(f"[RUN] {controller_name}")

    log_rows = []

    for step in range(CONTROL_STEPS):
        J = get_jacobian_6x6(robot, EE_LINK_NAME, arm_joint_indices)
        q_now = get_arm_positions(robot, arm_joint_indices)

        if controller_name == "pinv":
            qdot = qdot_pinv(J, XDOT_DES)
        elif controller_name == "dls":
            qdot = qdot_dls(J, XDOT_DES, lambda_val)
        else:
            raise ValueError("controller_name must be 'pinv' or 'dls'")

        xdot_actual = J @ qdot
        qdot_norm = float(np.linalg.norm(qdot))
        track_err = float(np.linalg.norm(XDOT_DES - xdot_actual))

        if step % 10 == 0 or step == CONTROL_STEPS - 1:
            print(
                f"[{controller_name:4s}] step={step:03d} | "
                f"||qdot||={qdot_norm:.6e} | "
                f"track_err={track_err:.6e}"
            )
            print("q_now_deg =", np.round(rad_to_deg(q_now), 3))
            print("qdot      =", np.round(qdot, 6))
            print("xdot_act  =", np.round(xdot_actual, 6))

        log_rows.append({
            "step": step,
            "q1_deg": float(rad_to_deg(q_now)[0]),
            "q2_deg": float(rad_to_deg(q_now)[1]),
            "q3_deg": float(rad_to_deg(q_now)[2]),
            "q4_deg": float(rad_to_deg(q_now)[3]),
            "q5_deg": float(rad_to_deg(q_now)[4]),
            "q6_deg": float(rad_to_deg(q_now)[5]),

            "qdot_1": float(qdot[0]),
            "qdot_2": float(qdot[1]),
            "qdot_3": float(qdot[2]),
            "qdot_4": float(qdot[3]),
            "qdot_5": float(qdot[4]),
            "qdot_6": float(qdot[5]),
            "qdot_norm": qdot_norm,

            "xdot_des_1": float(XDOT_DES[0]),
            "xdot_des_2": float(XDOT_DES[1]),
            "xdot_des_3": float(XDOT_DES[2]),
            "xdot_des_4": float(XDOT_DES[3]),
            "xdot_des_5": float(XDOT_DES[4]),
            "xdot_des_6": float(XDOT_DES[5]),

            "xdot_act_1": float(xdot_actual[0]),
            "xdot_act_2": float(xdot_actual[1]),
            "xdot_act_3": float(xdot_actual[2]),
            "xdot_act_4": float(xdot_actual[3]),
            "xdot_act_5": float(xdot_actual[4]),
            "xdot_act_6": float(xdot_actual[5]),

            "track_err": track_err,
        })

        apply_qdot_step(robot, arm_joint_indices, qdot, DT)
        world.step(render=True)

    fieldnames = [
        "step",
        "q1_deg", "q2_deg", "q3_deg", "q4_deg", "q5_deg", "q6_deg",
        "qdot_1", "qdot_2", "qdot_3", "qdot_4", "qdot_5", "qdot_6",
        "qdot_norm",
        "xdot_des_1", "xdot_des_2", "xdot_des_3", "xdot_des_4", "xdot_des_5", "xdot_des_6",
        "xdot_act_1", "xdot_act_2", "xdot_act_3", "xdot_act_4", "xdot_act_5", "xdot_act_6",
        "track_err",
    ]

    with open(save_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"[INFO] saved csv: {save_csv_path}")


# =========================================================
# 시작
# =========================================================
print("USD exists:", os.path.exists(USD_PATH), USD_PATH)
if not os.path.exists(USD_PATH):
    raise FileNotFoundError(f"USD not found: {USD_PATH}")

os.makedirs(SAVE_DIR, exist_ok=True)

omni.usd.get_context().open_stage(USD_PATH)
for _ in range(60):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Stage open failed")

if not stage.GetPrimAtPath(ROBOT_PRIM_PATH).IsValid():
    raise RuntimeError(f"Invalid robot prim path: {ROBOT_PRIM_PATH}")

world = World(stage_units_in_meters=1.0)
world.reset()

robot = SingleArticulation(ROBOT_PRIM_PATH, name="target_robot")
robot.initialize()

world.reset()
robot.initialize()

view = robot._articulation_view
arm_joint_indices = get_arm_joint_indices(robot)

print("\n[INFO] Robot initialized")
print("robot prim path   =", ROBOT_PRIM_PATH)
print("dof_names         =", list(robot.dof_names))
print("body_names        =", list(view.body_names))
print("arm_joint_indices =", arm_joint_indices.tolist())
print("start_pose_deg    =", START_POSE_DEG)
print("xdot_des          =", XDOT_DES)
print("lambda            =", LAMBDA)

# 시작 자세 적용
start_pose_rad = deg_to_rad(START_POSE_DEG)
set_arm_pose(robot, arm_joint_indices, start_pose_rad)

for _ in range(SETTLE_STEPS):
    world.step(render=True)

# 시작 자세 비교 출력
J0 = get_jacobian_6x6(robot, EE_LINK_NAME, arm_joint_indices)
q0 = get_arm_positions(robot, arm_joint_indices)

qdot0_pinv = qdot_pinv(J0, XDOT_DES)
qdot0_dls = qdot_dls(J0, XDOT_DES, LAMBDA)

print("\n" + "=" * 90)
print("[COMPARE at start pose]")
print("q_now_deg       =", np.round(rad_to_deg(q0), 3))
print("qdot_pinv       =", np.round(qdot0_pinv, 6))
print("qdot_dls        =", np.round(qdot0_dls, 6))
print("||qdot_pinv||   =", np.linalg.norm(qdot0_pinv))
print("||qdot_dls||    =", np.linalg.norm(qdot0_dls))

# 1) pinv
run_controller(
    world=world,
    robot=robot,
    arm_joint_indices=arm_joint_indices,
    controller_name="pinv",
    save_csv_path=PINV_CSV,
)

# 리셋
set_arm_pose(robot, arm_joint_indices, start_pose_rad)
for _ in range(SETTLE_STEPS):
    world.step(render=True)

# 2) dls
run_controller(
    world=world,
    robot=robot,
    arm_joint_indices=arm_joint_indices,
    controller_name="dls",
    save_csv_path=DLS_CSV,
    lambda_val=LAMBDA,
)

print("\n[INFO] Finished pinv vs DLS test")
print("[INFO] pinv csv:", PINV_CSV)
print("[INFO] dls  csv:", DLS_CSV)

for _ in range(120):
    world.step(render=True)

simulation_app.close()