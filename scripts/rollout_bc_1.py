from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import omni.usd

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim, Articulation
from isaacsim.core.utils.types import ArticulationAction


# =========================
# PATH
# =========================
PROJECT_ROOT = Path("/home/june/bimanual_handover_IL")

USD_PATH = str(PROJECT_ROOT / "bimanual_scene.usd")
MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / "bc_single_head" / "best_model.pt")

STATE_STATS_PATH = PROJECT_ROOT / "data" / "bc_ready" / "state_norm_stats.json"
ACTION_STATS_PATH = PROJECT_ROOT / "data" / "bc_ready" / "action_norm_stats.json"

LEFT_ROBOT = "/World/ur_left"
RIGHT_ROBOT = "/World/ur_right"

LEFT_TOOL0_PRIM = "/World/ur_left/wrist_3_link/flange/tool0"
RIGHT_TOOL0_PRIM = "/World/ur_right/wrist_3_link/flange/tool0"
OBJECT = "/World/Cylinder"


# =========================
# BASIC CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TCP_MAX_STEP = 0.02

GRIP_OPEN = 0.0
LEFT_GRIP_CLOSE = 0.43
RIGHT_GRIP_CLOSE = 0.43

LEFT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.4]
RIGHT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.4]
LEFT_OPEN_SEQ = [0.35, 0.25, 0.1, 0.0]

LEFT_HOME_DEG = [-45, -90, 90, 90, 90, 90]
RIGHT_HOME_DEG = [-135, -90, -90, 90, -90, 90]
RIGHT_WAIT_DEG = [-125, -62, -130, 180, -90, 80]
RIGHT_RECEIVE_DEG = [-121, -97, -139, 237, -90, 90]

LEFT_PRE_OFFSET = np.array([0.0, 0.0, 0.50], dtype=np.float64)
LEFT_GRASP_OFFSET = np.array([0.0, 0.0, 0.22], dtype=np.float64)

LEFT_LIFT_Z = 0.50
LEFT_TRANSFER_XY = np.array([0.50, 0.00], dtype=np.float64)
LEFT_POST_RELEASE_UP_DZ = 0.05
LEFT_HOME_TOL = 0.08

SETTLE_STEPS = 30
GRIP_STEP_HOLD = 30
END_HOLD_STEPS = 60
ROLLOUT_STEPS = 2000

# differential IK
DIFF_IK_ALPHA = 0.6
DIFF_IK_DAMPING = 0.05
MAX_Q_STEP = 0.08

# inspection 결과 반영
LEFT_JACOBIAN_INDEX = 12


# =========================
# PHASE MAP
# =========================
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
NUM_PHASES = 12


# =========================
# MODEL
# =========================
class MLPBCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256, 128), dropout_p=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = MLPBCPolicy(
        ckpt["input_dim"],
        ckpt["output_dim"],
        tuple(ckpt["hidden_dims"]),
        ckpt["dropout_p"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# =========================
# IO
# =========================
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# =========================
# UTILS
# =========================
def get_pos(xform_prim):
    positions, _ = xform_prim.get_world_poses()
    return np.array(positions[0], dtype=np.float64)


def arm6_deg_to_rad(arm_deg):
    return np.deg2rad(np.array(arm_deg, dtype=np.float64))


def full_q_from_arm6(arm_deg):
    return np.concatenate([arm6_deg_to_rad(arm_deg), np.zeros(6, dtype=np.float64)])


def clip_vec(x, max_norm):
    x = np.array(x, dtype=np.float64)
    n = np.linalg.norm(x)
    if n < 1e-9:
        return np.zeros_like(x)
    if n > max_norm:
        x = x / n * max_norm
    return x


def phase_onehot(phase_idx, n=NUM_PHASES):
    v = np.zeros(n, dtype=np.float32)
    v[phase_idx] = 1.0
    return v


def apply_arm_hold_with_grip(robot, arm_hold_q, finger_idx, grip_val):
    full = arm_hold_q.copy()
    full[finger_idx] = grip_val
    robot.apply_action(ArticulationAction(joint_positions=full))


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def normalize_value(x, mean, std):
    return (float(x) - float(mean)) / float(std)


def unnormalize_value(x, mean, std):
    return float(x) * float(std) + float(mean)


def build_state_vector_normalized(
    state_cols,
    phase_idx,
    l_pos,
    r_pos,
    o_pos,
    left_grasped,
    right_grasped,
    state_stats,
):
    d = {c: 0.0 for c in state_cols}

    if "left_tcp_x" in d:
        d["left_tcp_x"] = float(l_pos[0])
    if "left_tcp_y" in d:
        d["left_tcp_y"] = float(l_pos[1])
    if "left_tcp_z" in d:
        d["left_tcp_z"] = float(l_pos[2])

    if "right_tcp_x" in d:
        d["right_tcp_x"] = float(r_pos[0])
    if "right_tcp_y" in d:
        d["right_tcp_y"] = float(r_pos[1])
    if "right_tcp_z" in d:
        d["right_tcp_z"] = float(r_pos[2])

    if "obj_x" in d:
        d["obj_x"] = float(o_pos[0])
    if "obj_y" in d:
        d["obj_y"] = float(o_pos[1])
    if "obj_z" in d:
        d["obj_z"] = float(o_pos[2])

    if "left_grasped" in d:
        d["left_grasped"] = float(left_grasped)
    if "right_grasped" in d:
        d["right_grasped"] = float(right_grasped)

    phase_oh = phase_onehot(phase_idx, NUM_PHASES)
    for i in range(NUM_PHASES):
        key = f"phase_{i}"
        if key in d:
            d[key] = float(phase_oh[i])

    for c in state_stats["columns"]:
        d[c] = normalize_value(d[c], state_stats["mean"][c], state_stats["std"][c])

    return np.array([d[c] for c in state_cols], dtype=np.float32)


def unnormalize_action_output(pred, action_cols, action_stats):
    out = np.array(pred, dtype=np.float64).copy()
    for c in action_stats["columns"]:
        idx = action_cols.index(c)
        out[idx] = unnormalize_value(out[idx], action_stats["mean"][c], action_stats["std"][c])
    return out


# =========================
# DIFFERENTIAL IK
# =========================
def damped_least_squares(J, dx, damping=0.05):
    J = np.asarray(J, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64).reshape(-1)
    JJt = J @ J.T
    lam2I = (damping ** 2) * np.eye(JJt.shape[0], dtype=np.float64)
    dq = J.T @ np.linalg.solve(JJt + lam2I, dx)
    return dq


def get_tool0_position_jacobian(left_view, jacobian_index):
    J_all = np.asarray(left_view.get_jacobians())

    if J_all.ndim != 4:
        raise RuntimeError(f"Unexpected Jacobian shape: {J_all.shape}")

    J = J_all[0, jacobian_index]     # (6, num_dof)
    J_pos = J[0:3, 0:6]              # translational, arm 6 dof only
    return np.array(J_pos, dtype=np.float64), J_all.shape


def apply_diff_ik_step(left_robot, left_view, finger_idx, dx_tool0, grip_cmd, jacobian_index):
    q_now = np.array(left_robot.get_joint_positions(), dtype=np.float64)
    q_cmd = q_now.copy()

    J_pos, J_shape = get_tool0_position_jacobian(left_view, jacobian_index)
    dq_arm = damped_least_squares(J_pos, dx_tool0, damping=DIFF_IK_DAMPING)

    dq_arm = np.clip(dq_arm, -MAX_Q_STEP, MAX_Q_STEP)
    q_cmd[:6] = q_now[:6] + DIFF_IK_ALPHA * dq_arm
    q_cmd[finger_idx] = grip_cmd

    left_robot.apply_action(ArticulationAction(joint_positions=q_cmd))
    return dq_arm, J_shape, np.linalg.norm(J_pos), J_pos


# =========================
# SCENE OPEN
# =========================
omni.usd.get_context().open_stage(USD_PATH)
for _ in range(60):
    simulation_app.update()

world = World(stage_units_in_meters=1.0)
world.reset()

left_robot = SingleArticulation(LEFT_ROBOT, name="left_robot")
right_robot = SingleArticulation(RIGHT_ROBOT, name="right_robot")
left_robot.initialize()
right_robot.initialize()

left_view = Articulation(prim_paths_expr=LEFT_ROBOT, name="left_view")
left_view.initialize()

left_tool0 = XFormPrim(LEFT_TOOL0_PRIM)
right_tool0 = XFormPrim(RIGHT_TOOL0_PRIM)
obj = XFormPrim(OBJECT)

left_tool0.initialize()
right_tool0.initialize()
obj.initialize()

model, ckpt = load_model(MODEL_PATH)
state_cols = ckpt["state_cols"]
action_cols = ckpt["action_cols"]

state_stats = load_json(STATE_STATS_PATH)
action_stats = load_json(ACTION_STATS_PATH)

print("=" * 80)
print("CHECKPOINT / DIFF IK INFO")
print(f"input_dim   : {ckpt['input_dim']}")
print(f"output_dim  : {ckpt['output_dim']}")
print(f"state_cols  : {len(state_cols)}")
print(f"action_cols : {len(action_cols)}")
print(f"tool0 prim  : {LEFT_TOOL0_PRIM}")
print(f"jacobian index : {LEFT_JACOBIAN_INDEX}")
print("=" * 80)

LEFT_HOME = full_q_from_arm6(LEFT_HOME_DEG)
RIGHT_HOME = full_q_from_arm6(RIGHT_HOME_DEG)
RIGHT_WAIT = arm6_deg_to_rad(RIGHT_WAIT_DEG)
RIGHT_RECEIVE = arm6_deg_to_rad(RIGHT_RECEIVE_DEG)

left_finger_idx = left_robot.dof_names.index("finger_joint")
right_finger_idx = right_robot.dof_names.index("finger_joint")

left_robot.set_joint_positions(LEFT_HOME)
right_robot.set_joint_positions(RIGHT_HOME)
for _ in range(30):
    world.step(render=True)

obj_start = get_pos(obj)
left_pre_target_tool0 = obj_start + LEFT_PRE_OFFSET
left_grasp_target_tool0 = obj_start + LEFT_GRASP_OFFSET

phase = 0
settle_count = 0
sub_idx = 0
sub_hold = 0
end_hold = 0

left_target_tool0 = left_pre_target_tool0
left_grip_cmd = GRIP_OPEN
right_grip_cmd = GRIP_OPEN

left_grasped = 0.0
right_grasped = 0.0

left_lift_target_tool0 = None
left_transfer_target_tool0 = None
left_post_release_up_target_tool0 = None
left_home_tcp_target_tool0 = left_pre_target_tool0

left_arm_hold_q_close = None
left_arm_hold_q_release = None

for step in range(ROLLOUT_STEPS):
    left_tool0_pos = get_pos(left_tool0)
    right_tool0_pos = get_pos(right_tool0)
    obj_pos = get_pos(obj)
    right_q = right_robot.get_joint_positions()[:6]

    state = build_state_vector_normalized(
        state_cols=state_cols,
        phase_idx=phase,
        l_pos=left_tool0_pos,
        r_pos=right_tool0_pos,
        o_pos=obj_pos,
        left_grasped=left_grasped,
        right_grasped=right_grasped,
        state_stats=state_stats,
    )

    with torch.no_grad():
        pred = model(torch.from_numpy(state).to(DEVICE).unsqueeze(0))[0].cpu().numpy()

    pred_real = unnormalize_action_output(pred, action_cols, action_stats)

    bc_left_delta_tool0 = clip_vec(pred_real[0:3], TCP_MAX_STEP)
    _bc_right_delta = clip_vec(pred_real[3:6], TCP_MAX_STEP)
    _bc_left_grip = clamp01(pred_real[6])
    _bc_right_grip = clamp01(pred_real[7])

    # phase logic
    if phase == 0:
        left_target_tool0 = left_pre_target_tool0
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = GRIP_OPEN
        if np.linalg.norm(left_tool0_pos - left_target_tool0) < 0.03:
            phase = 1

    elif phase == 1:
        left_target_tool0 = left_grasp_target_tool0
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = GRIP_OPEN
        if np.linalg.norm(left_tool0_pos - left_target_tool0) < 0.035:
            phase = 2
            settle_count = 0

    elif phase == 2:
        left_target_tool0 = left_grasp_target_tool0
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = GRIP_OPEN
        settle_count += 1
        if settle_count >= SETTLE_STEPS:
            phase = 3
            sub_idx = 0
            sub_hold = 0
            left_arm_hold_q_close = left_robot.get_joint_positions().copy()

    elif phase == 3:
        left_target_tool0 = left_grasp_target_tool0
        left_grip_cmd = LEFT_CLOSE_SEQ[sub_idx]
        right_grip_cmd = GRIP_OPEN
        sub_hold += 1
        if sub_hold >= GRIP_STEP_HOLD:
            sub_hold = 0
            sub_idx += 1
            if sub_idx >= len(LEFT_CLOSE_SEQ):
                left_grip_cmd = LEFT_GRIP_CLOSE
                left_grasped = 1.0
                left_lift_target_tool0 = np.array(
                    [left_tool0_pos[0], left_tool0_pos[1], LEFT_LIFT_Z],
                    dtype=np.float64,
                )
                phase = 4

    elif phase == 4:
        left_target_tool0 = left_lift_target_tool0
        left_grip_cmd = LEFT_GRIP_CLOSE
        right_grip_cmd = GRIP_OPEN
        if np.linalg.norm(left_tool0_pos - left_target_tool0) < 0.035:
            dx = LEFT_TRANSFER_XY[0] - left_tool0_pos[0]
            dy = LEFT_TRANSFER_XY[1] - left_tool0_pos[1]
            left_transfer_target_tool0 = np.array(
                [left_tool0_pos[0] + dx, left_tool0_pos[1] + dy, left_tool0_pos[2]],
                dtype=np.float64,
            )
            phase = 5

    elif phase == 5:
        left_target_tool0 = left_transfer_target_tool0
        left_grip_cmd = LEFT_GRIP_CLOSE
        right_grip_cmd = GRIP_OPEN
        if np.linalg.norm(left_tool0_pos - left_target_tool0) < 0.04:
            phase = 6

    elif phase == 6:
        left_target_tool0 = left_transfer_target_tool0
        left_grip_cmd = LEFT_GRIP_CLOSE
        right_grip_cmd = GRIP_OPEN
        if np.all(np.abs(right_q - RIGHT_RECEIVE) < np.deg2rad(2.0)):
            phase = 7

    elif phase == 7:
        left_target_tool0 = left_transfer_target_tool0
        right_grip_cmd = RIGHT_CLOSE_SEQ[sub_idx]
        left_grip_cmd = LEFT_GRIP_CLOSE
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
        left_target_tool0 = left_transfer_target_tool0
        right_grip_cmd = RIGHT_GRIP_CLOSE
        left_grip_cmd = LEFT_OPEN_SEQ[sub_idx]
        sub_hold += 1
        if sub_hold >= GRIP_STEP_HOLD:
            sub_hold = 0
            sub_idx += 1
            if sub_idx >= len(LEFT_OPEN_SEQ):
                left_grip_cmd = GRIP_OPEN
                left_grasped = 0.0
                left_post_release_up_target_tool0 = np.array(
                    [left_tool0_pos[0], left_tool0_pos[1], left_tool0_pos[2] + LEFT_POST_RELEASE_UP_DZ],
                    dtype=np.float64,
                )
                phase = 9

    elif phase == 9:
        left_target_tool0 = left_post_release_up_target_tool0
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE
        if np.linalg.norm(left_tool0_pos - left_target_tool0) < 0.03:
            phase = 10

    elif phase == 10:
        left_target_tool0 = left_home_tcp_target_tool0
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE
        if np.linalg.norm(left_tool0_pos - left_target_tool0) < LEFT_HOME_TOL:
            phase = 11
            end_hold = 0

    elif phase == 11:
        left_target_tool0 = left_home_tcp_target_tool0
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE
        end_hold += 1
        if end_hold >= END_HOLD_STEPS:
            print("[INFO] rollout success end")
            break

    if phase in [0, 1, 2, 4, 5, 6, 9, 10]:
        dx_tool0 = bc_left_delta_tool0
    else:
        dx_tool0 = np.zeros(3, dtype=np.float64)

    prev_left_tool0 = left_tool0_pos.copy()

    # left control: differential IK
    if phase in [0, 1, 2, 4, 5, 6, 9, 10]:
        try:
            dq_arm, J_shape, J_norm, J_pos = apply_diff_ik_step(
                left_robot=left_robot,
                left_view=left_view,
                finger_idx=left_finger_idx,
                dx_tool0=dx_tool0,
                grip_cmd=left_grip_cmd,
                jacobian_index=LEFT_JACOBIAN_INDEX,
            )
            dx_pred = J_pos @ dq_arm
            diffik_success = True
            err_msg = ""
        except Exception as e:
            dq_arm = np.zeros(6, dtype=np.float64)
            J_shape = None
            J_norm = None
            dx_pred = np.zeros(3, dtype=np.float64)
            diffik_success = False
            err_msg = str(e)

            q_now = np.array(left_robot.get_joint_positions(), dtype=np.float64)
            q_now[left_finger_idx] = left_grip_cmd
            left_robot.apply_action(ArticulationAction(joint_positions=q_now))
    else:
        dq_arm = np.zeros(6, dtype=np.float64)
        J_shape = None
        J_norm = None
        dx_pred = np.zeros(3, dtype=np.float64)
        diffik_success = None
        err_msg = ""

        if phase == 3:
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
            left_full = np.array(left_robot.get_joint_positions(), dtype=np.float64)
            left_full[left_finger_idx] = left_grip_cmd
            left_robot.apply_action(ArticulationAction(joint_positions=left_full))

    # right arm scripted
    right_arm_target = RIGHT_WAIT if phase in [0, 1, 2, 3, 4, 5] else RIGHT_RECEIVE
    right_full = np.array(right_robot.get_joint_positions(), dtype=np.float64)
    right_full[:6] = right_arm_target
    right_full[right_finger_idx] = right_grip_cmd
    right_robot.apply_action(ArticulationAction(joint_positions=right_full))

    world.step(render=True)

    new_left_tool0 = get_pos(left_tool0)
    actual_left_delta_tool0 = new_left_tool0 - prev_left_tool0

    if step % 25 == 0:
        print(
            f"[{step:04d}] "
            f"phase={phase}({PHASE_NAMES[phase]}) | "
            f"tool0_pos={np.round(left_tool0_pos, 4)} | "
            f"bc_left_delta_tool0={np.round(dx_tool0, 4)} | "
            f"dx_pred={np.round(dx_pred, 4)} | "
            f"actual_delta_tool0={np.round(actual_left_delta_tool0, 4)} | "
            f"dq_arm={np.round(dq_arm, 4)} | "
            f"J_shape={J_shape} | J_norm={J_norm} | "
            f"diffik_success={diffik_success} | err={err_msg}"
        )

print("[INFO] finished")
simulation_app.close()