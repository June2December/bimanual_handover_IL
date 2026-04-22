from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
import numpy as np
import torch
import omni.usd
import isaacsim.robot_motion.motion_generation as mg

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction

# =========================
# path
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(SCRIPT_DIR, "utils")
ML_DIR = os.path.join(SCRIPT_DIR, "ml")

for p in [UTILS_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# utils
from rotation_utils import get_pose_and_rot6d, rot6d_to_quat
from robot_utils import arm6_deg_to_rad, full_q_from_arm6, apply_arm_hold_with_grip
from target_utils import get_right_target_poses
from phase_utils import PHASE_NAMES

# ml
from columns import (
    STATE_CONT_COLS,
    STATE_BIN_COLS,
    PHASE_COLS,
    ARM_ACTION_CONT_COLS,
    GRIP_ACTION_BIN_COLS,
)
from model import BCPolicy
from norm_utils import load_norm_stats

# =========================
# RMPFlow Controller
# =========================
class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name, robot_articulation, physics_dt=1.0 / 60.0):
        config = mg.interface_config_loader.load_supported_motion_policy_config("UR10", "RMPflow")
        rmp_flow = mg.lula.motion_policies.RmpFlow(**config)
        articulation_policy = mg.ArticulationMotionPolicy(robot_articulation, rmp_flow, physics_dt)
        super().__init__(name=name, articulation_motion_policy=articulation_policy)

        pos, ori = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=pos,
            robot_orientation=ori,
        )

# =========================
# config
# =========================
usd_path = "/home/june/bimanual_handover_IL/bimanual_scene.usd"

left_robot_path = "/World/ur_left"
right_robot_path = "/World/ur_right"

left_tcp_path = "/World/ur_left/wrist_3_link/flange/tool0"
right_tcp_path = "/World/ur_right/wrist_3_link/flange/tool0"
cylinder_path = "/World/Cylinder"

physics_dt = 1.0 / 60.0
MAX_STEPS = 800
PRINT_EVERY = 10

MODEL_PATH = "/home/june/bimanual_handover_IL/checkpoints/bc_rot6d/best.pt"
STATE_STATS_PATH = "/home/june/bimanual_handover_IL/checkpoints/bc_rot6d/state_norm_stats.json"
ACTION_STATS_PATH = "/home/june/bimanual_handover_IL/checkpoints/bc_rot6d/action_norm_stats.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEFT_HOME_DEG = [-45, -90, 90, 90, 90, 90]
RIGHT_HOME_DEG = [-135, -90, -90, 90, -90, 90]

RIGHT_WAIT_DEG = [-125, -62, -130, 180, -90, 80]
RIGHT_RECEIVE_DEG = [-121, -97, -139, 237, -90, 90]

LEFT_HOME = full_q_from_arm6(LEFT_HOME_DEG)
RIGHT_HOME = full_q_from_arm6(RIGHT_HOME_DEG)
RIGHT_WAIT = arm6_deg_to_rad(RIGHT_WAIT_DEG)
RIGHT_RECEIVE = arm6_deg_to_rad(RIGHT_RECEIVE_DEG)

RIGHT_WAIT_FULL = full_q_from_arm6(RIGHT_WAIT_DEG)
RIGHT_RECEIVE_FULL = full_q_from_arm6(RIGHT_RECEIVE_DEG)

GRIP_OPEN = 0.0
LEFT_GRIP_CLOSE = 0.43
RIGHT_GRIP_CLOSE = 0.43

LEFT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.4]
RIGHT_CLOSE_SEQ = [0.1, 0.2, 0.3, 0.4]
LEFT_OPEN_SEQ = [0.35, 0.25, 0.1, 0.0]

SETTLE_STEPS = 10
GRIP_STEP_HOLD = 10
END_HOLD_STEPS = 60

LEFT_HOME_TOL = 0.08

# =========================
# helpers
# =========================
def l2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

def build_phase_onehot(phase_idx, num_phases):
    v = np.zeros(num_phases, dtype=np.float32)
    v[phase_idx] = 1.0
    return v

def get_grip_state_from_robot(robot, threshold=0.42):
    q = robot.get_joint_positions()
    finger_q = q[6] if len(q) > 6 else 0.0
    return 1.0 if finger_q >= threshold else 0.0

def parse_arm_action(pred):
    left_pos = pred[0:3]
    left_rot6d = pred[3:9]
    right_pos = pred[9:12]
    right_rot6d = pred[12:18]
    return left_pos, left_rot6d, right_pos, right_rot6d

def normalize_state(state_cont, state_bin, phase_onehot, state_mean, state_std):
    state_cont_norm = (state_cont - state_mean) / state_std
    return np.concatenate([state_cont_norm, state_bin, phase_onehot], axis=0).astype(np.float32)

def denormalize_arm_action(arm_pred_norm, action_mean, action_std):
    return arm_pred_norm * action_std + action_mean

def make_state(left_tcp, right_tcp, cyl, left_robot, right_robot, phase_idx):
    left_pos, left_rot6d = get_pose_and_rot6d(left_tcp)
    right_pos, right_rot6d = get_pose_and_rot6d(right_tcp)
    obj_pos, obj_rot6d = get_pose_and_rot6d(cyl)

    left_grip_state = np.array([get_grip_state_from_robot(left_robot)], dtype=np.float32)
    right_grip_state = np.array([get_grip_state_from_robot(right_robot)], dtype=np.float32)
    phase_onehot = build_phase_onehot(phase_idx, len(PHASE_COLS))

    state_cont = np.concatenate([
        left_pos.astype(np.float32),
        left_rot6d.astype(np.float32),
        right_pos.astype(np.float32),
        right_rot6d.astype(np.float32),
        obj_pos.astype(np.float32),
        obj_rot6d.astype(np.float32),
    ], axis=0)

    state_bin = np.concatenate([left_grip_state, right_grip_state], axis=0)

    info = {
        "left_pos": left_pos.copy(),
        "right_pos": right_pos.copy(),
        "obj_pos": obj_pos.copy(),
    }
    return state_cont, state_bin, phase_onehot, info

# =========================
# open stage
# =========================
omni.usd.get_context().open_stage(usd_path)
for _ in range(60):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Stage open failed")

world = World(stage_units_in_meters=1.0)
world.reset()

left_robot = SingleArticulation(left_robot_path, name="left_robot")
right_robot = SingleArticulation(right_robot_path, name="right_robot")

left_tcp = XFormPrim(left_tcp_path)
right_tcp = XFormPrim(right_tcp_path)
cyl = XFormPrim(cylinder_path)

left_robot.initialize()
right_robot.initialize()
left_tcp.initialize()
right_tcp.initialize()
cyl.initialize()

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

right_wait_tcp_pos, right_wait_rot6d, right_receive_tcp_pos, right_receive_rot6d = get_right_target_poses(
    world=world,
    right_robot=right_robot,
    right_tcp=right_tcp,
    right_wait_full=RIGHT_WAIT_FULL,
    right_receive_full=RIGHT_RECEIVE_FULL,
    right_home=RIGHT_HOME,
    get_pose_and_rot6d=get_pose_and_rot6d,
)

left_ctrl = RMPFlowController("left_ctrl", left_robot, physics_dt)

# =========================
# load stats/model
# =========================
state_cols_loaded, state_mean, state_std = load_norm_stats(STATE_STATS_PATH)
action_cols_loaded, action_mean, action_std = load_norm_stats(ACTION_STATS_PATH)

assert state_cols_loaded == STATE_CONT_COLS
assert action_cols_loaded == ARM_ACTION_CONT_COLS

state_dim = len(STATE_CONT_COLS) + len(STATE_BIN_COLS) + len(PHASE_COLS)
arm_dim = len(ARM_ACTION_CONT_COLS)
grip_dim = len(GRIP_ACTION_BIN_COLS)

model = BCPolicy(
    state_dim=state_dim,
    arm_dim=arm_dim,
    grip_dim=grip_dim,
    hidden_dim=256,
).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
model.eval()

print("[INFO] model loaded")

# =========================
# phase vars
# =========================
phase = 0
sub_idx = 0
sub_hold = 0
settle_count = 0
end_hold = 0

left_target = None
left_grip_cmd = GRIP_OPEN
right_grip_cmd = GRIP_OPEN

left_grasped = 0.0
right_grasped = 0.0

left_arm_hold_q_close = None
left_arm_hold_q_release = None

# =========================
# rollout
# =========================
for step in range(MAX_STEPS):
    state_cont, state_bin, phase_onehot, info = make_state(
        left_tcp, right_tcp, cyl, left_robot, right_robot, phase
    )

    state = normalize_state(
        state_cont=state_cont,
        state_bin=state_bin,
        phase_onehot=phase_onehot,
        state_mean=state_mean,
        state_std=state_std,
    )

    x = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        arm_pred_norm, grip_pred = model(x)
        arm_pred_norm = arm_pred_norm.squeeze(0).cpu().numpy()

    arm_pred = denormalize_arm_action(arm_pred_norm, action_mean, action_std)
    pred_left_pos, pred_left_rot6d, pred_right_pos, pred_right_rot6d = parse_arm_action(arm_pred)
    pred_left_quat = np.array(rot6d_to_quat(pred_left_rot6d), dtype=np.float32)

    obj_pos = info["obj_pos"]

    # -------------------------
    # phase logic
    # -------------------------
    if phase == 0:
        left_target = pred_left_pos
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = GRIP_OPEN

        if l2(info["left_pos"], left_target) < 0.03:
            phase = 1

    elif phase == 1:
        left_target = pred_left_pos
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = GRIP_OPEN

        if l2(info["left_pos"], left_target) < 0.035:
            phase = 2
            settle_count = 0

    elif phase == 2:
        left_target = pred_left_pos
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = GRIP_OPEN
        settle_count += 1

        if settle_count >= SETTLE_STEPS:
            phase = 3
            sub_idx = 0
            sub_hold = 0
            left_arm_hold_q_close = left_robot.get_joint_positions().copy()

    elif phase == 3:
        left_target = pred_left_pos
        left_grip_cmd = LEFT_CLOSE_SEQ[sub_idx]
        right_grip_cmd = GRIP_OPEN

        sub_hold += 1
        if sub_hold >= GRIP_STEP_HOLD:
            sub_hold = 0
            sub_idx += 1

            if sub_idx >= len(LEFT_CLOSE_SEQ):
                left_grip_cmd = LEFT_GRIP_CLOSE
                left_grasped = 1.0
                phase = 4

    elif phase == 4:
        left_target = pred_left_pos
        left_grip_cmd = LEFT_GRIP_CLOSE
        right_grip_cmd = GRIP_OPEN

        if l2(info["left_pos"], left_target) < 0.04:
            phase = 5

    elif phase == 5:
        left_target = pred_left_pos
        left_grip_cmd = LEFT_GRIP_CLOSE
        right_grip_cmd = GRIP_OPEN

        if l2(info["left_pos"], left_target) < 0.04:
            phase = 6

    elif phase == 6:
        left_target = pred_left_pos
        left_grip_cmd = LEFT_GRIP_CLOSE
        right_grip_cmd = GRIP_OPEN

        right_q = right_robot.get_joint_positions()[:6]
        if np.all(np.abs(right_q - RIGHT_RECEIVE) < np.deg2rad(2.0)):
            phase = 7
            sub_idx = 0
            sub_hold = 0

    elif phase == 7:
        left_target = pred_left_pos
        left_grip_cmd = LEFT_GRIP_CLOSE
        right_grip_cmd = RIGHT_CLOSE_SEQ[sub_idx]

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
        left_target = pred_left_pos
        right_grip_cmd = RIGHT_GRIP_CLOSE
        left_grip_cmd = LEFT_OPEN_SEQ[sub_idx]

        sub_hold += 1
        if sub_hold >= GRIP_STEP_HOLD:
            sub_hold = 0
            sub_idx += 1

            if sub_idx >= len(LEFT_OPEN_SEQ):
                left_grip_cmd = GRIP_OPEN
                left_grasped = 0.0
                phase = 9

    elif phase == 9:
        left_target = pred_left_pos
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE

        if l2(info["left_pos"], left_target) < 0.04:
            phase = 10

    elif phase == 10:
        left_target = pred_left_pos
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE

        if l2(info["left_pos"], left_target) < LEFT_HOME_TOL:
            phase = 11
            end_hold = 0

    elif phase == 11:
        left_target = pred_left_pos
        left_grip_cmd = GRIP_OPEN
        right_grip_cmd = RIGHT_GRIP_CLOSE

        end_hold += 1
        if end_hold >= END_HOLD_STEPS:
            print("[DONE] success-ish end")
            break

    # -------------------------
    # debug
    # -------------------------
    if step % PRINT_EVERY == 0:
        print("=" * 80)
        print(f"[STEP {step}] phase={phase} ({PHASE_NAMES[phase]})")
        print("obj_pos         =", np.round(obj_pos, 4))
        print("left_cur        =", np.round(info["left_pos"], 4))
        print("pred_left_pos   =", np.round(pred_left_pos, 4))
        print("left_target     =", np.round(left_target, 4))
        print("left_target-cur =", np.round(left_target - info["left_pos"], 4))
        print("left_err        =", round(l2(info["left_pos"], left_target), 4))
        print("right_cur       =", np.round(info["right_pos"], 4))
        print("pred_right_pos  =", np.round(pred_right_pos, 4))
        print("pred_left_rot6d =", np.round(pred_left_rot6d, 4))
        print("pred_left_quat  =", np.round(pred_left_quat, 4))
        print("grip_cmd        =", left_grip_cmd, right_grip_cmd)

    # -------------------------
    # left arm control
    # -------------------------
    if phase in [0, 1, 2, 4, 5, 6, 7, 9, 10, 11]:
        left_action = left_ctrl.forward(
            target_end_effector_position=left_target,
            target_end_effector_orientation=pred_left_quat,
        )
        left_robot.apply_action(left_action)

    elif phase == 3:
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
        left_full = left_robot.get_joint_positions().copy()
        left_full[left_finger_idx] = left_grip_cmd
        left_robot.apply_action(ArticulationAction(joint_positions=left_full))

    # -------------------------
    # right arm control (expert 그대로)
    # -------------------------
    if phase in [0, 1, 2, 3, 4, 5]:
        right_arm_target = RIGHT_WAIT
    else:
        right_arm_target = RIGHT_RECEIVE

    right_full = right_robot.get_joint_positions().copy()
    right_full[:6] = right_arm_target
    right_full[right_finger_idx] = right_grip_cmd
    right_robot.apply_action(ArticulationAction(joint_positions=right_full))

    prev_left = info["left_pos"].copy()
    prev_right = info["right_pos"].copy()

    world.step(render=True)

    new_left_pos, _ = get_pose_and_rot6d(left_tcp)
    new_right_pos, _ = get_pose_and_rot6d(right_tcp)

    if step % PRINT_EVERY == 0:
        print("left_step_move  =", round(float(np.linalg.norm(new_left_pos - prev_left)), 4))
        print("right_step_move =", round(float(np.linalg.norm(new_right_pos - prev_right)), 4))

simulation_app.close()