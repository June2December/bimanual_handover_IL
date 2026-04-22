from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
import csv
import time
import numpy as np
import torch
import omni.usd
import isaacsim.robot_motion.motion_generation as mg

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, XFormPrim
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
from phase_utils import PHASE_NAMES, phase_feature_names, phase_to_onehot
from episode_checks_utils import check_episode_termination

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
NUM_EPISODES = 10

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

LEFT_TOP_ORI = np.array([0.70710677, 0.0, 0.70710677, 0.0], dtype=np.float32)

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

LIFT_CLIP_Z_LOW = 0.35
LIFT_CLIP_Z_HIGH = 0.60

SAVE_DIR = "/home/june/bimanual_handover_IL/data/rollout_bc_rot6d"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# helpers
# =========================
def l2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

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
    phase_onehot = phase_to_onehot(phase_idx)

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
        "left_rot6d": left_rot6d.copy(),
        "right_pos": right_pos.copy(),
        "right_rot6d": right_rot6d.copy(),
        "obj_pos": obj_pos.copy(),
        "obj_rot6d": obj_rot6d.copy(),
        "left_grip_state": float(left_grip_state[0]),
        "right_grip_state": float(right_grip_state[0]),
        "phase_onehot": phase_onehot.copy(),
    }
    return state_cont, state_bin, phase_onehot, info

def clip_lift_z(pred_left_pos):
    pred = pred_left_pos.copy()
    pred[2] = np.clip(pred[2], LIFT_CLIP_Z_LOW, LIFT_CLIP_Z_HIGH)
    return pred

def init_world_and_handles():
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

    return world, left_robot, right_robot, left_tcp, right_tcp, cyl, left_finger_idx, right_finger_idx

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
# rollout one episode
# =========================
def run_one_episode(ep_idx):
    world, left_robot, right_robot, left_tcp, right_tcp, cyl, left_finger_idx, right_finger_idx = init_world_and_handles()

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

    csv_path = os.path.join(SAVE_DIR, f"rollout_phase4_zonly_ori_fixed_ep{ep_idx:03d}_{int(time.time())}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    phase_cols = phase_feature_names()

    csv_writer.writerow([
        "step",
        "phase",
        "phase_name",

        "left_pos_x", "left_pos_y", "left_pos_z",
        "left_rot6d_0", "left_rot6d_1", "left_rot6d_2",
        "left_rot6d_3", "left_rot6d_4", "left_rot6d_5",

        "right_pos_x", "right_pos_y", "right_pos_z",
        "right_rot6d_0", "right_rot6d_1", "right_rot6d_2",
        "right_rot6d_3", "right_rot6d_4", "right_rot6d_5",

        "obj_pos_x", "obj_pos_y", "obj_pos_z",
        "obj_rot6d_0", "obj_rot6d_1", "obj_rot6d_2",
        "obj_rot6d_3", "obj_rot6d_4", "obj_rot6d_5",

        "left_grip_state",
        "right_grip_state",

        *phase_cols,

        "action_left_pos_x", "action_left_pos_y", "action_left_pos_z",
        "action_left_rot6d_0", "action_left_rot6d_1", "action_left_rot6d_2",
        "action_left_rot6d_3", "action_left_rot6d_4", "action_left_rot6d_5",
        "action_left_grip_cmd",

        "action_right_pos_x", "action_right_pos_y", "action_right_pos_z",
        "action_right_rot6d_0", "action_right_rot6d_1", "action_right_rot6d_2",
        "action_right_rot6d_3", "action_right_rot6d_4", "action_right_rot6d_5",
        "action_right_grip_cmd",
    ])

    phase = 0
    sub_idx = 0
    sub_hold = 0
    settle_count = 0
    end_hold = 0

    left_target = None
    left_grip_cmd = GRIP_OPEN
    right_grip_cmd = GRIP_OPEN

    left_arm_hold_q_close = None
    left_arm_hold_q_release = None

    progress_hist = []
    success = False
    fail_reason = None

    try:
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
                        phase = 4

            elif phase == 4:
                pred_lift = clip_lift_z(pred_left_pos)
                left_target = np.array([
                    info["left_pos"][0],
                    info["left_pos"][1],
                    pred_lift[2],
                ], dtype=np.float64)

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
                        phase = 8
                        sub_idx = 0
                        sub_hold = 0
                        left_arm_hold_q_release = left_robot.get_joint_positions().copy()

            elif phase == 8:
                left_target = pred_left_pos
                left_grip_cmd = LEFT_OPEN_SEQ[sub_idx]
                right_grip_cmd = RIGHT_GRIP_CLOSE

                sub_hold += 1
                if sub_hold >= GRIP_STEP_HOLD:
                    sub_hold = 0
                    sub_idx += 1
                    if sub_idx >= len(LEFT_OPEN_SEQ):
                        left_grip_cmd = GRIP_OPEN
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
                    success = True
                    print("[DONE] success-ish end")
                    break

            csv_writer.writerow([
                step,
                phase,
                PHASE_NAMES[phase],

                float(info["left_pos"][0]), float(info["left_pos"][1]), float(info["left_pos"][2]),
                float(info["left_rot6d"][0]), float(info["left_rot6d"][1]), float(info["left_rot6d"][2]),
                float(info["left_rot6d"][3]), float(info["left_rot6d"][4]), float(info["left_rot6d"][5]),

                float(info["right_pos"][0]), float(info["right_pos"][1]), float(info["right_pos"][2]),
                float(info["right_rot6d"][0]), float(info["right_rot6d"][1]), float(info["right_rot6d"][2]),
                float(info["right_rot6d"][3]), float(info["right_rot6d"][4]), float(info["right_rot6d"][5]),

                float(info["obj_pos"][0]), float(info["obj_pos"][1]), float(info["obj_pos"][2]),
                float(info["obj_rot6d"][0]), float(info["obj_rot6d"][1]), float(info["obj_rot6d"][2]),
                float(info["obj_rot6d"][3]), float(info["obj_rot6d"][4]), float(info["obj_rot6d"][5]),

                float(info["left_grip_state"]),
                float(info["right_grip_state"]),

                *info["phase_onehot"].tolist(),

                float(left_target[0]), float(left_target[1]), float(left_target[2]),
                float(pred_left_rot6d[0]), float(pred_left_rot6d[1]), float(pred_left_rot6d[2]),
                float(pred_left_rot6d[3]), float(pred_left_rot6d[4]), float(pred_left_rot6d[5]),
                float(left_grip_cmd),

                float(right_wait_tcp_pos[0] if phase in [0, 1, 2, 3, 4, 5] else right_receive_tcp_pos[0]),
                float(right_wait_tcp_pos[1] if phase in [0, 1, 2, 3, 4, 5] else right_receive_tcp_pos[1]),
                float(right_wait_tcp_pos[2] if phase in [0, 1, 2, 3, 4, 5] else right_receive_tcp_pos[2]),
                float(right_wait_rot6d[0] if phase in [0, 1, 2, 3, 4, 5] else right_receive_rot6d[0]),
                float(right_wait_rot6d[1] if phase in [0, 1, 2, 3, 4, 5] else right_receive_rot6d[1]),
                float(right_wait_rot6d[2] if phase in [0, 1, 2, 3, 4, 5] else right_receive_rot6d[2]),
                float(right_wait_rot6d[3] if phase in [0, 1, 2, 3, 4, 5] else right_receive_rot6d[3]),
                float(right_wait_rot6d[4] if phase in [0, 1, 2, 3, 4, 5] else right_receive_rot6d[4]),
                float(right_wait_rot6d[5] if phase in [0, 1, 2, 3, 4, 5] else right_receive_rot6d[5]),
                float(right_grip_cmd),
            ])

            progress_metric = l2(info["left_pos"], left_target)
            progress_hist.append(progress_metric)

            should_stop, fail_reason = check_episode_termination(
                step=step,
                max_steps=MAX_STEPS,
                progress_hist=progress_hist,
                phase=phase,
                left_grasped=1.0 if left_grip_cmd >= LEFT_GRIP_CLOSE else 0.0,
                right_grasped=1.0 if right_grip_cmd >= RIGHT_GRIP_CLOSE else 0.0,
                left_tcp_pos=info["left_pos"],
                right_tcp_pos=info["right_pos"],
                obj_pos=obj_pos,
            )
            if should_stop:
                print(f"[FAIL] episode stop: {fail_reason}")
                break

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

                if phase == 4:
                    print("phase4_z_only target =", np.round(left_target, 4))
                    print("phase4_ori_mode      = fixed")

            # left arm
            if phase in [0, 1, 2, 4, 5, 6, 7, 9, 10, 11]:
                if phase == 4:
                    left_ori = LEFT_TOP_ORI
                else:
                    left_ori = pred_left_quat

                left_action = left_ctrl.forward(
                    target_end_effector_position=left_target,
                    target_end_effector_orientation=left_ori,
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

            # right arm expert
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

    finally:
        csv_file.close()

    if success:
        print(f"[SAVE] success episode saved to {csv_path}")
    else:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        print(f"[DELETE] failed episode removed: {csv_path}")

# =========================
# main
# =========================
for ep in range(NUM_EPISODES):
    print(f"\n{'#'*30} EPISODE {ep} {'#'*30}")
    run_one_episode(ep)

simulation_app.close()