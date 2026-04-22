import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(SCRIPT_DIR, "ml")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

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
# Paths
# =========================
CSV_PATH = "/home/june/bimanual_handover_IL/data/merged/handover_merged_round1.csv"
MODEL_PATH = "/home/june/bimanual_handover_IL/checkpoints/bc_rot6d_round1/best.pt"
STATE_STATS_PATH = "/home/june/bimanual_handover_IL/checkpoints/bc_rot6d_round1/state_norm_stats.json"
ACTION_STATS_PATH = "/home/june/bimanual_handover_IL/checkpoints/bc_rot6d_round1/action_norm_stats.json"

OUT_DIR = "/home/june/bimanual_handover_IL/checkpoints/bc_rot6d_round1/eval"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_PHASES = [4, 5, 6, 7, 8]

# =========================
# Helpers
# =========================
def build_state_tensor(df, state_mean, state_std):
    state_cont = df[STATE_CONT_COLS].values.astype(np.float32)
    state_bin = df[STATE_BIN_COLS].values.astype(np.float32)
    phase = df[PHASE_COLS].values.astype(np.float32)

    state_cont_norm = (state_cont - state_mean) / state_std
    state = np.concatenate([state_cont_norm, state_bin, phase], axis=1).astype(np.float32)
    return state


def denorm_action(action_norm, action_mean, action_std):
    return action_norm * action_std + action_mean


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def row_l2(a, b):
    return np.linalg.norm(a - b, axis=1)


def safe_clip(x, eps=1e-8):
    return np.clip(x, -1.0 + eps, 1.0 - eps)


def rot6d_to_rotmat_np(rot6d):
    """
    rot6d: (N, 6) or (6,)
    column-wise 6D rotation representation
    """
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.ndim == 1:
        rot6d = rot6d[None, :]

    a1 = rot6d[:, 0:3]
    a2 = rot6d[:, 3:6]

    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True).clip(min=1e-8)
    proj = np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True).clip(min=1e-8)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=2)  # (N, 3, 3)
    return R


def geodesic_deg_from_rot6d(pred_rot6d, gt_rot6d):
    R_pred = rot6d_to_rotmat_np(pred_rot6d)
    R_gt = rot6d_to_rotmat_np(gt_rot6d)

    R_rel = np.matmul(np.transpose(R_pred, (0, 2, 1)), R_gt)
    trace = np.trace(R_rel, axis1=1, axis2=2)
    cos_theta = safe_clip((trace - 1.0) / 2.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def save_phase_summary_table(df_phase, out_csv):
    rows = []

    for p in TARGET_PHASES:
        sub = df_phase[df_phase["phase"] == p].copy()
        if len(sub) == 0:
            continue

        pos_total_mae = float(np.mean(sub["left_pos_total_err"].values))
        x_mae = float(np.mean(np.abs(sub["pred_left_pos_x"].values - sub["action_left_pos_x"].values)))
        y_mae = float(np.mean(np.abs(sub["pred_left_pos_y"].values - sub["action_left_pos_y"].values)))
        z_mae = float(np.mean(np.abs(sub["pred_left_pos_z"].values - sub["action_left_pos_z"].values)))
        rot_deg_mean = float(np.mean(sub["left_rot_deg_err"].values))
        grip_acc = float(np.mean((sub["pred_left_grip_bin"].values == sub["action_left_grip_cmd"].values).astype(np.float32)))

        rows.append({
            "phase": int(p),
            "count": int(len(sub)),
            "pos_total_mae_m": pos_total_mae,
            "x_mae_m": x_mae,
            "y_mae_m": y_mae,
            "z_mae_m": z_mae,
            "rot_mean_deg": rot_deg_mean,
            "left_grip_acc": grip_acc,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_csv, index=False)
    return summary_df


def make_figure1(df_phase, out_png):
    fig, axes = plt.subplots(5, 3, figsize=(14, 22))
    phase_to_row = {4: 0, 5: 1, 6: 2, 7: 3, 8: 4}

    axis_specs = [
        ("x", "action_left_pos_x", "pred_left_pos_x"),
        ("y", "action_left_pos_y", "pred_left_pos_y"),
        ("z", "action_left_pos_z", "pred_left_pos_z"),
    ]

    for p in TARGET_PHASES:
        sub = df_phase[df_phase["phase"] == p].copy()
        if len(sub) == 0:
            continue

        r = phase_to_row[p]

        for c, (axis_name, gt_col, pred_col) in enumerate(axis_specs):
            ax = axes[r, c]

            x = sub[gt_col].values
            y = sub[pred_col].values

            axis_mae = float(np.mean(np.abs(y - x)))

            mn = min(np.min(x), np.min(y))
            mx = max(np.max(x), np.max(y))

            ax.scatter(x, y, s=8, alpha=0.35)
            ax.plot([mn, mx], [mn, mx], linestyle="--")

            ax.set_xlabel(f"GT {axis_name}")
            ax.set_ylabel(f"Pred {axis_name}")
            ax.set_title(f"Phase {p} - {axis_name.upper()} (MAE={axis_mae:.4f})")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()


def main():
    print("[INFO] loading merged csv")
    df = pd.read_csv(CSV_PATH)

    print("[INFO] loading norm stats")
    state_cols_loaded, state_mean, state_std = load_norm_stats(STATE_STATS_PATH)
    action_cols_loaded, action_mean, action_std = load_norm_stats(ACTION_STATS_PATH)

    assert state_cols_loaded == STATE_CONT_COLS, (
        f"STATE_CONT_COLS mismatch\nloaded={state_cols_loaded}\nexpected={STATE_CONT_COLS}"
    )
    assert action_cols_loaded == ARM_ACTION_CONT_COLS, (
        f"ARM_ACTION_CONT_COLS mismatch\nloaded={action_cols_loaded}\nexpected={ARM_ACTION_CONT_COLS}"
    )

    state_dim = len(STATE_CONT_COLS) + len(STATE_BIN_COLS) + len(PHASE_COLS)
    arm_dim = len(ARM_ACTION_CONT_COLS)
    grip_dim = len(GRIP_ACTION_BIN_COLS)

    print("[INFO] loading model")
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

    print("[INFO] building state tensor")
    state_np = build_state_tensor(df, state_mean, state_std)
    x = torch.from_numpy(state_np).float().to(DEVICE)

    print("[INFO] running offline inference")
    with torch.no_grad():
        pred_arm_norm, pred_grip_logit = model(x)
        pred_arm_norm = pred_arm_norm.cpu().numpy()
        pred_grip_logit = pred_grip_logit.cpu().numpy()

    pred_arm = denorm_action(pred_arm_norm, action_mean, action_std)

    # GT
    gt_arm = df[ARM_ACTION_CONT_COLS].values.astype(np.float32)
    gt_grip = df[GRIP_ACTION_BIN_COLS].values.astype(np.float32)

    # Split arm action
    gt_left_pos = gt_arm[:, 0:3]
    gt_left_rot6d = gt_arm[:, 3:9]
    gt_right_pos = gt_arm[:, 9:12]
    gt_right_rot6d = gt_arm[:, 12:18]

    pred_left_pos = pred_arm[:, 0:3]
    pred_left_rot6d = pred_arm[:, 3:9]
    pred_right_pos = pred_arm[:, 9:12]
    pred_right_rot6d = pred_arm[:, 12:18]

    # Grip
    pred_grip_prob = sigmoid(pred_grip_logit)
    pred_grip_bin = (pred_grip_prob >= 0.5).astype(np.float32)

    # Errors
    left_pos_total_err = row_l2(pred_left_pos, gt_left_pos)
    right_pos_total_err = row_l2(pred_right_pos, gt_right_pos)

    left_rot_deg_err = geodesic_deg_from_rot6d(pred_left_rot6d, gt_left_rot6d)
    right_rot_deg_err = geodesic_deg_from_rot6d(pred_right_rot6d, gt_right_rot6d)

    # Attach prediction/eval columns
    result_df = df.copy()

    result_df["pred_left_pos_x"] = pred_left_pos[:, 0]
    result_df["pred_left_pos_y"] = pred_left_pos[:, 1]
    result_df["pred_left_pos_z"] = pred_left_pos[:, 2]

    result_df["pred_right_pos_x"] = pred_right_pos[:, 0]
    result_df["pred_right_pos_y"] = pred_right_pos[:, 1]
    result_df["pred_right_pos_z"] = pred_right_pos[:, 2]

    for i in range(6):
        result_df[f"pred_left_rot6d_{i}"] = pred_left_rot6d[:, i]
        result_df[f"pred_right_rot6d_{i}"] = pred_right_rot6d[:, i]

    result_df["pred_left_grip_logit"] = pred_grip_logit[:, 0]
    result_df["pred_right_grip_logit"] = pred_grip_logit[:, 1]
    result_df["pred_left_grip_prob"] = pred_grip_prob[:, 0]
    result_df["pred_right_grip_prob"] = pred_grip_prob[:, 1]
    result_df["pred_left_grip_bin"] = pred_grip_bin[:, 0]
    result_df["pred_right_grip_bin"] = pred_grip_bin[:, 1]

    result_df["left_pos_total_err"] = left_pos_total_err
    result_df["right_pos_total_err"] = right_pos_total_err
    result_df["left_rot_deg_err"] = left_rot_deg_err
    result_df["right_rot_deg_err"] = right_rot_deg_err

    # Filter key phases only
    key_df = result_df[result_df["phase"].isin(TARGET_PHASES)].copy()

    # Save detailed rows for key phases
    key_rows_csv = os.path.join(OUT_DIR, "eval_rows_phase458.csv")
    key_df.to_csv(key_rows_csv, index=False)

    # Save summary table
    table1_csv = os.path.join(OUT_DIR, "table1_phase458_summary.csv")
    table1_df = save_phase_summary_table(key_df, table1_csv)

    # Make figure 1
    figure1_png = os.path.join(OUT_DIR, "figure1_gt_vs_pred_xyz_phase458.png")
    make_figure1(key_df, figure1_png)

    # Console summary
    print("\n===== TABLE 1: OFFLINE PREDICTION SUMMARY (PHASE 4/5/6/7/8) =====")
    print(table1_df.to_string(index=False))

    print("\n[INFO] saved files:")
    print(key_rows_csv)
    print(table1_csv)
    print(figure1_png)


if __name__ == "__main__":
    main()