from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CKPT_DIR = PROJECT_ROOT / "checkpoints" / "bc_single_head"
CKPT_PATH = CKPT_DIR / "best_model.pt"

DATA_DIR = PROJECT_ROOT / "data" / "bc_ready"
TEST_CSV = DATA_DIR / "bc_test.csv"
META_PATH = DATA_DIR / "meta.json"

OUT_DIR = CKPT_DIR / "eval_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Model
# =========================================================
class MLPBCPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        dropout_p: float = 0.1,
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        out = self.head(h)
        return out


# =========================================================
# Utils
# =========================================================
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(device: str):
    ckpt = torch.load(CKPT_PATH, map_location=device)

    input_dim = ckpt["input_dim"]
    output_dim = ckpt["output_dim"]
    hidden_dims = tuple(ckpt["hidden_dims"])
    dropout_p = ckpt["dropout_p"]

    model = MLPBCPolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout_p=dropout_p,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def compute_errors(pred: np.ndarray, target: np.ndarray):
    abs_err = np.abs(pred - target)
    sq_err = (pred - target) ** 2
    return abs_err, sq_err


# =========================================================
# Main
# =========================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    if not TEST_CSV.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta JSON not found: {META_PATH}")

    meta = load_json(META_PATH)
    state_cols: List[str] = meta["state_cols_final"]
    action_cols: List[str] = meta["action_cols_final"]

    df = pd.read_csv(TEST_CSV)

    missing_state = [c for c in state_cols if c not in df.columns]
    missing_action = [c for c in action_cols if c not in df.columns]

    if missing_state:
        raise ValueError(f"Missing state columns: {missing_state}")
    if missing_action:
        raise ValueError(f"Missing action columns: {missing_action}")

    for c in state_cols + action_cols:
        df[c] = pd.to_numeric(df[c], errors="raise")

    x = df[state_cols].to_numpy(dtype=np.float32)
    y = df[action_cols].to_numpy(dtype=np.float32)

    model, ckpt = load_model(device)

    with torch.no_grad():
        x_tensor = torch.from_numpy(x).to(device)
        pred = model(x_tensor).cpu().numpy()

    abs_err, sq_err = compute_errors(pred, y)

    # -----------------------------------------------------
    # 1. sample-wise compare table
    # -----------------------------------------------------
    compare_rows = []
    meta_cols = ["episode_id", "original_file", "step", "phase", "phase_name", "episode_success"]
    existing_meta_cols = [c for c in meta_cols if c in df.columns]

    for i in range(len(df)):
        row = {}

        for c in existing_meta_cols:
            row[c] = df.iloc[i][c]

        for j, col in enumerate(action_cols):
            row[f"gt__{col}"] = float(y[i, j])
            row[f"pred__{col}"] = float(pred[i, j])
            row[f"abs_err__{col}"] = float(abs_err[i, j])
            row[f"sq_err__{col}"] = float(sq_err[i, j])

        row["abs_err_mean_all"] = float(abs_err[i].mean())
        row["mse_all"] = float(sq_err[i].mean())
        row["abs_err_mean_arm"] = float(abs_err[i, :6].mean())
        row["abs_err_mean_gripper"] = float(abs_err[i, 6:].mean())

        compare_rows.append(row)

    compare_df = pd.DataFrame(compare_rows)
    compare_csv_path = OUT_DIR / "test_predictions_compare.csv"
    compare_df.to_csv(compare_csv_path, index=False)

    # -----------------------------------------------------
    # 2. dimension-wise summary
    # -----------------------------------------------------
    summary_rows = []
    for j, col in enumerate(action_cols):
        summary_rows.append({
            "action_col": col,
            "mae": float(abs_err[:, j].mean()),
            "mse": float(sq_err[:, j].mean()),
            "rmse": float(np.sqrt(sq_err[:, j].mean())),
            "pred_mean": float(pred[:, j].mean()),
            "gt_mean": float(y[:, j].mean()),
            "pred_std": float(pred[:, j].std()),
            "gt_std": float(y[:, j].std()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = OUT_DIR / "test_dim_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    # -----------------------------------------------------
    # 3. top good / bad samples
    # -----------------------------------------------------
    best_samples = compare_df.nsmallest(20, "mse_all")
    worst_samples = compare_df.nlargest(20, "mse_all")

    best_path = OUT_DIR / "best_20_samples.csv"
    worst_path = OUT_DIR / "worst_20_samples.csv"

    best_samples.to_csv(best_path, index=False)
    worst_samples.to_csv(worst_path, index=False)

    # -----------------------------------------------------
    # 4. console print
    # -----------------------------------------------------
    print("=" * 90)
    print("EVAL DONE")
    print(f"checkpoint        : {CKPT_PATH}")
    print(f"test_csv          : {TEST_CSV}")
    print(f"num_test_rows     : {len(df)}")
    print("=" * 90)

    print("[Overall]")
    print(f"Mean MAE all      : {abs_err.mean():.6f}")
    print(f"Mean MSE all      : {sq_err.mean():.6f}")
    print(f"Mean MAE arm      : {abs_err[:, :6].mean():.6f}")
    print(f"Mean MAE gripper  : {abs_err[:, 6:].mean():.6f}")
    print()

    print("[Per-dimension summary]")
    print(summary_df.to_string(index=False))
    print()

    print("[Saved files]")
    print(f"- {compare_csv_path}")
    print(f"- {summary_csv_path}")
    print(f"- {best_path}")
    print(f"- {worst_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()