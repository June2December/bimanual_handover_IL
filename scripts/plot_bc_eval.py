
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_loss_curves(history_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(history_csv)

    if "epoch" not in df.columns:
        raise ValueError("train_history.csv must contain 'epoch' column")

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("BC Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_total.png", dpi=150)
    plt.close()

    if {"train_arm_loss", "val_arm_loss"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["train_arm_loss"], label="train_arm_loss")
        plt.plot(df["epoch"], df["val_arm_loss"], label="val_arm_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("BC Arm Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_arm.png", dpi=150)
        plt.close()

    if {"train_gripper_loss", "val_gripper_loss"}.issubset(df.columns):
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["train_gripper_loss"], label="train_gripper_loss")
        plt.plot(df["epoch"], df["val_gripper_loss"], label="val_gripper_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("BC Gripper Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_gripper.png", dpi=150)
        plt.close()


def plot_dim_summary(summary_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(summary_csv)

    required = {"action_col", "mae", "rmse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"test_dim_summary.csv missing columns: {sorted(missing)}")

    plt.figure(figsize=(12, 6))
    plt.bar(df["action_col"], df["mae"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("MAE")
    plt.title("Per-dimension MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "dim_mae.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(df["action_col"], df["rmse"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("RMSE")
    plt.title("Per-dimension RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "dim_rmse.png", dpi=150)
    plt.close()


def plot_sample_error(compare_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(compare_csv)

    if "mse_all" not in df.columns:
        raise ValueError("test_predictions_compare.csv must contain 'mse_all' column")

    x = np.arange(len(df))

    plt.figure(figsize=(12, 6))
    plt.plot(x, df["mse_all"])
    plt.xlabel("sample_index")
    plt.ylabel("mse_all")
    plt.title("Per-sample MSE")
    plt.tight_layout()
    plt.savefig(out_dir / "sample_mse_all.png", dpi=150)
    plt.close()

    worst = df.nlargest(20, "mse_all").copy()
    worst["sample_label"] = worst.index.astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(worst["sample_label"], worst["mse_all"])
    plt.xlabel("sample_index")
    plt.ylabel("mse_all")
    plt.title("Top 20 Worst Samples")
    plt.tight_layout()
    plt.savefig(out_dir / "worst_20_mse.png", dpi=150)
    plt.close()

    best = df.nsmallest(20, "mse_all").copy()
    best["sample_label"] = best.index.astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(best["sample_label"], best["mse_all"])
    plt.xlabel("sample_index")
    plt.ylabel("mse_all")
    plt.title("Top 20 Best Samples")
    plt.tight_layout()
    plt.savefig(out_dir / "best_20_mse.png", dpi=150)
    plt.close()

    if "phase_name" in df.columns:
        phase_df = df.groupby("phase_name", dropna=False)["mse_all"].mean().reset_index()
        plt.figure(figsize=(12, 6))
        plt.bar(phase_df["phase_name"].astype(str), phase_df["mse_all"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mean mse_all")
        plt.title("Phase-wise Mean MSE")
        plt.tight_layout()
        plt.savefig(out_dir / "phase_mean_mse.png", dpi=150)
        plt.close()

    if "episode_id" in df.columns:
        ep_df = df.groupby("episode_id", dropna=False)["mse_all"].mean().reset_index()
        ep_df = ep_df.sort_values("mse_all", ascending=False).head(20)
        plt.figure(figsize=(12, 6))
        plt.bar(ep_df["episode_id"].astype(str), ep_df["mse_all"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mean mse_all")
        plt.title("Top 20 Worst Episodes by Mean MSE")
        plt.tight_layout()
        plt.savefig(out_dir / "worst_episodes_mean_mse.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="/home/june/bimanual_handover_IL/checkpoints/bc_single_head/eval_samples",
    )
    parser.add_argument(
        "--history_csv",
        type=str,
        default="/home/june/bimanual_handover_IL/checkpoints/bc_single_head/train_history.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    history_csv = Path(args.history_csv)
    out_dir = Path(args.out_dir) if args.out_dir else (eval_dir / "plots")
    ensure_dir(out_dir)

    compare_csv = eval_dir / "test_predictions_compare.csv"
    summary_csv = eval_dir / "test_dim_summary.csv"

    if history_csv.exists():
        plot_loss_curves(history_csv, out_dir)
        print(f"[OK] history plots: {out_dir}")
    else:
        print(f"[SKIP] history csv not found: {history_csv}")

    if summary_csv.exists():
        plot_dim_summary(summary_csv, out_dir)
        print(f"[OK] dim plots: {out_dir}")
    else:
        print(f"[SKIP] summary csv not found: {summary_csv}")

    if compare_csv.exists():
        plot_sample_error(compare_csv, out_dir)
        print(f"[OK] sample plots: {out_dir}")
    else:
        print(f"[SKIP] compare csv not found: {compare_csv}")

    print(f"[DONE] output dir: {out_dir}")


if __name__ == "__main__":
    main()
