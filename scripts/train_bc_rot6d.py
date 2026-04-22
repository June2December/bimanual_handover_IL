import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.columns import *
from ml.dataset import BCDataset
from ml.model import BCPolicy
from ml.norm_utils import compute_mean_std, save_norm_stats

import matplotlib.pyplot as plt

CSV_PATH = "/home/june/bimanual_handover_IL/data/merged/handover_merged_round3.csv"
CKPT_DIR = "/home/june/bimanual_handover_IL/checkpoints/bc_rot6d_round3"

BATCH_SIZE = 512
EPOCHS = 200
LR = 1e-3

# early stopping
PATIENCE = 30
MIN_DELTA = 1e-5

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_by_episode(df):
    """
    - 원본(non-dagger) episode만 train/val/test split
    - dagger row는 전부 train에만 추가
    """
    episode_str = df["episode_id"].astype(str)
    dagger_mask = episode_str.str.contains("dagger", na=False)

    df_dagger = df[dagger_mask].copy()
    df_base = df[~dagger_mask].copy()

    eps = df_base["episode_id"].astype(str).unique()
    np.random.shuffle(eps)

    n = len(eps)
    train_eps = eps[:int(0.7 * n)]
    val_eps = eps[int(0.7 * n):int(0.85 * n)]
    test_eps = eps[int(0.85 * n):]

    df_train_base = df_base[df_base["episode_id"].astype(str).isin(train_eps)].copy()
    df_val = df_base[df_base["episode_id"].astype(str).isin(val_eps)].copy()
    df_test = df_base[df_base["episode_id"].astype(str).isin(test_eps)].copy()

    # dagger는 train에만 추가
    df_train = pd.concat([df_train_base, df_dagger], ignore_index=True)

    return df_train, df_val, df_test

def run_epoch(model, loader, opt=None):
    model.train() if opt is not None else model.eval()

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    total = 0.0
    count = 0

    for b in loader:
        s = b["state"].to(DEVICE)
        a = b["arm_action"].to(DEVICE)
        g = b["grip_action"].to(DEVICE)

        with torch.set_grad_enabled(opt is not None):
            a_pred, g_pred = model(s)

            loss_arm = mse(a_pred, a)
            loss_grip = bce(g_pred, g)
            loss = loss_arm + loss_grip

            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

        total += loss.item() * len(s)
        count += len(s)

    return total / count


def main():
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    df_train, df_val, df_test = split_by_episode(df)

    print(f"[INFO] total rows : {len(df)}")
    print(f"[INFO] train rows : {len(df_train)}")
    print(f"[INFO] val rows   : {len(df_val)}")
    print(f"[INFO] test rows  : {len(df_test)}")

    print(f"[INFO] train episodes: {df_train['episode_id'].nunique()}")
    print(f"[INFO] val episodes  : {df_val['episode_id'].nunique()}")
    print(f"[INFO] test episodes : {df_test['episode_id'].nunique()}")

    state_mean, state_std = compute_mean_std(df_train, STATE_CONT_COLS)
    act_mean, act_std = compute_mean_std(df_train, ARM_ACTION_CONT_COLS)

    save_norm_stats(
        os.path.join(CKPT_DIR, "state_norm_stats.json"),
        STATE_CONT_COLS,
        state_mean,
        state_std,
    )

    save_norm_stats(
        os.path.join(CKPT_DIR, "action_norm_stats.json"),
        ARM_ACTION_CONT_COLS,
        act_mean,
        act_std,
    )
    
    train_ds = BCDataset(
        df_train,
        STATE_CONT_COLS,
        STATE_BIN_COLS,
        PHASE_COLS,
        ARM_ACTION_CONT_COLS,
        GRIP_ACTION_BIN_COLS,
        state_mean,
        state_std,
        act_mean,
        act_std,
    )

    val_ds = BCDataset(
        df_val,
        STATE_CONT_COLS,
        STATE_BIN_COLS,
        PHASE_COLS,
        ARM_ACTION_CONT_COLS,
        GRIP_ACTION_BIN_COLS,
        state_mean,
        state_std,
        act_mean,
        act_std,
    )

    test_ds = BCDataset(
        df_test,
        STATE_CONT_COLS,
        STATE_BIN_COLS,
        PHASE_COLS,
        ARM_ACTION_CONT_COLS,
        GRIP_ACTION_BIN_COLS,
        state_mean,
        state_std,
        act_mean,
        act_std,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


    print("train dagger ids:", df_train[df_train["episode_id"].astype(str).str.contains("dagger", na=False)]["episode_id"].unique())
    print("val dagger ids:", df_val[df_val["episode_id"].astype(str).str.contains("dagger", na=False)]["episode_id"].unique())
    print("test dagger ids:", df_test[df_test["episode_id"].astype(str).str.contains("dagger", na=False)]["episode_id"].unique())
    model = BCPolicy(
        state_dim=len(STATE_COLS),
        arm_dim=len(ARM_ACTION_CONT_COLS),
        grip_dim=len(GRIP_ACTION_BIN_COLS),
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    for ep in range(EPOCHS):
        train_loss = run_epoch(model, train_loader, opt)
        val_loss = run_epoch(model, val_loader)

        history["epoch"].append(ep)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = (best_val - val_loss) > MIN_DELTA

        if improved:
            best_val = val_loss
            best_epoch = ep
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pt"))
            save_tag = "  <-- best"
        else:
            patience_counter += 1
            save_tag = ""

        print(
            f"[{ep:03d}] "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f} "
            f"patience={patience_counter}/{PATIENCE}"
            f"{save_tag}"
        )

        if patience_counter >= PATIENCE:
            print(f"[INFO] Early stopping triggered at epoch {ep}")
            break

    # 마지막 epoch 모델도 저장
    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last.pt"))

    # history 저장
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(CKPT_DIR, "train_history.csv"), index=False)

    # best 모델 다시 로드해서 val / test 확인
    best_model = BCPolicy(
        state_dim=len(STATE_COLS),
        arm_dim=len(ARM_ACTION_CONT_COLS),
        grip_dim=len(GRIP_ACTION_BIN_COLS),
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location=DEVICE))

    final_val_loss = run_epoch(best_model, val_loader)
    final_test_loss = run_epoch(best_model, test_loader)

    print("\n====================")
    print(f"[INFO] best_epoch     : {best_epoch}")
    print(f"[INFO] best_val_loss  : {best_val:.6f}")
    print(f"[INFO] final_val_loss : {final_val_loss:.6f}")
    print(f"[INFO] test_loss      : {final_test_loss:.6f}")
    print("====================\n")

    # 전체 loss curve
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("BC training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CKPT_DIR, "loss_curve.png"), dpi=150)
    plt.show()

    # 40 epoch 이후만 확대
    epochs_arr = np.array(history["epoch"])
    train_losses_arr = np.array(history["train_loss"])
    val_losses_arr = np.array(history["val_loss"])

    mask = epochs_arr >= 40

    if mask.any():
        plt.figure()
        plt.plot(epochs_arr[mask], train_losses_arr[mask], label="train")
        plt.plot(epochs_arr[mask], val_losses_arr[mask], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("BC training history (epoch >= 40)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CKPT_DIR, "loss_curve_after_40.png"), dpi=150)
        plt.show()

    print("done")


if __name__ == "__main__":
    main()