from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "bc_ready"
META_PATH = DATA_DIR / "meta.json"

TRAIN_CSV = DATA_DIR / "bc_train.csv"
VAL_CSV = DATA_DIR / "bc_val.csv"
TEST_CSV = DATA_DIR / "bc_test.csv"

SAVE_DIR = PROJECT_ROOT / "checkpoints" / "bc_single_head"

# =========================================================
# Config
# =========================================================
@dataclass
class TrainConfig:
    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5

    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    dropout_p: float = 0.1

    lambda_gripper: float = 1.0

    num_workers: int = 4
    seed: int = 42
    patience: int = 15
    grad_clip_norm: float = 5.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = TrainConfig()

# =========================================================
# Utils
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

# =========================================================
# Dataset
# =========================================================
class BCDataset(Dataset):
    def __init__(self, csv_path: Path, state_cols: List[str], action_cols: List[str]):
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)

        self.state_cols = state_cols
        self.action_cols = action_cols

        missing_state = [c for c in state_cols if c not in self.df.columns]
        missing_action = [c for c in action_cols if c not in self.df.columns]

        if missing_state:
            raise ValueError(f"Missing state columns in {csv_path.name}: {missing_state}")
        if missing_action:
            raise ValueError(f"Missing action columns in {csv_path.name}: {missing_action}")

        for c in state_cols + action_cols:
            self.df[c] = pd.to_numeric(self.df[c], errors="raise")

        self.x = self.df[state_cols].to_numpy(dtype=np.float32)
        self.y = self.df[action_cols].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "state": torch.from_numpy(self.x[idx]),
            "action": torch.from_numpy(self.y[idx]),
        }


# =========================================================
# Model
# =========================================================
class MLPBCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims: Tuple[int, ...] = (256, 256, 128), dropout_p: float = 0.1,):
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

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        out = self.head(h)
        return out

# =========================================================
# Loss
# =========================================================
class BCActionLoss(nn.Module):
    """
    single-head 출력(8차원)을 그대로 쓰되,
    loss는 arm(앞 6차원) / gripper(뒤 2차원)로 나눠서 계산.
    """
    def __init__(self, lambda_gripper: float = 1.0):
        super().__init__()
        self.lambda_gripper = lambda_gripper
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_arm = pred[:, :6]
        target_arm = target[:, :6]

        pred_gripper = pred[:, 6:]
        target_gripper = target[:, 6:]

        loss_arm = self.mse(pred_arm, target_arm)
        loss_gripper = self.mse(pred_gripper, target_gripper)

        loss = loss_arm + self.lambda_gripper * loss_gripper
        return loss, loss_arm.detach(), loss_gripper.detach()

# =========================================================
# Train / Eval
# =========================================================
def run_one_epoch(model, loader, optimizer, criterion, device, train: bool = True, grad_clip_norm: float | None = None,) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_arm_loss = 0.0
    total_gripper_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["state"].to(device)
        y = batch["action"].to(device)

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss, loss_arm, loss_gripper = criterion(pred, y)

            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()

                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_arm_loss += loss_arm.item() * bs
        total_gripper_loss += loss_gripper.item() * bs
        total_count += bs

    return {
        "loss": total_loss / max(total_count, 1),
        "arm_loss": total_arm_loss / max(total_count, 1),
        "gripper_loss": total_gripper_loss / max(total_count, 1),
    }

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: BCActionLoss, device: str,) -> Dict[str, float]:
    return run_one_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
        grad_clip_norm=None,
    )

# =========================================================
# Main
# =========================================================
def main() -> None:
    set_seed(CFG.seed)
    ensure_dir(SAVE_DIR)

    if not META_PATH.exists():
        raise FileNotFoundError(f"meta.json not found: {META_PATH}")

    meta = load_json(META_PATH)
    state_cols = meta["state_cols_final"]
    action_cols = meta["action_cols_final"]

    if len(action_cols) != 8:
        raise ValueError(
            f"Expected 8 action dims, but got {len(action_cols)}. "
            f"action_cols_final={action_cols}"
        )

    train_dataset = BCDataset(TRAIN_CSV, state_cols, action_cols)
    val_dataset = BCDataset(VAL_CSV, state_cols, action_cols)
    test_dataset = BCDataset(TEST_CSV, state_cols, action_cols)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available(),)
    val_loader = DataLoader(
        val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available(),)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=torch.cuda.is_available(),)

    input_dim = len(state_cols)
    output_dim = len(action_cols)

    model = MLPBCPolicy(
        input_dim=input_dim, output_dim=output_dim,
        hidden_dims=CFG.hidden_dims, dropout_p=CFG.dropout_p,).to(CFG.device)

    criterion = BCActionLoss(lambda_gripper=CFG.lambda_gripper)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    train_meta = {
        "config": asdict(CFG),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "state_cols": state_cols,
        "action_cols": action_cols,
    }
    (SAVE_DIR / "train_meta.json").write_text(
        json.dumps(train_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_val_loss = math.inf
    best_epoch = -1
    patience_count = 0
    history = []

    print("=" * 90)
    print("BC TRAIN START")
    print(f"device      : {CFG.device}")
    print(f"input_dim   : {input_dim}")
    print(f"output_dim  : {output_dim}")
    print(f"train size  : {len(train_dataset)}")
    print(f"val size    : {len(val_dataset)}")
    print(f"test size   : {len(test_dataset)}")
    print("=" * 90)

    for epoch in range(1, CFG.num_epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=CFG.device,
            train=True,
            grad_clip_norm=CFG.grad_clip_norm,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=CFG.device,
            train=False,
            grad_clip_norm=None,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_arm_loss": train_metrics["arm_loss"],
            "train_gripper_loss": train_metrics["gripper_loss"],
            "val_loss": val_metrics["loss"],
            "val_arm_loss": val_metrics["arm_loss"],
            "val_gripper_loss": val_metrics["gripper_loss"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train={train_metrics['loss']:.6f} "
            f"(arm={train_metrics['arm_loss']:.6f}, grip={train_metrics['gripper_loss']:.6f}) | "
            f"val={val_metrics['loss']:.6f} "
            f"(arm={val_metrics['arm_loss']:.6f}, grip={val_metrics['gripper_loss']:.6f})"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_count = 0

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dims": CFG.hidden_dims,
                "dropout_p": CFG.dropout_p,
                "state_cols": state_cols,
                "action_cols": action_cols,
            }
            torch.save(ckpt, SAVE_DIR / "best_model.pt")
        else:
            patience_count += 1

        pd.DataFrame(history).to_csv(SAVE_DIR / "train_history.csv", index=False)

        if patience_count >= CFG.patience:
            print(f"[Early Stop] no improvement for {CFG.patience} epochs.")
            break

    print("=" * 90)
    print(f"BEST EPOCH    : {best_epoch}")
    print(f"BEST VAL LOSS : {best_val_loss:.6f}")
    print("=" * 90)

    ckpt_path = SAVE_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=CFG.device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=CFG.device,
    )

    print("[TEST RESULT]")
    print(
        f"test={test_metrics['loss']:.6f} "
        f"(arm={test_metrics['arm_loss']:.6f}, grip={test_metrics['gripper_loss']:.6f})"
    )

    (SAVE_DIR / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Training finished.")


if __name__ == "__main__":
    main()