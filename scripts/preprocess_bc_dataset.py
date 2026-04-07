"""
BC preprocessing script for merged bimanual handover dataset.

Notes
- Raw CSV is never overwritten.
- Metadata columns are kept in output CSVs, but NOT used as model input.
- phase one-hot columns are detected from the CSV at runtime.
- Config JSON is the source of truth for base column groups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "bc_columns.json"
INPUT_CSV = PROJECT_ROOT / "data" / "merged_raw_success.csv"
OUT_DIR = PROJECT_ROOT / "data" / "bc_ready"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42


# =========================================================
# Helpers
# =========================================================
@dataclass
class NormStats:
    mean: Dict[str, float]
    std: Dict[str, float]
    columns: List[str]


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_standardization(df: pd.DataFrame, columns: List[str]) -> NormStats:
    mean = df[columns].mean(axis=0).to_dict()
    std = df[columns].std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0).to_dict()
    return NormStats(mean=mean, std=std, columns=columns)


def apply_standardization(df: pd.DataFrame, stats: NormStats) -> pd.DataFrame:
    out = df.copy()
    for c in stats.columns:
        out[c] = (out[c] - stats.mean[c]) / stats.std[c]
    return out


def split_episodes(
    episode_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    eps = np.array(sorted(set(episode_ids)))
    rng.shuffle(eps)

    n = len(eps)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_eps = eps[:n_train].tolist()
    val_eps = eps[n_train:n_train + n_val].tolist()
    test_eps = eps[n_train + n_val:].tolist()
    return train_eps, val_eps, test_eps


def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config JSON not found: {CONFIG_PATH}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1) Load config
    # -----------------------------------------------------
    config = load_json(CONFIG_PATH)

    meta_cols = config["meta_cols"]
    state_continuous_cols = config["state_continuous_cols"]
    state_binary_cols = config["state_binary_cols"]
    action_continuous_cols = config["action_continuous_cols"]
    action_binary_cols = config["action_binary_cols"]

    required_base_cols = (
        meta_cols[:-1]  # episode_success is not expected in raw csv
        + state_continuous_cols
        + state_binary_cols
        + action_continuous_cols
        + action_binary_cols
    )

    # -----------------------------------------------------
    # 2) Load raw CSV
    # -----------------------------------------------------
    df = pd.read_csv(INPUT_CSV)

    missing = [c for c in required_base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw CSV: {missing}")

    df = df.copy()

    # -----------------------------------------------------
    # 3) Type cleanup
    # -----------------------------------------------------
    df["episode_id"] = df["episode_id"].astype(str)
    df["original_file"] = df["original_file"].astype(str)
    df["phase_name"] = df["phase_name"].astype(str)

    numeric_cols = (
        ["step", "phase"]
        + state_continuous_cols
        + state_binary_cols
        + action_continuous_cols
        + action_binary_cols
    )

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # success-only merged file: keep this simple
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    df["step"] = df["step"].astype(int)
    df["phase"] = df["phase"].astype(int)

    df = df.sort_values(["episode_id", "step"]).reset_index(drop=True)

    # -----------------------------------------------------
    # 4) Add success label
    # -----------------------------------------------------
    df["episode_success"] = 1

    # -----------------------------------------------------
    # 5) Dynamic phase one-hot from actual CSV values
    # -----------------------------------------------------
    phase_dummies = pd.get_dummies(df["phase"], prefix="phase")
    phase_onehot_cols = phase_dummies.columns.tolist()
    df = pd.concat([df, phase_dummies], axis=1)

    # -----------------------------------------------------
    # 6) Final state/action columns
    # -----------------------------------------------------
    state_cols_final = state_continuous_cols + state_binary_cols + phase_onehot_cols
    action_cols_final = action_continuous_cols + action_binary_cols

    final_cols = (
        ["episode_id", "original_file", "step", "phase", "phase_name", "episode_success"]
        + state_cols_final
        + action_cols_final
    )
    df = df[final_cols].copy()

    # -----------------------------------------------------
    # 7) Split by episode
    # -----------------------------------------------------
    train_eps, val_eps, test_eps = split_episodes(
        episode_ids=df["episode_id"].unique().tolist(),
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
    )

    train_df = df[df["episode_id"].isin(train_eps)].copy()
    val_df = df[df["episode_id"].isin(val_eps)].copy()
    test_df = df[df["episode_id"].isin(test_eps)].copy()

    # -----------------------------------------------------
    # 8) Standardize continuous columns (fit on train only)
    # -----------------------------------------------------
    state_stats = fit_standardization(train_df, state_continuous_cols)
    action_stats = fit_standardization(train_df, action_continuous_cols)

    train_df = apply_standardization(train_df, state_stats)
    val_df = apply_standardization(val_df, state_stats)
    test_df = apply_standardization(test_df, state_stats)

    train_df = apply_standardization(train_df, action_stats)
    val_df = apply_standardization(val_df, action_stats)
    test_df = apply_standardization(test_df, action_stats)

    # -----------------------------------------------------
    # 9) Save outputs
    # -----------------------------------------------------
    train_df.to_csv(OUT_DIR / "bc_train.csv", index=False)
    val_df.to_csv(OUT_DIR / "bc_val.csv", index=False)
    test_df.to_csv(OUT_DIR / "bc_test.csv", index=False)

    save_json(asdict(state_stats), OUT_DIR / "state_norm_stats.json")
    save_json(asdict(action_stats), OUT_DIR / "action_norm_stats.json")
    save_json({"train_episode_ids": train_eps}, OUT_DIR / "train_episodes.json")
    save_json({"val_episode_ids": val_eps}, OUT_DIR / "val_episodes.json")
    save_json({"test_episode_ids": test_eps}, OUT_DIR / "test_episodes.json")

    # meta for downstream BC training
    meta = {
        "config_path": str(CONFIG_PATH),
        "input_csv": str(INPUT_CSV),
        "output_dir": str(OUT_DIR),
        "n_rows_total": int(len(df)),
        "n_episodes_total": int(df["episode_id"].nunique()),
        "n_rows_train": int(len(train_df)),
        "n_rows_val": int(len(val_df)),
        "n_rows_test": int(len(test_df)),
        "n_episodes_train": int(train_df["episode_id"].nunique()),
        "n_episodes_val": int(val_df["episode_id"].nunique()),
        "n_episodes_test": int(test_df["episode_id"].nunique()),
        "meta_cols": ["episode_id", "original_file", "step", "phase", "phase_name", "episode_success"],
        "state_continuous_cols": state_continuous_cols,
        "state_binary_cols": state_binary_cols,
        "phase_onehot_cols": phase_onehot_cols,
        "action_continuous_cols": action_continuous_cols,
        "action_binary_cols": action_binary_cols,
        "state_cols_final": state_cols_final,
        "action_cols_final": action_cols_final,
        "success_rule": "all rows set to episode_success = 1 because input file is merged_raw_success.csv",
    }
    save_json(meta, OUT_DIR / "meta.json")

    print("=== DONE ===")
    print(f"Config JSON: {CONFIG_PATH}")
    print(f"Input CSV:   {INPUT_CSV}")
    print(f"Output dir:  {OUT_DIR}")
    print(f"Total rows: {len(df)}")
    print(f"Total episodes: {df['episode_id'].nunique()}")
    print(f"Train/Val/Test rows: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Train/Val/Test episodes: {train_df['episode_id'].nunique()} / {val_df['episode_id'].nunique()} / {test_df['episode_id'].nunique()}")
    print(f"Detected phase one-hot cols: {phase_onehot_cols}")


if __name__ == "__main__":
    main()