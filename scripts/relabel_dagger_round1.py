import pandas as pd
from pathlib import Path

# =========================
# 경로 (고정)
# =========================
ROLLOUT_PATH = Path("/home/june/bimanual_handover_IL/data/rollout_bc_rot6d/phase4_zonly_ori_pred_1776827506.csv")
BASE_MERGED_PATH = Path("/home/june/bimanual_handover_IL/data/merged/handover_merged.csv")
SAVE_DIR = Path("/home/june/bimanual_handover_IL/data/merged")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 설정
# =========================
TARGET_PHASES = [4, 5, 6, 7, 8]

# expert target
LEFT_LIFT_Z = 0.50
HANDOVER_X = 0.50
HANDOVER_Y = 0.00
HANDOVER_Z = 0.47

# =========================
# load
# =========================
rollout_df = pd.read_csv(ROLLOUT_PATH)
base_df = pd.read_csv(BASE_MERGED_PATH)

print(f"[INFO] rollout rows: {len(rollout_df)}")
print(f"[INFO] base rows   : {len(base_df)}")

# =========================
# phase 4~8 필터
# =========================
dagger_df = rollout_df[rollout_df["phase"].isin(TARGET_PHASES)].copy()

print(f"[INFO] dagger rows (phase 4~8): {len(dagger_df)}")

# =========================
# relabel (pos만)
# =========================
def relabel_pos(row):
    phase = int(row["phase"])

    if phase == 4:
        # lift
        return pd.Series([
            row["obj_pos_x"],
            row["obj_pos_y"],
            LEFT_LIFT_Z
        ])

    else:
        # transfer / handover
        return pd.Series([
            HANDOVER_X,
            HANDOVER_Y,
            HANDOVER_Z
        ])

dagger_df[[
    "action_left_pos_x",
    "action_left_pos_y",
    "action_left_pos_z"
]] = dagger_df.apply(relabel_pos, axis=1)

# =========================
# episode_id 추가 (없으면)
# =========================
dagger_df["episode_id"] = "dagger_round1"

# =========================
# 저장
# =========================
relabel_path = SAVE_DIR / "dagger_round1_relabel.csv"
merged_path = SAVE_DIR / "handover_merged_round1.csv"

dagger_df.to_csv(relabel_path, index=False)

merged_df = pd.concat([base_df, dagger_df], ignore_index=True)
merged_df.to_csv(merged_path, index=False)

print("\n[SAVE]")
print(f"relabel: {relabel_path}")
print(f"merged : {merged_path}")
print(f"merged total rows: {len(merged_df)}")