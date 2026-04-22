import pandas as pd
from pathlib import Path

# =========================
# 경로 (고정)
# =========================
ROLLOUT_DIR = Path(
    "/home/june/bimanual_handover_IL/data/rollout_bc_rot6d_round1_eval"
)

BASE_MERGED_PATH = Path(
    "/home/june/bimanual_handover_IL/data/merged/handover_merged_round1.csv"
)

SAVE_DIR = Path("/home/june/bimanual_handover_IL/data/merged")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 설정
# =========================
TARGET_PHASES = [9, 10, 11]

# =========================
# load
# =========================
base_df = pd.read_csv(BASE_MERGED_PATH, low_memory=False)
rollout_paths = sorted(ROLLOUT_DIR.glob("*.csv"))

if len(rollout_paths) == 0:
    raise RuntimeError(f"[ERROR] no rollout csv found in {ROLLOUT_DIR}")

print(f"[INFO] base rows   : {len(base_df)}")
print(f"[INFO] rollout csv count: {len(rollout_paths)}")

# =========================
# round1 merged에서 phase별 target 계산
# action_left_pos median 사용
# grip은 binary 유지
# =========================
phase_target_map = {}

for phase in TARGET_PHASES:
    sub = base_df[base_df["phase"] == phase].copy()
    if len(sub) == 0:
        raise RuntimeError(f"[ERROR] base merged has no rows for phase {phase}")

    tx = float(sub["action_left_pos_x"].median())
    ty = float(sub["action_left_pos_y"].median())
    tz = float(sub["action_left_pos_z"].median())

    left_grip_bin = int(round(float(sub["action_left_grip_cmd"].median())))
    right_grip_bin = int(round(float(sub["action_right_grip_cmd"].median())))

    phase_target_map[phase] = {
        "left_pos": (tx, ty, tz),
        "left_grip": left_grip_bin,
        "right_grip": right_grip_bin,
    }

print("\n[INFO] phase target map")
for phase in TARGET_PHASES:
    cfg = phase_target_map[phase]
    print(
        f"phase {phase}: "
        f"left_pos={cfg['left_pos']}, "
        f"left_grip={cfg['left_grip']}, "
        f"right_grip={cfg['right_grip']}"
    )

# =========================
# rollout 10개 전부 모아서 phase 9~11만 추출
# =========================
dagger_parts = []

for path in rollout_paths:
    df = pd.read_csv(path, low_memory=False)
    sub = df[df["phase"].isin(TARGET_PHASES)].copy()

    if len(sub) == 0:
        print(f"[WARN] no phase 9/10/11 rows in {path.name}")
        continue

    # 파일별로 episode_id 분리
    sub["episode_id"] = f"dagger_round2_{path.stem}"

    dagger_parts.append(sub)
    print(f"[INFO] loaded {path.name}: {len(sub)} rows")

if len(dagger_parts) == 0:
    raise RuntimeError("[ERROR] no dagger rows collected from rollout csv files")

dagger_df = pd.concat(dagger_parts, ignore_index=True)
print(f"\n[INFO] total dagger rows (phase 9~11): {len(dagger_df)}")

# =========================
# relabel
# left pos만 교정
# grip은 binary 유지
# =========================
def relabel_row(row):
    phase = int(row["phase"])
    cfg = phase_target_map[phase]

    row["action_left_pos_x"] = cfg["left_pos"][0]
    row["action_left_pos_y"] = cfg["left_pos"][1]
    row["action_left_pos_z"] = cfg["left_pos"][2]

    row["action_left_grip_cmd"] = cfg["left_grip"]
    row["action_right_grip_cmd"] = cfg["right_grip"]

    return row

dagger_df = dagger_df.apply(relabel_row, axis=1)

# =========================
# base merged와 동일 컬럼만 유지
# =========================
base_cols = list(base_df.columns)

missing_cols = [c for c in base_cols if c not in dagger_df.columns]
if missing_cols:
    raise RuntimeError(f"[ERROR] rollout csv is missing required columns: {missing_cols}")

dagger_df = dagger_df[base_cols].copy()

# dtype 정리
dagger_df["action_left_grip_cmd"] = dagger_df["action_left_grip_cmd"].astype(int)
dagger_df["action_right_grip_cmd"] = dagger_df["action_right_grip_cmd"].astype(int)
dagger_df["episode_id"] = dagger_df["episode_id"].astype(str)

# =========================
# 저장
# =========================
relabel_path = SAVE_DIR / "dagger_round2_relabel.csv"
merged_path = SAVE_DIR / "handover_merged_round2.csv"

dagger_df.to_csv(relabel_path, index=False)

merged_df = pd.concat([base_df, dagger_df], ignore_index=True)
merged_df.to_csv(merged_path, index=False)

print("\n[SAVE]")
print(f"relabel: {relabel_path}")
print(f"merged : {merged_path}")
print(f"relabel rows: {len(dagger_df)}")
print(f"merged total rows: {len(merged_df)}")

print("\n[INFO] dagger episode_ids:")
print(dagger_df["episode_id"].value_counts())