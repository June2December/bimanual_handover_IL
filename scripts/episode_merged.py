import os
import glob
import pandas as pd

# =========================
# 1. 경로 설정
# =========================
dir_tuned = "/home/june/bimanual_handover_IL/data/left_RMPFlow_right_joint_logged"
dir_untuned = "/home/june/bimanual_handover_IL/data/left_RMPFlow_right_joint_logged_1"
save_dir = "/home/june/bimanual_handover_IL/data"

merged_csv_path = os.path.join(save_dir, "merged_raw_success.csv")
summary_csv_path = os.path.join(save_dir, "merged_raw_summary.csv")

# =========================
# 2. CSV 파일 수집
# =========================
csv_files_tuned = sorted(glob.glob(os.path.join(dir_tuned, "*.csv")))
csv_files_untuned = sorted(glob.glob(os.path.join(dir_untuned, "*.csv")))

all_csv_files = csv_files_tuned + csv_files_untuned

print(f"[INFO] tuned csv count   : {len(csv_files_tuned)}")
print(f"[INFO] untuned csv count : {len(csv_files_untuned)}")
print(f"[INFO] total csv count   : {len(all_csv_files)}")

if len(all_csv_files) == 0:
    raise FileNotFoundError("병합할 csv 파일을 찾지 못했습니다.")

# =========================
# 3. 병합
# =========================
merged_list = []
episode_counter = 0

for file_path in all_csv_files:
    df = pd.read_csv(file_path)

    # 빈 파일 방지
    if df.empty:
        print(f"[WARN] empty file skipped: {file_path}")
        continue

    # episode_id / original_file 추가
    df["episode_id"] = f"ep_{episode_counter:03d}"
    df["original_file"] = file_path

    merged_list.append(df)
    episode_counter += 1

if len(merged_list) == 0:
    raise ValueError("읽을 수 있는 유효한 csv가 없습니다.")

merged_df = pd.concat(merged_list, ignore_index=True)

# =========================
# 4. 컬럼 순서 정리
# =========================
front_cols = ["episode_id", "original_file"]
other_cols = [c for c in merged_df.columns if c not in front_cols]
merged_df = merged_df[front_cols + other_cols]

# =========================
# 5. merged 저장
# =========================
merged_df.to_csv(merged_csv_path, index=False)

print(f"[INFO] merged saved to: {merged_csv_path}")
print(f"[INFO] total episodes : {merged_df['episode_id'].nunique()}")
print(f"[INFO] total rows     : {len(merged_df)}")

# =========================
# 6. summary 생성
# =========================
# step 컬럼이 있다고 가정
sort_cols = ["episode_id"]
if "step" in merged_df.columns:
    sort_cols.append("step")

merged_sorted = merged_df.sort_values(sort_cols).copy()

agg_dict = {
    "original_file": "first",
}

if "step" in merged_sorted.columns:
    agg_dict["step"] = "max"
if "phase" in merged_sorted.columns:
    agg_dict["phase"] = "last"
if "phase_name" in merged_sorted.columns:
    agg_dict["phase_name"] = "last"
if "obj_z" in merged_sorted.columns:
    agg_dict["obj_z"] = "last"
if "left_grasped" in merged_sorted.columns:
    agg_dict["left_grasped"] = "last"
if "right_grasped" in merged_sorted.columns:
    agg_dict["right_grasped"] = "last"

summary_df = merged_sorted.groupby("episode_id", as_index=False).agg(agg_dict)

# 컬럼명 조금 보기 좋게 변경
rename_map = {}
if "step" in summary_df.columns:
    rename_map["step"] = "final_step"
if "phase" in summary_df.columns:
    rename_map["phase"] = "final_phase"
if "phase_name" in summary_df.columns:
    rename_map["phase_name"] = "final_phase_name"
if "obj_z" in summary_df.columns:
    rename_map["obj_z"] = "final_obj_z"
if "left_grasped" in summary_df.columns:
    rename_map["left_grasped"] = "final_left_grasped"
if "right_grasped" in summary_df.columns:
    rename_map["right_grasped"] = "final_right_grasped"

summary_df = summary_df.rename(columns=rename_map)

summary_df.to_csv(summary_csv_path, index=False)

print(f"[INFO] summary saved to: {summary_csv_path}")
print("[INFO] done.")