import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/home/june/bimanual_handover_IL/data/merged/handover_merged.csv"

df = pd.read_csv(CSV_PATH)

# ===== 핵심 값 =====
df["left_z_rel"] = df["action_left_pos_z"] - df["obj_pos_z"]
df["right_z_rel"] = df["action_right_pos_z"] - df["obj_pos_z"]

# phase가 있다고 가정 (없으면 이거 먼저 확인해야 함)
assert "phase" in df.columns, "phase column 없음"

# ===== phase별 데이터 묶기 =====
phase_list = sorted(df["phase"].unique())

left_data = []
right_data = []

for p in phase_list:
    sub = df[df["phase"] == p]

    left_data.append(sub["left_z_rel"].values)
    right_data.append(sub["right_z_rel"].values)

# ===== plot =====

plt.figure(figsize=(12, 5))

# 왼팔
plt.subplot(1, 2, 1)
plt.boxplot(left_data, labels=phase_list, showfliers=False)
plt.title("LEFT: action_z - obj_z by phase")
plt.xlabel("phase")
plt.ylabel("z_rel (m)")

# 오른팔
plt.subplot(1, 2, 2)
plt.boxplot(right_data, labels=phase_list, showfliers=False)
plt.title("RIGHT: action_z - obj_z by phase")
plt.xlabel("phase")
plt.ylabel("z_rel (m)")

plt.tight_layout()
plt.show()