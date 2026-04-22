import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/home/june/bimanual_handover_IL/data/merged/handover_merged.csv"

df = pd.read_csv(CSV_PATH)

print("===== BASIC INFO =====")
print("rows:", len(df))
print("episodes:", df["episode_id"].nunique())
print()

# =========================
# 1️⃣ Action boxplot
# =========================

action_cols = [c for c in df.columns if c.startswith("action")]

plt.figure(figsize=(12,6))
plt.boxplot([df[c].values for c in action_cols])
plt.xticks(range(1, len(action_cols)+1), action_cols, rotation=90)
plt.title("Action distribution (boxplot)")
plt.tight_layout()
plt.show()


# =========================
# 2️⃣ Action magnitude
# =========================

left_cols = [c for c in df.columns if "action_left_pos" in c]
right_cols = [c for c in df.columns if "action_right_pos" in c]

left_mag = np.linalg.norm(df[left_cols].values, axis=1)
right_mag = np.linalg.norm(df[right_cols].values, axis=1)

plt.figure()
plt.hist(left_mag, bins=50, alpha=0.6, label="left")
plt.hist(right_mag, bins=50, alpha=0.6, label="right")
plt.legend()
plt.title("Action magnitude")
plt.show()


# =========================
# 3️⃣ rot6d check
# =========================

def check_rot6d(prefix):
    v1 = df[[f"{prefix}_rot6d_{i}" for i in range(3)]].values
    v2 = df[[f"{prefix}_rot6d_{i}" for i in range(3,6)]].values

    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    dot = np.sum(v1 * v2, axis=1)

    return norm1, norm2, dot

n1, n2, d1 = check_rot6d("left")
n3, n4, d2 = check_rot6d("right")

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.hist(n1, bins=50, alpha=0.5, label="left")
plt.hist(n3, bins=50, alpha=0.5, label="right")
plt.title("norm(v1)")

plt.subplot(1,3,2)
plt.hist(n2, bins=50, alpha=0.5, label="left")
plt.hist(n4, bins=50, alpha=0.5, label="right")
plt.title("norm(v2)")

plt.subplot(1,3,3)
plt.hist(d1, bins=50, alpha=0.5, label="left")
plt.hist(d2, bins=50, alpha=0.5, label="right")
plt.title("dot(v1,v2)")

plt.legend()
plt.tight_layout()
plt.show()