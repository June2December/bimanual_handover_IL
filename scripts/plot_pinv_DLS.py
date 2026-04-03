import os
import pandas as pd
import matplotlib.pyplot as plt

SAVE_DIR = "/home/june/bimanual_handover_IL/data/"
PINV_CSV = os.path.join(SAVE_DIR, "pinv_log.csv")
DLS_CSV = os.path.join(SAVE_DIR, "dls_log.csv")

if not os.path.exists(PINV_CSV):
    raise FileNotFoundError(f"not found: {PINV_CSV}")

if not os.path.exists(DLS_CSV):
    raise FileNotFoundError(f"not found: {DLS_CSV}")

pinv = pd.read_csv(PINV_CSV)
dls = pd.read_csv(DLS_CSV)

# ---------------------------------------------------------
# 1) qdot norm
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(pinv["step"], pinv["qdot_norm"], label="pinv")
plt.plot(dls["step"], dls["qdot_norm"], label="dls")
plt.xlabel("step")
plt.ylabel("qdot_norm")
plt.title("qdot norm: pinv vs dls")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_qdot_norm.png"), dpi=150)

# ---------------------------------------------------------
# 2) tracking error
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(pinv["step"], pinv["track_err"], label="pinv")
plt.plot(dls["step"], dls["track_err"], label="dls")
plt.xlabel("step")
plt.ylabel("tracking_error")
plt.title("tracking error: pinv vs dls")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_tracking_error.png"), dpi=150)

# ---------------------------------------------------------
# 3) q5 변화
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(pinv["step"], pinv["q5_deg"], label="pinv")
plt.plot(dls["step"], dls["q5_deg"], label="dls")
plt.xlabel("step")
plt.ylabel("q5_deg")
plt.title("q5 angle change")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "plot_q5_deg.png"), dpi=150)

# ---------------------------------------------------------
# 4) qdot 각 성분 중 2,3,4번 비교
# ---------------------------------------------------------
for idx in [2, 3, 4]:
    col = f"qdot_{idx}"
    plt.figure(figsize=(8, 5))
    plt.plot(pinv["step"], pinv[col], label=f"pinv {col}")
    plt.plot(dls["step"], dls[col], label=f"dls {col}")
    plt.xlabel("step")
    plt.ylabel(col)
    plt.title(f"{col}: pinv vs dls")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"plot_{col}.png"), dpi=150)

plt.show()