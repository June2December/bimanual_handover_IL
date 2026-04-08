import pandas as pd

# =========================
# 설정
# =========================
CSV_PATH = "/home/june/bimanual_handover_IL/checkpoints/bc_single_head/eval_samples/test_predictions_compare.csv"

PHASE_NAME = "LEFT_POST_RELEASE_UP"   # phase 이름 (본인 데이터에 맞게 수정 가능)
TOP_K = 20                           # 상위 error 몇 개 볼지

# =========================
# 로드
# =========================
df = pd.read_csv(CSV_PATH)

# =========================
# phase 필터링
# =========================
df_phase = df[df["phase_name"] == PHASE_NAME].copy()

# =========================
# error 기준 정렬
# =========================
df_phase = df_phase.sort_values(by="mse_all", ascending=False)

# 상위 K개
df_top = df_phase.head(TOP_K)

print(f"\n[INFO] {PHASE_NAME} total samples: {len(df_phase)}")
print(f"[INFO] showing top {TOP_K} high-error samples\n")

# =========================
# 필요한 컬럼
# =========================
cols = [
    "sample_idx",
    "mse_all",

    # GT (정답)
    "gt__action_left_dx",
    "gt__action_left_dy",
    "gt__action_left_dz",

    # Pred (모델)
    "pred__action_left_dx",
    "pred__action_left_dy",
    "pred__action_left_dz",
]

# 존재하는 컬럼만 사용
cols = [c for c in cols if c in df_top.columns]

# =========================
# 출력
# =========================
print(df_top[cols].to_string(index=False))