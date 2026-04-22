import os
import glob
import pandas as pd

# ===== 경로 설정 =====
INPUT_DIR = "/home/june/bimanual_handover_IL/data/handover_logging_3"
OUTPUT_PATH = "/home/june/bimanual_handover_IL/data/merged/handover_merged.csv"

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    assert len(csv_files) > 0, "csv 파일 없음"

    print(f"[INFO] found {len(csv_files)} files")

    dfs = []

    for i, path in enumerate(csv_files):
        try:
            df = pd.read_csv(path)

            # episode_id 추가 (파일 단위)
            df["episode_id"] = i

            dfs.append(df)

        except Exception as e:
            print(f"[WARN] skip file: {path}, error: {e}")

    merged = pd.concat(dfs, ignore_index=True)

    print(f"[INFO] merged shape: {merged.shape}")

    # 저장
    merged.to_csv(OUTPUT_PATH, index=False)

    df = pd.read_csv(OUTPUT_PATH)

    print(df.shape)
    print(df.columns.tolist())
    print(df["episode_id"].nunique())
    print(f"[DONE] saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()