import json
import numpy as np


def compute_mean_std(df, cols):
    arr = df[cols].values.astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def save_norm_stats(path, cols, mean, std):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cols": cols,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            f,
            indent=2,
        )


def load_norm_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    cols = stats["cols"]
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    return cols, mean, std