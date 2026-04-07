import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATH
# =========================
CSV_PATH = "/home/june/bimanual_handover_IL/data/merged_raw_success.csv"
SAVE_DIR = "/home/june/bimanual_handover_IL/data/data_analysis/"


# =========================
# BASIC
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    numeric_cols = [
        "step",
        "phase",
        "obj_x", "obj_y", "obj_z",
        "left_tcp_x", "left_tcp_y", "left_tcp_z",
        "right_tcp_x", "right_tcp_y", "right_tcp_z",
        "left_grasped", "right_grasped",
        "action_left_dx", "action_left_dy", "action_left_dz",
        "action_right_dx", "action_right_dy", "action_right_dz",
        "action_left_gripper", "action_right_gripper",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # episode_id는 문자열(ep_000) 유지
    if "episode_id" in df.columns:
        df["episode_id"] = df["episode_id"].astype(str)

    return df


def save_fig(save_path: str):
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# OBJ PLOTS
# =========================
def plot_obj_histograms(df: pd.DataFrame, save_dir: str):
    obj_cols = ["obj_x", "obj_y", "obj_z"]

    for col in obj_cols:
        plt.figure(figsize=(7, 4.5))
        plt.hist(df[col].dropna(), bins=40)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"{col} histogram")
        save_fig(os.path.join(save_dir, f"{col}_histogram.png"))


def plot_obj_xy_scatter(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(df["obj_x"], df["obj_y"], s=8, alpha=0.4)
    plt.xlabel("obj_x")
    plt.ylabel("obj_y")
    plt.title("obj_x vs obj_y")
    save_fig(os.path.join(save_dir, "obj_xy_scatter.png"))


def plot_obj_z_timeseries_sample_episodes(df: pd.DataFrame, save_dir: str, num_episodes: int = 3):
    episode_ids = sorted(df["episode_id"].dropna().unique())[:num_episodes]

    for ep in episode_ids:
        ep_df = df[df["episode_id"] == ep].sort_values("step")

        if ep_df.empty:
            continue

        plt.figure(figsize=(10, 4.5))
        plt.plot(ep_df["step"], ep_df["obj_z"])
        plt.xlabel("step")
        plt.ylabel("obj_z")
        plt.title(f"obj_z over step - {ep}")
        save_fig(os.path.join(save_dir, f"obj_z_timeseries_{ep}.png"))


def plot_obj_z_by_grasp_state(df: pd.DataFrame, save_dir: str):
    left_data = [
        df.loc[df["left_grasped"] == 0, "obj_z"].dropna(),
        df.loc[df["left_grasped"] == 1, "obj_z"].dropna(),
    ]

    plt.figure(figsize=(6, 4.5))
    plt.boxplot(left_data, tick_labels=["left_grasped=0", "left_grasped=1"])
    plt.ylabel("obj_z")
    plt.title("obj_z by left_grasped")
    save_fig(os.path.join(save_dir, "obj_z_by_left_grasped_boxplot.png"))

    right_data = [
        df.loc[df["right_grasped"] == 0, "obj_z"].dropna(),
        df.loc[df["right_grasped"] == 1, "obj_z"].dropna(),
    ]

    plt.figure(figsize=(6, 4.5))
    plt.boxplot(right_data, tick_labels=["right_grasped=0", "right_grasped=1"])
    plt.ylabel("obj_z")
    plt.title("obj_z by right_grasped")
    save_fig(os.path.join(save_dir, "obj_z_by_right_grasped_boxplot.png"))


def plot_final_obj_z_per_episode(df: pd.DataFrame, save_dir: str):
    final_df = (
        df.sort_values(["episode_id", "step"])
        .groupby("episode_id", as_index=False)
        .tail(1)
        .sort_values("episode_id")
    )

    plt.figure(figsize=(12, 4.5))
    plt.bar(final_df["episode_id"], final_df["obj_z"])
    plt.xticks(rotation=90)
    plt.xlabel("episode_id")
    plt.ylabel("final_obj_z")
    plt.title("Final obj_z per episode")
    save_fig(os.path.join(save_dir, "final_obj_z_per_episode.png"))


# =========================
# LEFT TCP PLOTS
# =========================
def plot_left_tcp_histograms(df: pd.DataFrame, save_dir: str):
    cols = ["left_tcp_x", "left_tcp_y", "left_tcp_z"]

    for col in cols:
        plt.figure(figsize=(7, 4.5))
        plt.hist(df[col].dropna(), bins=40)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"{col} histogram")
        save_fig(os.path.join(save_dir, f"{col}_hist.png"))


def plot_left_tcp_timeseries(df: pd.DataFrame, save_dir: str, num_episodes: int = 3):
    episode_ids = sorted(df["episode_id"].dropna().unique())[:num_episodes]

    for ep in episode_ids:
        ep_df = df[df["episode_id"] == ep].sort_values("step")

        if ep_df.empty:
            continue

        plt.figure(figsize=(10, 4.5))
        plt.plot(ep_df["step"], ep_df["left_tcp_x"], label="x")
        plt.plot(ep_df["step"], ep_df["left_tcp_y"], label="y")
        plt.plot(ep_df["step"], ep_df["left_tcp_z"], label="z")
        plt.xlabel("step")
        plt.ylabel("left_tcp")
        plt.title(f"left_tcp trajectory - {ep}")
        plt.legend()
        save_fig(os.path.join(save_dir, f"left_tcp_timeseries_{ep}.png"))


def plot_left_tcp_phase_box(df: pd.DataFrame, save_dir: str):
    cols = ["left_tcp_x", "left_tcp_y", "left_tcp_z"]

    phase_order = (
        df[["phase", "phase_name"]]
        .drop_duplicates()
        .sort_values("phase")
    )

    for col in cols:
        data = [
            df.loc[df["phase_name"] == pname, col].dropna()
            for pname in phase_order["phase_name"]
        ]

        plt.figure(figsize=(12, 5))
        plt.boxplot(data, tick_labels=phase_order["phase_name"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(col)
        plt.title(f"{col} by phase")
        save_fig(os.path.join(save_dir, f"{col}_phase_box.png"))


def plot_left_tcp_relative(df: pd.DataFrame, save_dir: str):
    df["rel_left_x"] = df["obj_x"] - df["left_tcp_x"]
    df["rel_left_y"] = df["obj_y"] - df["left_tcp_y"]
    df["rel_left_z"] = df["obj_z"] - df["left_tcp_z"]

    cols = ["rel_left_x", "rel_left_y", "rel_left_z"]

    for col in cols:
        plt.figure(figsize=(7, 4.5))
        plt.hist(df[col].dropna(), bins=40)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"{col} histogram")
        save_fig(os.path.join(save_dir, f"{col}_hist.png"))


# =========================
# RIGHT TCP PLOTS
# =========================
def plot_right_tcp_histograms(df: pd.DataFrame, save_dir: str):
    cols = ["right_tcp_x", "right_tcp_y", "right_tcp_z"]

    for col in cols:
        plt.figure(figsize=(7, 4.5))
        plt.hist(df[col].dropna(), bins=40)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"{col} histogram")
        save_fig(os.path.join(save_dir, f"{col}_hist.png"))


def plot_right_tcp_timeseries(df: pd.DataFrame, save_dir: str, num_episodes: int = 3):
    episode_ids = sorted(df["episode_id"].dropna().unique())[:num_episodes]

    for ep in episode_ids:
        ep_df = df[df["episode_id"] == ep].sort_values("step")

        if ep_df.empty:
            continue

        plt.figure(figsize=(10, 4.5))
        plt.plot(ep_df["step"], ep_df["right_tcp_x"], label="x")
        plt.plot(ep_df["step"], ep_df["right_tcp_y"], label="y")
        plt.plot(ep_df["step"], ep_df["right_tcp_z"], label="z")
        plt.xlabel("step")
        plt.ylabel("right_tcp")
        plt.title(f"right_tcp trajectory - {ep}")
        plt.legend()
        save_fig(os.path.join(save_dir, f"right_tcp_timeseries_{ep}.png"))


def plot_right_tcp_phase_box(df: pd.DataFrame, save_dir: str):
    cols = ["right_tcp_x", "right_tcp_y", "right_tcp_z"]

    phase_order = (
        df[["phase", "phase_name"]]
        .drop_duplicates()
        .sort_values("phase")
    )

    for col in cols:
        data = [
            df.loc[df["phase_name"] == pname, col].dropna()
            for pname in phase_order["phase_name"]
        ]

        plt.figure(figsize=(12, 5))
        plt.boxplot(data, tick_labels=phase_order["phase_name"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(col)
        plt.title(f"{col} by phase")
        save_fig(os.path.join(save_dir, f"{col}_phase_box.png"))


def plot_right_tcp_relative(df: pd.DataFrame, save_dir: str):
    df["rel_right_x"] = df["obj_x"] - df["right_tcp_x"]
    df["rel_right_y"] = df["obj_y"] - df["right_tcp_y"]
    df["rel_right_z"] = df["obj_z"] - df["right_tcp_z"]

    cols = ["rel_right_x", "rel_right_y", "rel_right_z"]

    for col in cols:
        plt.figure(figsize=(7, 4.5))
        plt.hist(df[col].dropna(), bins=40)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"{col} histogram")
        save_fig(os.path.join(save_dir, f"{col}_hist.png"))


# =========================
# ACTION LEFT PLOTS
# =========================
def add_action_left_norm(df: pd.DataFrame):
    df["action_left_norm"] = (
        df["action_left_dx"] ** 2 +
        df["action_left_dy"] ** 2 +
        df["action_left_dz"] ** 2
    ) ** 0.5


def plot_action_left_histograms(df: pd.DataFrame, save_dir: str):
    cols = ["action_left_dx", "action_left_dy", "action_left_dz"]

    for col in cols:
        plt.figure(figsize=(7, 4.5))
        plt.hist(df[col].dropna(), bins=40)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"{col} histogram")
        save_fig(os.path.join(save_dir, f"{col}_hist.png"))


def plot_action_left_norm_histogram(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(7, 4.5))
    plt.hist(df["action_left_norm"].dropna(), bins=40)
    plt.xlabel("action_left_norm")
    plt.ylabel("count")
    plt.title("action_left_norm histogram")
    save_fig(os.path.join(save_dir, "action_left_norm_hist.png"))


def plot_action_left_phase_box(df: pd.DataFrame, save_dir: str):
    cols = ["action_left_dx", "action_left_dy", "action_left_dz"]

    phase_order = (
        df[["phase", "phase_name"]]
        .drop_duplicates()
        .sort_values("phase")
    )

    for col in cols:
        data = [
            df.loc[df["phase_name"] == pname, col].dropna()
            for pname in phase_order["phase_name"]
        ]

        plt.figure(figsize=(12, 5))
        plt.boxplot(data, tick_labels=phase_order["phase_name"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(col)
        plt.title(f"{col} by phase")
        save_fig(os.path.join(save_dir, f"{col}_phase_box.png"))


def plot_action_left_norm_phase_box(df: pd.DataFrame, save_dir: str):
    phase_order = (
        df[["phase", "phase_name"]]
        .drop_duplicates()
        .sort_values("phase")
    )

    data = [
        df.loc[df["phase_name"] == pname, "action_left_norm"].dropna()
        for pname in phase_order["phase_name"]
    ]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, tick_labels=phase_order["phase_name"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("action_left_norm")
    plt.title("action_left_norm by phase")
    save_fig(os.path.join(save_dir, "action_left_norm_phase_box.png"))


def plot_action_left_timeseries(df: pd.DataFrame, save_dir: str, num_episodes: int = 3):
    episode_ids = sorted(df["episode_id"].dropna().unique())[:num_episodes]

    for ep in episode_ids:
        ep_df = df[df["episode_id"] == ep].sort_values("step")

        if ep_df.empty:
            continue

        plt.figure(figsize=(10, 4.5))
        plt.plot(ep_df["step"], ep_df["action_left_dx"], label="dx")
        plt.plot(ep_df["step"], ep_df["action_left_dy"], label="dy")
        plt.plot(ep_df["step"], ep_df["action_left_dz"], label="dz")
        plt.xlabel("step")
        plt.ylabel("action_left")
        plt.title(f"action_left trajectory - {ep}")
        plt.legend()
        save_fig(os.path.join(save_dir, f"action_left_timeseries_{ep}.png"))


# =========================
# ACTION RIGHT PLOTS
# =========================
def add_action_right_norm(df: pd.DataFrame):
    df["action_right_norm"] = (
        df["action_right_dx"] ** 2 +
        df["action_right_dy"] ** 2 +
        df["action_right_dz"] ** 2
    ) ** 0.5


def plot_action_right_histograms(df: pd.DataFrame, save_dir: str):
    cols = ["action_right_dx", "action_right_dy", "action_right_dz"]

    for col in cols:
        plt.figure(figsize=(7, 4.5))
        plt.hist(df[col].dropna(), bins=40)
        plt.xlabel(col)
        plt.ylabel("count")
        plt.title(f"{col} histogram")
        save_fig(os.path.join(save_dir, f"{col}_hist.png"))


def plot_action_right_norm_histogram(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(7, 4.5))
    plt.hist(df["action_right_norm"].dropna(), bins=40)
    plt.xlabel("action_right_norm")
    plt.ylabel("count")
    plt.title("action_right_norm histogram")
    save_fig(os.path.join(save_dir, "action_right_norm_hist.png"))


def plot_action_right_phase_box(df: pd.DataFrame, save_dir: str):
    cols = ["action_right_dx", "action_right_dy", "action_right_dz"]

    phase_order = (
        df[["phase", "phase_name"]]
        .drop_duplicates()
        .sort_values("phase")
    )

    for col in cols:
        data = [
            df.loc[df["phase_name"] == pname, col].dropna()
            for pname in phase_order["phase_name"]
        ]

        plt.figure(figsize=(12, 5))
        plt.boxplot(data, tick_labels=phase_order["phase_name"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(col)
        plt.title(f"{col} by phase")
        save_fig(os.path.join(save_dir, f"{col}_phase_box.png"))


def plot_action_right_norm_phase_box(df: pd.DataFrame, save_dir: str):
    phase_order = (
        df[["phase", "phase_name"]]
        .drop_duplicates()
        .sort_values("phase")
    )

    data = [
        df.loc[df["phase_name"] == pname, "action_right_norm"].dropna()
        for pname in phase_order["phase_name"]
    ]

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, tick_labels=phase_order["phase_name"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("action_right_norm")
    plt.title("action_right_norm by phase")
    save_fig(os.path.join(save_dir, "action_right_norm_phase_box.png"))


def plot_action_right_timeseries(df: pd.DataFrame, save_dir: str, num_episodes: int = 3):
    episode_ids = sorted(df["episode_id"].dropna().unique())[:num_episodes]

    for ep in episode_ids:
        ep_df = df[df["episode_id"] == ep].sort_values("step")

        if ep_df.empty:
            continue

        plt.figure(figsize=(10, 4.5))
        plt.plot(ep_df["step"], ep_df["action_right_dx"], label="dx")
        plt.plot(ep_df["step"], ep_df["action_right_dy"], label="dy")
        plt.plot(ep_df["step"], ep_df["action_right_dz"], label="dz")
        plt.xlabel("step")
        plt.ylabel("action_right")
        plt.title(f"action_right trajectory - {ep}")
        plt.legend()
        save_fig(os.path.join(save_dir, f"action_right_timeseries_{ep}.png"))


def plot_gripper_actions_with_phase(df: pd.DataFrame, save_dir: str, num_episodes: int = 3):
    episode_ids = sorted(df["episode_id"].dropna().unique())[:num_episodes]

    for ep in episode_ids:
        ep_df = df[df["episode_id"] == ep].sort_values("step").copy()

        if ep_df.empty:
            continue

        plt.figure(figsize=(12, 4.5))
        plt.plot(ep_df["step"], ep_df["action_left_gripper"], label="left_gripper")
        plt.plot(ep_df["step"], ep_df["action_right_gripper"], label="right_gripper")

        phase_change_idx = ep_df["phase"].ne(ep_df["phase"].shift()).fillna(True)
        phase_rows = ep_df.loc[phase_change_idx, ["step", "phase_name"]]

        y_top = max(
            ep_df["action_left_gripper"].max(),
            ep_df["action_right_gripper"].max()
        )

        for _, row in phase_rows.iterrows():
            x = row["step"]
            plt.axvline(x=x, linestyle="--", alpha=0.5)
            plt.text(
                x,
                y_top,
                str(row["phase_name"]),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8
            )

        plt.xlabel("step")
        plt.ylabel("gripper action")
        plt.title(f"gripper actions with phase - episode {ep}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"gripper_actions_with_phase_{ep}.png"),
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()


# =========================
# MAIN
# =========================
def main():
    ensure_dir(SAVE_DIR)
    df = load_data(CSV_PATH)

    # obj
    plot_obj_histograms(df, SAVE_DIR)
    plot_obj_xy_scatter(df, SAVE_DIR)
    plot_obj_z_timeseries_sample_episodes(df, SAVE_DIR, num_episodes=3)
    plot_obj_z_by_grasp_state(df, SAVE_DIR)
    plot_final_obj_z_per_episode(df, SAVE_DIR)

    # left tcp
    plot_left_tcp_histograms(df, SAVE_DIR)
    plot_left_tcp_timeseries(df, SAVE_DIR, num_episodes=3)
    plot_left_tcp_phase_box(df, SAVE_DIR)
    plot_left_tcp_relative(df, SAVE_DIR)

    # right tcp
    plot_right_tcp_histograms(df, SAVE_DIR)
    plot_right_tcp_timeseries(df, SAVE_DIR, num_episodes=3)
    plot_right_tcp_phase_box(df, SAVE_DIR)
    plot_right_tcp_relative(df, SAVE_DIR)

    # action left delta
    add_action_left_norm(df)
    plot_action_left_histograms(df, SAVE_DIR)
    plot_action_left_norm_histogram(df, SAVE_DIR)
    plot_action_left_phase_box(df, SAVE_DIR)
    plot_action_left_norm_phase_box(df, SAVE_DIR)
    plot_action_left_timeseries(df, SAVE_DIR, num_episodes=3)

    # action right delta
    add_action_right_norm(df)
    plot_action_right_histograms(df, SAVE_DIR)
    plot_action_right_norm_histogram(df, SAVE_DIR)
    plot_action_right_phase_box(df, SAVE_DIR)
    plot_action_right_norm_phase_box(df, SAVE_DIR)
    plot_action_right_timeseries(df, SAVE_DIR, num_episodes=3)

    # gripper action
    plot_gripper_actions_with_phase(df, SAVE_DIR, num_episodes=3)

    print("Saved plots to:", SAVE_DIR)


if __name__ == "__main__":
    main()