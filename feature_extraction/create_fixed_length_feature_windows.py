import pandas as pd
import numpy as np

def build_containers(
    df: pd.DataFrame,
    use_rep_id: bool = True,
) -> pd.DataFrame:
    """
    Builds containers based on rep_id if available (lab data) and label if not (real-world data).
    Similar to build boundaries from previously!
    """

    df = df.reset_index(drop=True).copy()

    if use_rep_id and "rep_id" in df.columns and df["rep_id"].notna().any():
        key = df["rep_id"].fillna("none").astype(str).to_numpy()
        container_prefix = "rep"
    else:
        if "label" not in df.columns:
            raise ValueError("No label column available - cannot build container for segmentation.")
        key = df["label"].astype(str).to_numpy()
        container_prefix = "labelrun"

    change_points = np.where(key[:-1] != key[1:])[0] + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(df)]))

    rows = []
    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        rows.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),  # exclusive
                "label": df.iloc[start_idx]["label"] if "label" in df.columns else None,
                "rep_id": df.iloc[start_idx]["rep_id"] if "rep_id" in df.columns else None,
                "container_id": f"{container_prefix}_{i}",
            }
        )

    return pd.DataFrame(rows)

def generate_fixed_length_windows_centered(
    containers: pd.DataFrame,
    fs: int,
    window_sec: float,
) -> pd.DataFrame:
    """
    Split each container into fixed-length windows (no overlap).
    If container length is not a perfect multiple, crop symmetrically
    at start and end so windows are centered.
    """

    window_size = int(round(window_sec * fs))
    rows = []

    for _, row in containers.iterrows():

        c_start = int(row["start_idx"])
        c_end = int(row["end_idx"])
        container_len = c_end - c_start

        n_windows = container_len // window_size

        if n_windows == 0:
            continue

        usable_len = n_windows * window_size
        leftover = container_len - usable_len

        crop_start = leftover // 2
        new_start = c_start + crop_start

        for w in range(n_windows):

            start_idx = new_start + w * window_size
            end_idx = start_idx + window_size

            rows.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "label": row["label"],
                    "rep_id": row["rep_id"],
                    "container_id": row["container_id"],
                    "window_id": w,
                }
            )

    return pd.DataFrame(rows)

def generate_fixed_length_windows_centered(
    containers: pd.DataFrame,
    fs: int,
    window_sec: float,
) -> pd.DataFrame:
    """
    Split each container into fixed-length windows (NO overlap).
    If container length is not a perfect multiple, crop symmetrically
    at start and end so windows are centered.
    """

    window_size = int(round(window_sec * fs))
    rows = []

    for _, row in containers.iterrows():

        c_start = int(row["start_idx"])
        c_end = int(row["end_idx"])
        container_len = c_end - c_start

        n_windows = container_len // window_size

        if n_windows == 0:
            continue

        usable_len = n_windows * window_size
        leftover = container_len - usable_len

        crop_start = leftover // 2
        new_start = c_start + crop_start

        for w in range(n_windows):

            start_idx = new_start + w * window_size
            end_idx = start_idx + window_size

            rows.append(
                {
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "label": row["label"],
                    "rep_id": row["rep_id"],
                    "container_id": row["container_id"],
                    "window_id": w,
                }
            )

    return pd.DataFrame(rows)