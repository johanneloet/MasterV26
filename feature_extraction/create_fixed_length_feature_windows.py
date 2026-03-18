import pandas as pd
import numpy as np

# def build_containers(
#     df: pd.DataFrame,
#     use_rep_id: bool = True,
# ) -> pd.DataFrame:
#     """
#     Builds containers based on rep_id if available (lab data) and label if not (real-world data).
#     Similar to build boundaries from previously!
#     """

#     df = df.reset_index(drop=True).copy()

#     if use_rep_id and "rep_id" in df.columns and df["rep_id"].notna().any():
#         key = df["rep_id"].fillna("none").astype(str).to_numpy()
#         container_prefix = "rep"
#     else:
#         if "label" not in df.columns:
#             raise ValueError("No label column available - cannot build container for segmentation.")
#         key = df["label"].astype(str).to_numpy()
#         container_prefix = "labelrun"

#     change_points = np.where(key[:-1] != key[1:])[0] + 1
#     starts = np.concatenate(([0], change_points))
#     ends = np.concatenate((change_points, [len(df)]))

#     rows = []
#     for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
#         rows.append(
#             {
#                 "start_idx": int(start_idx),
#                 "end_idx": int(end_idx),  # exclusive
#                 "label": df.iloc[start_idx]["label"] if "label" in df.columns else None,
#                 "rep_id": df.iloc[start_idx]["rep_id"] if "rep_id" in df.columns else None,
#                 "container_id": f"{container_prefix}_{i}",
#             }
#         )

#     return pd.DataFrame(rows)

def build_containers(
    df: pd.DataFrame,
    use_rep_id: bool = True,
    fixed_labels=(
        "standing",
        "sitting",
        "walking",
        "neutral_load_left",
        "neutral_load_right",
    ),
) -> pd.DataFrame:
    """
    Build coarse containers for segmentation.

    Rules:
    - If use_rep_id=False: segment by consecutive label runs.
    - If use_rep_id=True:
        * for fixed_labels: segment by consecutive label runs (ignore rep_id)
        * for all other labels: segment by rep_id changes within each label run,
          but only for rows where rep_id is not NaN
    """

    df = df.reset_index(drop=True).copy()
    print(df.head())

    if "label" not in df.columns:
        raise ValueError("No label column available - cannot build containers.")

    labels = df["label"].fillna("").astype(str).str.lower().to_numpy()

    if use_rep_id and "rep_id" in df.columns:
        rep_ids_raw = df["rep_id"]
        rep_ids = rep_ids_raw.astype(str).to_numpy()
        valid_rep_mask = rep_ids_raw.notna().to_numpy()
    else:
        rep_ids = None
        valid_rep_mask = None

    # --- first split by label runs ---
    label_change_points = np.where(labels[:-1] != labels[1:])[0] + 1
    label_starts = np.concatenate(([0], label_change_points))
    label_ends = np.concatenate((label_change_points, [len(df)]))

    rows = []
    container_counter = 0

    for start_idx, end_idx in zip(label_starts, label_ends):
        label_value = labels[start_idx]

        # Case 1: not using rep_id at all
        if rep_ids is None:
            rows.append(
                {
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),  # exclusive
                    "label": df.iloc[start_idx]["label"],
                    "rep_id": df.iloc[start_idx]["rep_id"] if "rep_id" in df.columns else None,
                    "container_id": f"labelrun_{container_counter}",
                }
            )
            container_counter += 1
            continue

        # Case 2: fixed labels -> ignore rep_id, keep whole label run together
        if any(label_value.startswith(fl) or fl in label_value for fl in fixed_labels):
            rows.append(
                {
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),  # exclusive
                    "label": df.iloc[start_idx]["label"],
                    "rep_id": df.iloc[start_idx]["rep_id"] if "rep_id" in df.columns else None,
                    "container_id": f"fixedlabel_{container_counter}",
                }
            )
            container_counter += 1
            continue

        # Case 3: other labels -> split by rep_id within this label run,
        # but only use rows with non-NaN rep_id
        sub_rep_ids = rep_ids[start_idx:end_idx]
        sub_valid_mask = valid_rep_mask[start_idx:end_idx]

        if not np.any(sub_valid_mask):
            # no valid rep_id in this label run -> skip
            continue

        # Find contiguous valid stretches first
        valid_change_points = np.where(sub_valid_mask[:-1] != sub_valid_mask[1:])[0] + 1
        valid_starts = np.concatenate(([0], valid_change_points))
        valid_ends = np.concatenate((valid_change_points, [len(sub_valid_mask)]))

        for vs, ve in zip(valid_starts, valid_ends):
            # only keep stretches where rep_id is valid
            if not sub_valid_mask[vs]:
                continue

            abs_start = start_idx + vs
            abs_end = start_idx + ve

            # within this valid stretch, split whenever rep_id changes
            valid_rep_segment = rep_ids[abs_start:abs_end]
            rep_change_points = np.where(valid_rep_segment[:-1] != valid_rep_segment[1:])[0] + 1
            rep_starts = np.concatenate(([abs_start], abs_start + rep_change_points))
            rep_ends = np.concatenate((abs_start + rep_change_points, [abs_end]))

            for sub_start, sub_end in zip(rep_starts, rep_ends):
                rows.append(
                    {
                        "start_idx": int(sub_start),
                        "end_idx": int(sub_end),  # exclusive
                        "label": df.iloc[sub_start]["label"],
                        "rep_id": df.iloc[sub_start]["rep_id"],
                        "container_id": f"rep_{container_counter}",
                    }
                )
                container_counter += 1

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