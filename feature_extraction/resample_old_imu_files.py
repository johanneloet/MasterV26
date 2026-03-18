from scipy.signal import resample_poly
from fractions import Fraction
import numpy as np
import pandas as pd


def resample_imu_dataframe(
    df: pd.DataFrame,
    original_fs: int,
    target_fs: int,
):
    if original_fs == target_fs:
        return df.reset_index(drop=True).copy()

    if "ReconstructedTime" not in df.columns:
        raise ValueError("Missing required time column: ReconstructedTime")

    df = df.sort_values("ReconstructedTime").reset_index(drop=True)

    ratio = Fraction(target_fs, original_fs).limit_denominator()
    up = ratio.numerator
    down = ratio.denominator

    print(f"Resampling: up={up}, down={down}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if "ReconstructedTime" in numeric_cols:
        numeric_cols.remove("ReconstructedTime")

    resampled_channels = {}
    for col in numeric_cols:
        resampled_channels[col] = resample_poly(df[col].values, up, down)

    n_new = len(next(iter(resampled_channels.values())))

    # Use full original time span
    t_old = df["ReconstructedTime"].values
    new_time = np.linspace(t_old[0], t_old[-1], n_new)

    df_new = pd.DataFrame({"ReconstructedTime": new_time})

    for col in numeric_cols:
        df_new[col] = resampled_channels[col]

    # True nearest-neighbour mapping for categorical columns
    idx_right = np.searchsorted(t_old, new_time, side="left")
    idx_right = np.clip(idx_right, 0, len(t_old) - 1)
    idx_left = np.clip(idx_right - 1, 0, len(t_old) - 1)

    choose_left = np.abs(new_time - t_old[idx_left]) <= np.abs(t_old[idx_right] - new_time)
    nearest_idx = np.where(choose_left, idx_left, idx_right)

    if "label" in df.columns:
        df_new["label"] = df["label"].iloc[nearest_idx].values

    if "rep_id" in df.columns:
        df_new["rep_id"] = df["rep_id"].iloc[nearest_idx].values

    return df_new