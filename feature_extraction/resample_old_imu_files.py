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

    df = df.sort_values('ReconstructedTime').reset_index(drop=True)

    if 'ReconstructedTime' not in df.columns:
        raise ValueError(f"Missing required time column: {'ReconstructedTime'}")

    ratio = Fraction(target_fs, original_fs).limit_denominator()
    up = ratio.numerator
    down = ratio.denominator

    print(f"Resampling: up={up}, down={down}")

    # resample only numeric
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('ReconstructedTime')

    resampled_channels = {}

    for col in numeric_cols:
        resampled_channels[col] = resample_poly(
            df[col].values,
            up,
            down
        )

    n_new = len(next(iter(resampled_channels.values())))

    # new time vector
    start_t = df['ReconstructedTime'].iloc[0]
    new_time = start_t + np.arange(n_new) / target_fs

    df_new = pd.DataFrame({'ReconstructedTime': new_time})

    for col in numeric_cols:
        df_new[col] = resampled_channels[col]

    # label / rep_id -> nearest mapping
    t_old = df['ReconstructedTime'].values
    nearest_idx = np.searchsorted(t_old, new_time, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(df) - 1)

    if "label" in df.columns:
        df_new["label"] = df["label"].iloc[nearest_idx].values

    if "rep_id" in df.columns:
        df_new["rep_id"] = df["rep_id"].iloc[nearest_idx].values

    return df_new