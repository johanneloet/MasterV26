from scipy import signal
import numpy as np
import pandas as pd
import math


def downsample_channel(df_channel, target_num_samples):
    """
    Downsample a single IMU channel (DataFrame or Series) to a fixed number of samples.
    Only performs downsampling (raises error if input is shorter than target length).

    Chatgpt aided the development of this function.
    """
    # Extract the column
    if isinstance(df_channel, pd.DataFrame):
        if df_channel.shape[1] != 1:
            raise ValueError(
                "DataFrame must have exactly one column for a single channel."
            )
        col_name = df_channel.columns[0]
        x = df_channel[col_name].to_numpy()
    else:
        col_name = df_channel.name if df_channel.name else "channel"
        x = df_channel.to_numpy()

    n = len(x)
    if n == 0:
        raise ValueError("Input channel has no samples.")
    if n < target_num_samples:
        raise ValueError(
            f"Cannot downsample: input has {n} samples, "
            f"but target_num_samples is {target_num_samples} (must be smaller)."
        )
    if n == target_num_samples:
        return pd.Series(x, name=col_name)

    # Handle NaNs
    if np.isnan(x).any():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()

    # Polyphase resample (anti-aliasing)
    # print("downsampling")
    g = math.gcd(target_num_samples, n)
    up = target_num_samples // g
    down = n // g
    # print("right before resample_poly, gdc is", g, 'up is', up, 'down is', down)

    y = signal.resample_poly(x, up, down, padtype="edge")

    # exact length
    if len(y) > target_num_samples:
        y = y[:target_num_samples]
    elif len(y) < target_num_samples:
        y = np.pad(y, (0, target_num_samples - len(y)), mode="edge")

    return pd.Series(y, name=col_name)
