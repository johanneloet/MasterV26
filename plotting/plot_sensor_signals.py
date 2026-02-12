import matplotlib.pyplot as plt
import numpy as np

from utils import get_imu_cols, get_fsr_cols


def get_time_axis(df, mode="Timestamp"):
    """
    Allowed modes:
        'Timestamp'
        'ReconstructedTime'
        'index'
    """
    if mode == "index":
        return np.arange(len(df))

    if mode in df.columns:
        t = df[mode].to_numpy()
        return t - t[0]

    raise ValueError(f"Could not find time axis. No {mode} column in provided dataframe.")


def plot_fsr(side, df, fsr_cols, time_mode="Timestamp"):

    x = get_time_axis(df, time_mode)

    plt.figure(figsize=(12, 5))

    for c in fsr_cols:
        if c in df.columns:
            plt.plot(x, df[c], label=c)

    plt.title(f"FSR {side}")
    plt.xlabel(time_mode)
    plt.ylabel("FSR value")
    plt.grid(True)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()



def plot_single_imu(location, df, imu_cols, time_mode="Timestamp"):

    x = get_time_axis(df, time_mode)

    plt.figure(figsize=(12, 5))

    for c in imu_cols:
        if c in df.columns:
            plt.plot(x, df[c], label=c)

    plt.title(f"{location} IMU signal")
    plt.xlabel(time_mode)
    plt.ylabel("IMU signal")
    plt.grid(True)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()



def plot_all_sensors(dfs, time_mode="Timestamp", names=None):
    """
    Plot multiple sensors in separate subplots.

    Sensor type is automatically detected:
        - If any FSR column exists → sensor is FSR
        - Else → sensor is IMU

    dfs: list of DataFrames
    time_mode: 'Timestamp', 'ReconstructedTime', or 'index'
    names: optional list of sensor names
    """
    imu_cols = get_imu_cols()
    fsr_cols = get_fsr_cols()

    n_sensors = len(dfs)
    if names is None:
        names = [f"Sensor {i+1}" for i in range(n_sensors)]

    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3*n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for ax, df, name in zip(axes, dfs, names):
        if df is None:
            continue
        x = get_time_axis(df, time_mode)

        # Detect sensor type automatically
        has_fsr = any(c in df.columns for c in fsr_cols)
        if has_fsr:
            cols_to_plot = [c for c in fsr_cols if c in df.columns]
        else:
            cols_to_plot = [c for c in imu_cols if c in df.columns]

        for c in cols_to_plot:
            ax.plot(x, df[c], label=c)

        ax.set_title(f"{name} plotted by {time_mode}")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(1.02,1))

    axes[-1].set_xlabel(time_mode)
    fig.tight_layout()
    plt.show()