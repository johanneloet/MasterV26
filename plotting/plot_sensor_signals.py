import matplotlib.pyplot as plt
import numpy as np

from utils import get_imu_cols, get_fsr_cols

# plotting functions made withthe aid of chatgpt.

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



def plot_all_sensors(
    dfs, 
    time_mode="Timestamp", 
    names=None, 
    t_start=None, 
    t_end=None,
    imu_components=["Axl", "Gyr", "Mag"],
    fsr_mode="All",
    fsr_mean_name="FSR_mean",
    plot_fsr=True
                    ):
    """
    Plot multiple sensors in separate subplots.

    Sensor type is automatically detected:
        - If any FSR column exists -> sensor is FSR
        - Else -> sensor is IMU

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
        
        # If we're in reconstructed time mode, optionally window by seconds
        df_plot = df
        x_plot = x
        if time_mode == "ReconstructedTime" and (t_start is not None or t_end is not None):
            # x is expected to be "seconds since start" (or similar)
            mask = True
            if t_start is not None:
                mask = (x >= t_start)
            if t_end is not None:
                mask = mask & (x <= t_end) if not isinstance(mask, bool) else (x <= t_end)

            # Apply mask safely
            df_plot = df.loc[mask].copy()
            x_plot = x[mask]

        # Detect sensor type automatically
        has_fsr = any(c in df.columns for c in fsr_cols)
        if has_fsr:
            if fsr_mode == 'Mean':
                df_plot['fsr_mean'] = df_plot[fsr_cols].mean(axis=1)
                cols_to_plot = ['fsr_mean']
            elif fsr_mode == 'All':
                cols_to_plot = [c for c in fsr_cols if c in df_plot.columns]
        else:
            plot_imu_cols = []
            for col in imu_cols:
                component = col.split('.')[0]
                if component in imu_components:
                    plot_imu_cols.append(col)
            cols_to_plot = [c for c in plot_imu_cols if c in df_plot.columns]

        for c in cols_to_plot:
            ax.plot(x_plot, df_plot[c], label=c)

        ax.set_title(f"{name} plotted by {time_mode}")
        ax.grid(True)
        ax.legend(fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(1.02,1))

    axes[-1].set_xlabel(time_mode)
    fig.tight_layout()
    plt.show()