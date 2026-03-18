"""
File purpose: segmentation of repetitions based on manual observations in acceleration plots.

This file is written by Johanne, though the approach is inspired by Maria's repetition segmentation approach.
"""

# Imports
# from get_paths import get_test_file_paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import re


def plot_activity_accelerations_peaks_and_magnitude(
    df,
    activity_label,
    height=1100,
    distance=1100,
    peak_indices=None,
    colors=None,
    figsize=(12, 6),
):
    """
    Plot acceleration components (X, Y, Z), magnitude, and detected peaks
    for a given activity label from an IMU DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with accelerometer data and a 'label' column and a 'ReconstructedTime' column.
    activity_label : str
        The movement label to filter (e.g., 'hand_up_back').
    height : float, optional
        Minimum height (in mg) for peak detection.
    distance : int, optional
        Minimum distance (in samples) between peaks.
    peak_indices : array-like, optional
        Custom peak indices to plot. If None, peaks will be auto-detected.
    colors : dict, optional
        Color dictionary for the curves.
    figsize : tuple, optional
        Figure size for the plot.

    Returns
    -------
    subset : pandas.DataFrame
        Subset of the data corresponding to the selected label.
    peak_indices : np.ndarray
        Indices of the plotted peaks.
    properties : dict or None
        Properties from scipy.signal.find_peaks if peaks were auto-detected.
    """

    if colors is None:
        colors = {
            "mag": "#1F4E99",  # strong, clear blue (focus signal)
            "x": "#B0C4DE",  # light steel blue (faded)
            "y": "#C5D1E0",  # very light grey-blue
            "z": "#D6DFEB",  # near-background blue-grey
        }

    # Suppose your DataFrame has a non-continuous or meaningful index
    subset = df[df["label"] == activity_label].copy()

    # Compute magnitude
    subset["Accel mag"] = np.sqrt(
        subset["Axl.X"] ** 2 + subset["Axl.Y"] ** 2 + subset["Axl.Z"] ** 2
    )

    # Call find_peaks on the NumPy array
    if peak_indices == None:
        local_peak_indices, properties = find_peaks(
            subset["Accel mag"].to_numpy(), height=height, distance=distance
        )

        # Map back to original indices
        original_peak_indices = subset.index[local_peak_indices]

        print("Local:", local_peak_indices[:5])
        print("Original:", original_peak_indices[:5])
    else:
        local_peak_indices = peak_indices
        original_peak_indices = None
        properties = None

    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.plot(
        subset["ReconstructedTime"],
        subset["Axl.X"],
        label="Axl.X",
        color=colors["x"],
        linewidth=1.5,
    )
    plt.plot(
        subset["ReconstructedTime"],
        subset["Axl.Y"],
        label="Axl.Y",
        color=colors["y"],
        linewidth=1.5,
    )
    plt.plot(
        subset["ReconstructedTime"],
        subset["Axl.Z"],
        label="Axl.Z",
        color=colors["z"],
        linewidth=1.5,
    )
    plt.plot(
        subset["ReconstructedTime"],
        subset["Accel mag"],
        label="Axl. magnitude",
        color=colors["mag"],
        linewidth=1.5,
    )

    # Plot peaks
    plt.plot(
        subset["ReconstructedTime"].iloc[local_peak_indices],
        subset["Accel mag"].iloc[local_peak_indices],
        "rx",
        label="Peaks",
    )

    # --- Style ---
    plt.title(
        f"Acceleration Components over Time ({activity_label})",
        fontsize=16,
        weight="bold",
    )
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Acceleration (mg)", fontsize=13)
    plt.legend(title="Axes", fontsize=11)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig("acceleration_peaks.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return subset, original_peak_indices, local_peak_indices


def get_start_stop_times_from_peaks(
    df,
    peaks,
    activity_name,
    num_reps=6,
    time_col="ReconstructedTime",
    side_mode=None,  # None, "alternate", "sequential"
):
    peaks = np.sort(np.asarray(peaks, dtype=int))
    if len(peaks) < 2:
        raise ValueError("Need at least two peaks to form intervals.")

    times = df[time_col].to_numpy()

    if len(peaks) < num_reps + 1:
        raise ValueError("Not enough peaks for the requested number of reps.")

    start_stop_times_reps = {}
    intervals = []

    for i in range(num_reps):
        rep_start_idx = peaks[i]
        rep_stop_idx = peaks[i + 1]
        rep_start_time = times[rep_start_idx]
        rep_stop_time = times[rep_stop_idx]

        rep_number = i + 1

        # Side variations
        # Important assumption!! Data is collected left first then right
        if side_mode == "alternate":
            # L, R, L, R...
            if rep_number % 2 == 0:
                side = "right"
            else:
                side = "left"
            rep_name = f"{activity_name}_{side}_{rep_number}"

        elif side_mode == "sequential":
            # First half left, second half right
            half = num_reps // 2
            if i < half:
                side = "left"
            else:
                side = "right"
            rep_name = f"{activity_name}_{side}_{rep_number}"

        else:
            rep_name = f"{activity_name}_{rep_number}"

        start_stop_times_reps[rep_name] = (rep_start_time, rep_stop_time)
        intervals.append((rep_name, rep_start_time, rep_stop_time))

    rep_intervals_df = pd.DataFrame(
        intervals, columns=["rep_id", "start_time", "stop_time"]
    )

    return start_stop_times_reps, rep_intervals_df


def assign_rep_ids(
    sensor_dfs,
    output_dir,
    numbered_labels,
    static_labels=[
        "standing",
        "sitting",
        "walking",
        "neutral_load_left",
        "neutral_load_right",
    ],
    save_files = True
):
    valid_dfs = {}
    skipped = []

    # Validate rep_id presence
    for name, df in sensor_dfs.items():
        if "rep_id" not in df.columns:
            print(f"⚠️ WARNING: '{name}' skipped (missing 'rep_id' column)")
            skipped.append(name)
        else:
            valid_dfs[name] = df

    if not valid_dfs:
        print("❌ No valid dataframes to process. Exiting.")
        return sensor_dfs

    dfs = valid_dfs.values()

    ## Clear rep_id for rows that are not already labeled
    numbered_pattern = "|".join(numbered_labels + static_labels)

    for df in dfs:
        mask = df["label"].str.contains(numbered_pattern, case=False, na=False)
        df.loc[~mask, "rep_id"] = None

    # Use one dataframe to get activities
    activities = next(iter(dfs))["label"].unique().tolist()
    print("Activities:", activities)

    for activity in activities:
        if activity in static_labels:
            continue

        # handle labels that already have assigned rep ids in their label.
        if re.search(r"_\d+$", str(activity).lower()):
            for df in dfs:
                mask = df["label"].astype(str).str.lower().eq(str(activity).lower())

                # rep_id = original label
                df.loc[mask, "rep_id"] = df.loc[mask, "label"]

                # label = base
                df.loc[mask, "label"] = (
                    df.loc[mask, "label"]
                    .astype(str)
                    .str.extract(r"^(.+?)_\d+$", expand=False)
                    .str.lower()
                )

            print(
                f"✅ Normalized numbered label '{activity}' -> base label + rep_id preserved."
            )

        else:
            rep_start_stop_time_path = (
                output_dir / f"start_stop_rep_times_{activity}.csv"
            )
            try:
                start_stop_df = pd.read_csv(rep_start_stop_time_path)
            except FileNotFoundError:
                print(f"⚠️ WARNING: Missing CSV for activity '{activity}', skipping")
                continue

            for _, rep in start_stop_df.iterrows():
                rep_id = rep["rep_id"]
                start_time = rep["start_time"]
                stop_time = rep["stop_time"]

                for df in dfs:
                    mask = (
                        (df["label"] == activity)
                        & (df["ReconstructedTime"] >= start_time)
                        & (df["ReconstructedTime"] < stop_time)
                    )
                    df.loc[mask, "rep_id"] = rep_id

            print(f"✅ Assigned rep_ids for {activity}")
    if save_files == True:
        for name, df in valid_dfs.items():
            save_path = output_dir / f"{name}_with_rep_ids.csv"
            df.to_csv(save_path, index=False)
            print(f"💾 Saved: {save_path}")

    return sensor_dfs
