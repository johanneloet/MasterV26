"""
File purpose: segmentation of repetitions based on manual observations in acceleration plots.

This file is written by Johanne, though the approach is inspired by Maria's repetition segmentation approach.
"""
# Imports
#from get_paths import get_test_file_paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def plot_activity_accelerations_peaks_and_magnitude(
        df,
        activity_label,
        height=1100,
        distance=1100,
        peak_indices=None,
        colors=None,
        figsize=(12, 6)):
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
            "mag": "#1F4E99",   # strong, clear blue (focus signal)
            "x":   "#B0C4DE",   # light steel blue (faded)
            "y":   "#C5D1E0",   # very light grey-blue
            "z":   "#D6DFEB"    # near-background blue-grey
        }
        
        
    # Suppose your DataFrame has a non-continuous or meaningful index
    subset = df[df["label"] == activity_label].copy()

    # Compute magnitude
    subset["Accel mag"] = np.sqrt(subset["Axl.X"]**2 + subset["Axl.Y"]**2 + subset["Axl.Z"]**2)

    # Call find_peaks on the NumPy array
    if peak_indices == None:
        local_peak_indices, properties = find_peaks(subset["Accel mag"].to_numpy(), height=1100, distance=1100)

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
    plt.plot(subset["ReconstructedTime"], subset["Axl.X"],
             label="Axl.X", color=colors["x"], linewidth=1.5)
    plt.plot(subset["ReconstructedTime"], subset["Axl.Y"],
             label="Axl.Y", color=colors["y"], linewidth=1.5)
    plt.plot(subset["ReconstructedTime"], subset["Axl.Z"],
             label="Axl.Z", color=colors["z"], linewidth=1.5)
    plt.plot(subset["ReconstructedTime"], subset["Accel mag"],
            label="Axl. magnitude", color=colors["mag"], linewidth=1.5)

    # Plot peaks
    plt.plot(subset["ReconstructedTime"].iloc[local_peak_indices],
             subset["Accel mag"].iloc[local_peak_indices],
             "rx", label="Peaks")

    # --- Style ---
    plt.title(f"Acceleration Components over Time ({activity_label})",
              fontsize=16, weight='bold')
    plt.xlabel("Time (s)", fontsize=13)
    plt.ylabel("Acceleration (mg)", fontsize=13)
    plt.legend(title="Axes", fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("acceleration_peaks.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    return subset, original_peak_indices, local_peak_indices



def get_start_stop_times_from_peaks(df, peaks, activity_name, num_reps = 6,time_col="ReconstructedTime", L_R_alternate=False):
    peaks = np.sort(np.asarray(peaks, dtype=int))
    if len(peaks) < 2:
        raise ValueError("Need at least two peaks to form intervals.")

    times = df[time_col].to_numpy()

    start_stop_times_reps = {}
    intervals = []
    assert len(peaks) >= num_reps
    for i in range(num_reps):
        rep_start_idx = peaks[i]
        rep_stop_idx = peaks[i + 1]
        rep_start_time = times[rep_start_idx]
        rep_stop_time = times[rep_stop_idx]
        if L_R_alternate == True:
            # IMPORTANT ASSUMPTION: the ordering during data collection was left first, then right. We can then assume that every odd repetition number is a left rep
            # and any even rep number is a right side rep. This was protocol during the prelim data collection. Any deviations will be noted in the thesis. 
            # Therefore read data collection notes thoroughly!
            if (i+1) % 2 == 0:
                rep_name = f"{activity_name}_right_{i+1}"
            else:
                rep_name = f"{activity_name}_left_{i+1}"
        else:    
            rep_name = f"{activity_name}_{i+1}"

        start_stop_times_reps[rep_name] = (rep_start_time, rep_stop_time)
        intervals.append((rep_name, rep_start_time, rep_stop_time))

    rep_intervals_df = pd.DataFrame(intervals, columns=["rep_id", "start_time", "stop_time"])
    return start_stop_times_reps, rep_intervals_df


# def read_and_apply_rep_ids(
#     rep_activity_list : list, 
#     static_activity_list : list, 
#     data_csv_path_list : list,
#     lower_sample_lim_muse : int,
    
#     """_summary_

#     Args:
#         rep_activity_list (list): list of activities that are performed in repetitions, and therefore have variable lengths.
#         static_activity_list (list): list of static activities 
#         data_csv_path_list (list): _description_
#         lower_sample_lim_muse.
#     """
    