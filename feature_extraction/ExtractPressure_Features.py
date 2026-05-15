import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

"""
from Maria_code.data_analysis.get_Time_Domain_features_of_signal import get_Time_Domain_features_of_signal
from Maria_code.data_analysis.get_Freq_Domain_features_of_signal import get_Freq_Domain_features_of_signal
"""

from feature_extraction.get_Time_Domain_features_of_signal import (
    get_Time_Domain_features_of_signal,
)
from feature_extraction.get_Freq_Domain_features_of_signal import (
    get_Freq_Domain_features_of_signal,
)
from feature_extraction.fsr_features import (
    per_sample_features,
    aggregate_per_window,
    aggregate_per_rep,
)
from feature_extraction.resample_windows import downsample_channel
from feature_extraction.create_feature_windows import build_boundaries
from feature_extraction.create_fixed_length_feature_windows import build_containers, generate_fixed_length_windows_centered

"Based on Royas code for extracting IMU features"


def ExtractPressure_Features(
    fsr_data, sensor_name, window_length, mean_fsr, fs, feature_space="baseline"
):
    """
    Extracts features from the FSR data from one side.

    Valid feature spaces:

    """
    if not isinstance(fsr_data, pd.DataFrame):
        fsr_data = pd.read_csv(fsr_data)

    """EXTRACT COLUMNS"""
    time_data = fsr_data["ReconstructedTime"]

    fsr_columns = [
        f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)
    ]  # "Fsr.01" to "Fsr.16"
    fsr_data = fsr_data[fsr_columns]  # Select only the FSR columns

    # Define a list to store features for each window
    all_window_features = []

    # Calculate the number of windows
    num_samples = len(time_data)
    num_windows = num_samples // window_length
    print(f"Number of fsr  windows: {num_windows}")

    for i in range(num_windows):
        # Define the start and end index for the window
        start_idx = i * window_length
        end_idx = start_idx + window_length

        if feature_space == "baseline":
            aggregated_fsr_features = {}
        elif feature_space == "expanded+baseline":
            fsr_win = fsr_data[start_idx:end_idx]
            per_sample_feature_df = per_sample_features(fsr_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_rep(
                per_sample_feature_df, sensor_name
            )
        elif feature_space == "expanded_only":
            fsr_win = fsr_data[start_idx:end_idx]
            per_sample_feature_df = per_sample_features(fsr_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_rep(
                per_sample_feature_df, sensor_name
            )
            window_features = {**aggregated_fsr_features}
        else:
            fsr_win = fsr_data[start_idx:end_idx]
            per_sample_feature_df = per_sample_features(fsr_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_rep(
                per_sample_feature_df, sensor_name, start_idx, end_idx
            )

        if mean_fsr == False and not feature_space == "expanded_only":
            sum_fsr = fsr_data.sum(axis=1)

            window_fsr_sum = sum_fsr[start_idx:end_idx]

            window_features_fsr_sum_Time = get_Time_Domain_features_of_signal(
                window_fsr_sum, f"fsr_sum_{sensor_name}"
            )
            window_features_fsr_sum_Freq = get_Freq_Domain_features_of_signal(
                window_fsr_sum, f"fsr_sum_{sensor_name}", fs
            )

            if feature_space == "time_only":
                window_features = {**window_features_fsr_sum_Time}
            elif feature_space == "freq_only":
                window_features == {**window_features_fsr_sum_Freq}
            else:
                window_features = {
                    **window_features_fsr_sum_Time,
                    **window_features_fsr_sum_Freq,
                    **aggregated_fsr_features,
                }

        if mean_fsr == True and not feature_space == "expanded_only":
            average_fsr = fsr_data.mean(axis=1)

            window_fsr_aver = average_fsr[start_idx:end_idx]

            window_features_fsr_aver_Time = get_Time_Domain_features_of_signal(
                window_fsr_aver, f"fsr_aver_{sensor_name}"
            )
            window_features_fsr_aver_Freq = get_Freq_Domain_features_of_signal(
                window_fsr_aver, f"fsr_aver_{sensor_name}", fs
            )

            if feature_space == "time_only":
                window_features = {**window_features_fsr_aver_Time}
            elif feature_space == "freq_only":
                print("freq only")
                window_features = {**window_features_fsr_aver_Freq}
                print("frequency only wondow features", window_features)
            elif feature_space == "time_only+exp_FSR":
                window_features = {
                    **window_features_fsr_aver_Time,
                    **aggregated_fsr_features,
                }
            elif feature_space == "freq_only+exp_FSR":
                window_features = {
                    **window_features_fsr_aver_Freq,
                    **aggregated_fsr_features,
                }
            else:
                window_features = {
                    **window_features_fsr_aver_Time,
                    **window_features_fsr_aver_Freq,
                    **aggregated_fsr_features,
                }

        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)

    return feature_df


def ExtractPressure_Features_repetitions_based(
    fsr_data,
    sensor_name,
    mean_fsr,
    fs,
    feature_space="baseline",
    target_num_samples=350,
):
    """
    Extracts features from the FSR data from one side.

    Valid feature spaces:

    """
    if not isinstance(fsr_data, pd.DataFrame):
        fsr_data = pd.read_csv(fsr_data)

    print("FSR data cols", fsr_data.columns)
    """EXTRACT COLUMNS"""
    time_data = fsr_data["ReconstructedTime"]

    # Define a list to store features for each window
    all_window_features = []
    all_window_labels = []

    boundaries = build_boundaries(fsr_data, fixed_len=target_num_samples)

    fsr_columns = [
        f"Fsr.{str(sensor).zfill(2)}" for sensor in range(1, 17)
    ]  # "Fsr.01" to "Fsr.16"
    fsr_data_filtered = fsr_data[fsr_columns]  # Select only the FSR columns

    for _, row in boundaries.iterrows():
        start_idx = int(row.start_idx)
        end_idx = int(row.end_idx)
        window_label = fsr_data.iloc[start_idx]["label"]
        window_rep_id = fsr_data.iloc[start_idx]["rep_id"]

        fsr_win = fsr_data_filtered[start_idx:end_idx]
        fsr_win = fsr_win.copy()
        print(f"FSR sanity check [is label consistent with rep_id?]")
        print("checking....")
        
        
        
        if window_label not in ["walking","standing","sitting","neutral_load","neutral_load_left","neutral_load_right"]:
            print("Label is", window_label)
            if window_label in fsr_data.iloc[start_idx]["rep_id"]:
                print("Sanity passed")
            else:
                print(
                    f"Rep id is {fsr_data.iloc[start_idx]['rep_id']}, while label is {window_label}"
                )
                print("Sleeping for 60 seconds")
                time.sleep(60)
        else: 
            print("Not relevant, label is continuous...")
        # if window_label in fsr_data.iloc[start_idx]["rep_id"]:
        #     print("Sanity OKAY:)")
        # else:
        #     print("Sanity NOT OKAY:((")
        #     print("rep_id", fsr_data.iloc[start_idx]["rep_id"])
        #     print("label:", window_label)
        #     time.sleep(60)

        time_win = time_data[start_idx:end_idx]

        # plt.plot(time_win, fsr_win)
        # plt.title(window_label)

        # plt.show()
        if len(fsr_win) < target_num_samples:
            print("DROPPING WINDOW THAT IS SHORTER THAN TARGET LENGTH")
            print(
                f"length is {len(fsr_win)}, rep_id is {fsr_data.iloc[start_idx]['rep_id']}"
            )
            time.sleep(10)
            continue
        all_window_labels.append(window_label)
        ds_win = pd.DataFrame()
        ds_win["label"] = window_label
        ds_win["rep_id"] = window_rep_id
        for col in fsr_win.columns:
            ds_fsr_col = downsample_channel(
                fsr_win[col], target_num_samples=target_num_samples
            )
            ds_win[col] = ds_fsr_col

        if feature_space == "baseline":
            aggregated_fsr_features = {}
        elif feature_space == "expanded+baseline":
            per_sample_feature_df = per_sample_features(ds_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_window(
                per_sample_feature_df, sensor_name, start_idx, end_idx
            )
            # time.sleep(2)
        elif feature_space == "expanded_only":
            per_sample_feature_df = per_sample_features(ds_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_window(
                per_sample_feature_df, sensor_name, start_idx, end_idx
            )
            window_features = {**aggregated_fsr_features}
        else:
            per_sample_feature_df = per_sample_features(ds_win, sensor_name, None)
            aggregated_fsr_features = aggregate_per_window(
                per_sample_feature_df, sensor_name, start_idx, end_idx
            )

        if mean_fsr == False and not feature_space == "expanded_only":
            print("Invalid, use mean not sum")
            # sum_fsr = fsr_data_filtered.sum(axis=1)

            # window_fsr_sum = sum_fsr[start_idx:end_idx]

            # window_features_fsr_sum_Time = get_Time_Domain_features_of_signal(window_fsr_sum, f"fsr_sum_{sensor_name}")
            # window_features_fsr_sum_Freq = get_Freq_Domain_features_of_signal(window_fsr_sum, f"fsr_sum_{sensor_name}", fs)

            # if feature_space == 'time_only':
            #     window_features = {**window_features_fsr_sum_Time}
            # elif feature_space == 'freq_only':
            #     window_features == {**window_features_fsr_sum_Freq}
            # else:
            #     window_features = {**window_features_fsr_sum_Time,
            #                     **window_features_fsr_sum_Freq,
            #                     **aggregated_fsr_features}

        if mean_fsr == True and not feature_space == "expanded_only":
            average_fsr = fsr_data_filtered.mean(axis=1)

            window_fsr_aver = average_fsr[start_idx:end_idx]

            if len(window_fsr_aver) < target_num_samples:
                print("DROPPING WINDOW THAT IS SHORTER THAN TARGET LENGTH")
                print(
                    f"length is {len(window_fsr_aver)}, rep_id is {fsr_data.iloc[start_idx]['rep_id']}"
                )
                time.sleep(10)
                continue
            # all_window_labels.append(window_label)
            t_orig = np.linspace(0, 1, len(window_fsr_aver))
            # plt.plot(t_orig, window_fsr_aver, color='red')
            window_fsr_aver = downsample_channel(
                window_fsr_aver, target_num_samples=target_num_samples
            )
            t_ds = np.linspace(0, 1, len(window_fsr_aver))
            # plt.plot(t_ds, window_fsr_aver, color='blue')
            # plt.title(fsr_data.iloc[start_idx]['rep_id'])
            # plt.show()

            window_features_fsr_aver_Time = get_Time_Domain_features_of_signal(
                window_fsr_aver, f"fsr_aver_{sensor_name}"
            )
            window_features_fsr_aver_Freq = get_Freq_Domain_features_of_signal(
                window_fsr_aver, f"fsr_aver_{sensor_name}", fs
            )

            if feature_space == "time_only":
                window_features = {**window_features_fsr_aver_Time}
            elif feature_space == "freq_only":
                print("freq only")
                window_features = {**window_features_fsr_aver_Freq}
                print("frequency only window features", window_features)
            elif feature_space == "time_only+exp_FSR":
                window_features = {
                    **window_features_fsr_aver_Time,
                    **aggregated_fsr_features,
                }
            elif feature_space == "freq_only+exp_FSR":
                window_features = {
                    **window_features_fsr_aver_Freq,
                    **aggregated_fsr_features,
                }
            else:
                window_features = {
                    **window_features_fsr_aver_Time,
                    **window_features_fsr_aver_Freq,
                    **aggregated_fsr_features,
                }

        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)

    return feature_df, all_window_labels


def ExtractPressure_Features_window_based(
    fsr_data,
    sensor_name,
    fs,
    window_sec=3.5,
    use_rep_id=True,
    feature_space="baseline",   # "baseline" or "expanded+baseline"
    mean_fsr=True,
):
    """
    Window-based FSR feature extraction (no resampling).

    Parameters
    ----------
    fsr_data : pd.DataFrame or str
        FSR dataframe or path to csv.
    sensor_name : str
        Sensor/side name, e.g. "Left" or "Right".
    fs : int or float
        Sampling rate.
    window_sec : float
        Window length in seconds.
    use_rep_id : bool
        If True, build containers from rep_id when available.
        Otherwise use consecutive runs of the same label.
    feature_space : str
        Either "baseline" or "expanded+baseline".
    mean_fsr : bool
        If True, compute baseline features from the mean FSR signal.
        Currently expected to be True.

    Returns
    -------
    feature_df : pd.DataFrame
        Metadata columns + feature columns, one row per window.
    all_window_labels : list
        Label per window.
    """

    if feature_space not in ["baseline", "expanded+baseline"]:
        raise ValueError("feature_space must be either 'baseline' or 'expanded+baseline'")

    if not mean_fsr:
        raise ValueError("Currently only mean_fsr=True is supported.")

    if not isinstance(fsr_data, pd.DataFrame):
        fsr_data = pd.read_csv(fsr_data)

    fsr_data = fsr_data.reset_index(drop=True).copy()

    # -----------------------------------------------------
    # Build containers and windows
    # -----------------------------------------------------
    containers = build_containers(fsr_data, use_rep_id=use_rep_id)

    windows = generate_fixed_length_windows_centered(
        containers=containers,
        fs=fs,
        window_sec=window_sec,
    )

    if windows.empty:
        print(f"No windows generated for FSR {sensor_name}")
        return pd.DataFrame(), []

    # -----------------------------------------------------
    # Select FSR columns
    # -----------------------------------------------------
    fsr_columns = [f"Fsr.{str(i).zfill(2)}" for i in range(1, 17)]
    fsr_df = fsr_data[fsr_columns]

    all_window_features = []
    all_window_labels = []
    all_meta = []

    # -----------------------------------------------------
    # Loop over windows
    # -----------------------------------------------------
    for _, w in windows.iterrows():
        s = int(w["start_idx"])
        e = int(w["end_idx"])

        fsr_win = fsr_df.iloc[s:e]
        label = w["label"]
        rep_id = w["rep_id"]

        # -----------------------------
        # Metadata
        # -----------------------------
        all_meta.append(
            {
                "label": label,
                "rep_id": rep_id,
                "container_id": w["container_id"],
                "window_id": w["window_id"],
                "start_idx": s,
                "end_idx": e,
            }
        )
        all_window_labels.append(label)

        # -----------------------------
        # Baseline features
        # -----------------------------
        mean_signal = fsr_win.mean(axis=1)

        baseline_time_feats = get_Time_Domain_features_of_signal(
            mean_signal,
            f"fsr_aver_{sensor_name}",
        )

        baseline_freq_feats = get_Freq_Domain_features_of_signal(
            mean_signal,
            f"fsr_aver_{sensor_name}",
            fs,
        )

        baseline_features = {
            **baseline_time_feats,
            **baseline_freq_feats,
        }

        # -----------------------------
        # Expanded features (optional)
        # -----------------------------
        if feature_space == "expanded+baseline":
            tmp = fsr_win.copy()
            tmp["label"] = label
            tmp["rep_id"] = rep_id

            per_sample_df = per_sample_features(tmp, sensor_name, None)

            expanded_features = aggregate_per_window(
                per_sample_df,
                sensor_name,
                s,
                e,
            )

            window_features = {
                **baseline_features,
                **expanded_features,
            }

        else:  # baseline
            window_features = baseline_features

        all_window_features.append(window_features)

    feature_df = pd.DataFrame(all_window_features)
    meta_df = pd.DataFrame(all_meta)

    return feature_df, all_window_labels