import pandas as pd

from feature_extraction.ExtractIMU_Features import (
    ExtractIMU_Features,
    ExtractIMU_features_repetitions_based,
    ExtractIMU_features_window_based
)
from feature_extraction.ExtractPressure_Features import (
    ExtractPressure_Features,
    ExtractPressure_Features_repetitions_based,
    ExtractPressure_Features_window_based
)
from feature_extraction.get_paths import get_test_file_paths, get_one_foler_path
from feature_extraction.create_feature_windows import drop_last_for_label
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction
import time
from collections import Counter
from pathlib import Path

# Description of module:
# Implements feature extraction for a maximum number of 6 sensors, or a subset of fewer. Features are: frequency domain,
# time domain (<- Maria's work) and expanded fsr features (<- Alba's work).
# Previoulsy implemented scenarios are not considered at this time, but will be included again at a later convenience.
# Right now this file is only implemented for preliminary analyses using the basline (Maria) feature space, or optionally adding
# additional insole-derived features.
# Previous functionality of using norm IMU axes and HDR accelerometer has also been removed to reduce clutter. They may be added back in
# at a later convenince.


def run_feature_extraction(
    output_dir,
    test_id,
    right_arm_df=None,
    left_arm_df=None,
    lower_back_df=None,
    upper_back_df=None,
    left_fsr_df=None,
    right_fsr_df=None,
    expanded_fsr=False,
    IMU_sampling_rates=800,
    mode = 'Window',
    window_sec = 3.5,
    resample_IMU = True,
    target_resampling_rate = 100,
    use_rep_id = True

):
    """
    Allows modes 'Window' which uses fixed seconds based windows WITHIN EACH rep_id or label, or 'Repetition'. 'Repetition' corresponds to the
    methodology from specialization project.
    """
    print("Starting...")
    # define constants
    base_IMU_rate = 800
    base_target_samples = 2800
    sampling_rate_ratio = IMU_sampling_rates / base_IMU_rate
    target_IMU_samples = int(base_target_samples * sampling_rate_ratio)
    print("Target IMU samples is", target_IMU_samples)

    if (
        right_arm_df is None
        or lower_back_df is None
        or left_fsr_df is None
        or right_fsr_df is None
    ):
        print(
            "WARNING Data file [right arm, lower back, left sole or right sole] is not provided."
        )
        # answer = input("Was this intentional? (y/n): ").strip().lower()
        # if answer == "n":
        #     print("Stopping...")
        #     return
        # else:
        #     print("Continuing...")
    # SEPARATE BY MODES
    if mode == 'Repetition':
        if right_arm_df is not None:
            feat_muse_rarm, window_labels_rarm, window_static_labels_rarm = ExtractIMU_features_repetitions_based(
                right_arm_df,
                "R_Arm",
                fs=IMU_sampling_rates,
                target_num_samples=target_IMU_samples,
            )
        else:
            feat_muse_rarm, window_labels_rarm, window_static_labels_rarm = None, None, None

            # Left Arm
        if left_arm_df is not None:
            feat_muse_larm, window_labels_larm, window_static_labels_larm = ExtractIMU_features_repetitions_based(
                left_arm_df,
                "L_Arm",
                fs=IMU_sampling_rates,
                target_num_samples=target_IMU_samples,
            )
        else:
            feat_muse_larm, window_labels_larm, window_static_labels_larm = None, None, None

        # Lower Back
        if lower_back_df is not None:
            feat_muse_lback, window_labels_lback, window_static_labels_lback = ExtractIMU_features_repetitions_based(
                lower_back_df,
                "Lower_Back",
                fs=IMU_sampling_rates,
                target_num_samples=target_IMU_samples,
            )
        else:
            feat_muse_lback, window_labels_lback, window_static_labels_lback = None, None, None

        # Upper Back
        if upper_back_df is not None:
            feat_muse_uback, window_labels_uback, window_static_labels_uback = ExtractIMU_features_repetitions_based(
                upper_back_df,
                "Upper_Back",
                fs=IMU_sampling_rates,
                target_num_samples=target_IMU_samples,
            )
        else:
            feat_muse_uback, window_labels_uback, window_static_labels_uback = None, None, None

        # Left FSR
        if left_fsr_df is not None:
            if expanded_fsr == False:
                feat_fsr_left, window_labels_fsr_left = (
                    ExtractPressure_Features_repetitions_based(
                        left_fsr_df, "Left", mean_fsr=True, fs=100, feature_space="baseline"
                    )
                )
            elif expanded_fsr == True:
                feat_fsr_left, window_labels_fsr_left = (
                    ExtractPressure_Features_repetitions_based(
                        left_fsr_df,
                        "Left",
                        mean_fsr=True,
                        fs=100,
                        feature_space="expanded+baseline",
                    )
                )
        else:
            feat_fsr_left, window_labels_fsr_left = None, None

        # Right FSR
        if right_fsr_df is not None:
            if expanded_fsr == False:
                feat_fsr_right, window_labels_fsr_right = (
                    ExtractPressure_Features_repetitions_based(
                        right_fsr_df,
                        "Right",
                        mean_fsr=True,
                        fs=100,
                        feature_space="baseline",
                    )
                )
            elif expanded_fsr == True:
                feat_fsr_right, window_labels_fsr_right = (
                    ExtractPressure_Features_repetitions_based(
                        right_fsr_df,
                        "Right",
                        mean_fsr=True,
                        fs=100,
                        feature_space="expanded+baseline",
                    )
                )
        else:
            feat_fsr_right, window_labels_fsr_right = None, None


    elif mode == 'Window':
        if right_arm_df is not None:
            feat_muse_rarm, window_labels_rarm, window_static_labels_rarm = ExtractIMU_features_window_based(
                imu_data=right_arm_df,
                sensor_name="R_Arm",
                fs=IMU_sampling_rates,
                window_sec=window_sec,
                use_rep_id=use_rep_id,
                resample_signal=resample_IMU,
                target_fs=target_resampling_rate,
            )
        else:
            feat_muse_rarm, window_labels_rarm, window_static_labels_rarm = None, None, None

            # Left Arm
        if left_arm_df is not None:
            feat_muse_larm, window_labels_larm, window_static_labels_larm = ExtractIMU_features_window_based(
                imu_data=left_arm_df,
                sensor_name="L_Arm",
                fs=IMU_sampling_rates,
                window_sec=window_sec,
                use_rep_id=use_rep_id,
                resample_signal=resample_IMU,
                target_fs=target_resampling_rate,
            )
        else:
            feat_muse_larm, window_labels_larm, window_static_labels_larm = None, None, None

        # Lower Back
        if lower_back_df is not None:
            feat_muse_lback, window_labels_lback, window_static_labels_lback = ExtractIMU_features_window_based(
                imu_data=lower_back_df,
                sensor_name="Lower_Back",
                fs=IMU_sampling_rates,
                window_sec=window_sec,
                use_rep_id=use_rep_id,
                resample_signal=resample_IMU,
                target_fs=target_resampling_rate,
            )
        else:
            feat_muse_lback, window_labels_lback, window_static_labels_lback = None, None, None

        # Upper Back
        if upper_back_df is not None:
            feat_muse_uback, window_labels_uback, window_static_labels_uback = ExtractIMU_features_window_based(
                imu_data=upper_back_df,
                sensor_name="Upper_Back",
                fs=IMU_sampling_rates,
                window_sec=window_sec,
                use_rep_id=use_rep_id,
                resample_signal=resample_IMU,
                target_fs=target_resampling_rate,
            )
        else:
            feat_muse_uback, window_labels_uback,  window_static_labels_uback = None, None, None

        # Left FSR
        if left_fsr_df is not None:
            if expanded_fsr == False:
                feat_fsr_left, window_labels_fsr_left = (
                    ExtractPressure_Features_window_based(
                        fsr_data = left_fsr_df,
                        sensor_name = "Left",
                        mean_fsr=True,
                        fs=100, 
                        feature_space="baseline",
                        window_sec=window_sec,
                        use_rep_id=use_rep_id
                    )
                )
            elif expanded_fsr == True:
                feat_fsr_left, window_labels_fsr_left = (
                    ExtractPressure_Features_window_based(
                        fsr_data = left_fsr_df,
                        sensor_name = "Left",
                        mean_fsr=True,
                        fs=100, 
                        feature_space="expanded+baseline",
                        window_sec=window_sec,
                        use_rep_id=use_rep_id
                    )
                )
        else:
            feat_fsr_left, window_labels_fsr_left = None, None

        # Right FSR
        if right_fsr_df is not None:
            if expanded_fsr == False:
                feat_fsr_right, window_labels_fsr_right = (
                    ExtractPressure_Features_window_based(
                        fsr_data = right_fsr_df,
                        sensor_name = "Right",
                        mean_fsr=True,
                        fs=100, 
                        feature_space="baseline",
                        window_sec=window_sec,
                        use_rep_id=use_rep_id
                    )
                )
            elif expanded_fsr == True:
                feat_fsr_right, window_labels_fsr_right = (
                    ExtractPressure_Features_window_based(
                        fsr_data = right_fsr_df,
                        sensor_name = "Right",
                        mean_fsr=True,
                        fs=100, 
                        feature_space="expanded+baseline",
                        window_sec=window_sec,
                        use_rep_id=use_rep_id
                    )
                )
        else:
            feat_fsr_right, window_labels_fsr_right = None, None

    else:
        raise ValueError('Mode for feature extraction is not recognized, must be either Window or Repetition.')

    features_dict = {
        "right_arm": feat_muse_rarm,
        "left_arm": feat_muse_larm,
        "lower_back": feat_muse_lback,
        "upper_back": feat_muse_uback,
        "left_fsr": feat_fsr_left,
        "right_fsr": feat_fsr_right,
    }

    label_dict = {
        "right_arm": window_labels_rarm,
        "left_arm": window_labels_larm,
        "lower_back": window_labels_lback,
        "upper_back": window_labels_uback,
        "left_fsr": window_labels_fsr_left,
        "right_fsr": window_labels_fsr_right,
    }

    static_label_dict = {
        "right_arm": window_static_labels_rarm,
        "left_arm": window_static_labels_larm,
        "lower_back": window_static_labels_lback,
        "upper_back": window_static_labels_uback,
        "left_fsr": None,
        "right_fsr": None
    }
    # Filter out missing sensors
    available_sensors = [s for s in features_dict if features_dict[s] is not None]
    

    # Align labels across all available sensors
    all_labels = sorted(set(l for s in available_sensors for l in label_dict[s]))

    for label in all_labels:
        # find min count of this label among all sensors
        min_count = min(Counter(label_dict[s]).get(label, 0) for s in available_sensors)
        for sensor in available_sensors:
            current_count = Counter(label_dict[sensor]).get(label, 0)
            while current_count > min_count:
                # print("CURRENT COUNT HIGH!!!")
                # print("sensor is ", sensor)
                # print("label is", label)

                # time.sleep(5)
                features_dict[sensor], label_dict[sensor] = drop_last_for_label(
                    features_dict[sensor], label_dict[sensor], static_label_dict[sensor],label
                )
                current_count -= 1
    # find static vs non static consensus
    for s in available_sensors:
        if s == "left_fsr" or s == "right_fsr": # ignore soles for this
            continue
        print(s, len(static_label_dict[s]))
        print(s, len(label_dict[s]))

        n_windows = len(static_label_dict[available_sensors[0]])

        consensus_static_labels = []
        for i in range(n_windows):
            votes_transient = 0 # reset per window

            for s in available_sensors:
                if s == "left_fsr" or s == "right_fsr": # ignore soles for this
                    continue
                static_label = static_label_dict[s][i]

                if "transient" in static_label:
                    print(f"sensor {s} detected transient")
                    votes_transient += 1

            # low bar for transient: if any sensor says transient
            if votes_transient >= 1:
                consensus_static_labels.append("transient")
            else:
                print(f"static rep")
                #time.sleep(1)
                consensus_static_labels.append("static")
        

    # Check all lengths are equal
    lengths = [len(features_dict[s]) for s in available_sensors]
    if len(set(lengths)) > 1:
        print("STOPPING ... different number of windows across sensors!")
        for sensor in available_sensors:
            print(f"  {sensor:12}: {len(features_dict[sensor])}")
        return None
   
    # Combine features
    all_features = pd.concat([features_dict[s] for s in available_sensors], axis=1)
    all_features["label"] = label_dict[available_sensors[0]]  # first sensor's labels (these should be consistent!)
    print("len av sensors:", len(available_sensors[0]))
    print("len static", len(consensus_static_labels))
    print("applying static labels")
    all_features["static_label"] = consensus_static_labels
    print("succeeded")
    sensor_combo_scenario = "_".join(available_sensors)

    output_filename = (
        f"Features_{mode}_{test_id}_expanded{expanded_fsr}_SEG{mode}{window_sec}_{sensor_combo_scenario}.csv"
    )

    all_features.to_csv(Path(output_dir) / output_filename)

    return all_features


def run_feature_extraction_for_multiple_tests(
    scenario: list[str],
    expanded_fsr: bool = False,
    test_ids: str | list = "All",
    stop_if_one_fails: bool = True,
    IMU_sampling_rate: int = 800,

    # ⭐ NEW PARAMETERS
    mode: str = "Window",          # "Repetition" or "Window"
    resample_signal: bool = True,
    target_fs: int = 100,
    window_sec: float = 3.5,
    use_rep_id: bool = True,
) -> bool:

    file_dict = get_test_file_paths()
    start = time.time()

    for test_id, paths in file_dict.items():

        if test_ids != "All" and test_id not in test_ids:
            continue

        print(f"\n--- Running feature extraction for {test_id} ---")

        try:
            right_arm_path = (
                paths.get("right_arm") or paths.get("arm")
                if "right_arm" in scenario else None
            )

            left_arm_path = paths.get("left_arm") if "left_arm" in scenario else None

            lower_back_path = (
                paths.get("lower_back") or paths.get("back")
                if "lower_back" in scenario else None
            )

            upper_back_path = (
                paths.get("upper_back") if "upper_back" in scenario else None
            )

            left_fsr_path = (
                paths.get("left") or paths.get("left_fsr")
                if "left_fsr" in scenario else None
            )

            right_fsr_path = (
                paths.get("right") or paths.get("right_fsr")
                if "right_fsr" in scenario else None
            )

            df_rarm = pd.read_csv(right_arm_path) if right_arm_path else None
            df_larm = pd.read_csv(left_arm_path) if left_arm_path else None
            df_lback = pd.read_csv(lower_back_path) if lower_back_path else None
            df_uback = pd.read_csv(upper_back_path) if upper_back_path else None
            df_lfsr = pd.read_csv(left_fsr_path) if left_fsr_path else None
            df_rfsr = pd.read_csv(right_fsr_path) if right_fsr_path else None

           # print(df_rarm.head())

            feat_dir = get_one_foler_path(test_id)
            print(f"Directory to save features: {feat_dir}")

            continuous_labels = [
            "walking",
            "standing",
            "sitting",
            "neutral_load",
            "neutral_load_left",
            "neutral_load_right",
            ]

            if test_id == 'prelim_8':
                continuous_labels = [
                "walking",
                "standing",
                "sitting",
                "neutral_load",
                "neutral_load_left",
                "neutral_load_right",
                "drag"
                ] # <- drag was recorded in one segment for prelim_8, therefore treat it like a continuous label for this test

            dfs = [df_rarm, df_larm, df_lback, df_uback, df_lfsr, df_rfsr]

            if mode == "Repetition":
                filtered_dfs = []

                for df in dfs:
                    if df is not None and "rep_id" in df.columns:
                        df["label"] = df["label"].astype("string")

                        keep_mask = (
                            df["rep_id"].notna()
                            | df["label"].isin(continuous_labels)
                        )

                        df = df.loc[keep_mask].copy()

                    filtered_dfs.append(df)

                df_rarm, df_larm, df_lback, df_uback, df_lfsr, df_rfsr = filtered_dfs
            all_features = run_feature_extraction(
                output_dir=feat_dir,
                test_id=test_id,
                right_arm_df=df_rarm,
                left_arm_df=df_larm,
                lower_back_df=df_lback,
                upper_back_df=df_uback,
                left_fsr_df=df_lfsr,
                right_fsr_df=df_rfsr,
                expanded_fsr=expanded_fsr,
                IMU_sampling_rates=IMU_sampling_rate,

                # ⭐ NEW PIPELINE OPTIONS
                mode=mode,
                resample_IMU=resample_signal,
                target_resampling_rate=target_fs,
                window_sec=window_sec,
                use_rep_id=use_rep_id,
            )

            if all_features is None:
                print(f"Feature extraction failed for {test_id}")
                if stop_if_one_fails:
                    return False

        except Exception as e:
            print(f"Failed for {test_id}: {e}")
            if stop_if_one_fails:
                return False
    end = time.time()
    print(f"\n🕒 Done! Total time used: {end - start:.2f} seconds")

    return True

if __name__ == "__main__":
    # Define which test ids to run feature extraction for
    run = [
        # 'prelim_1',
        # 'prelim_2',
        # 'prelim_3',
        # 'prelim_4',
        # 'prelim_5',
        # 'prelim_6',
       # 'prelim_7',
        'prelim_8',
        # 'aksoprotocol_1',
        # 'aksoprotocol_2',
        # 'aksoprotocol_3',
        # 'aksoprotocol_4',
        #'aksoprotocol_5',
        # 'aksowork_1',
        # 'aksowork_2',
        # 'aksowork_3',
        # 'aksowork_4',
        # 'aksowork_5'
        #'aksowork_8',
        # 'test_1',
        # 'test_2',
        # 'test_3',
        # 'test_4',
        # 'test_5',
        # 'test_6',
        #'test_7',
        # 'test_8',
        #'test_9',
        # 'test_10',
        # 'test_11',
        # 'test_12',
        # 'test_13',
        #'test_14',
        # 'test_15',
        # 'test_16',
        # 'test_17',
        # 'test_18',
        # 'test_19',
        # 'test_20'
    ]

    #### Define scenarios to be used in feature extraction ####
    # scenario with all available sensors
    scenario_SC1 = [
        "right_arm",
        "left_arm",
        "lower_back",
        "upper_back",
        "left_fsr",
        "right_fsr",
    ]
    # scenario with only 4 sensors (the old setup)
    scenario_SC2 = ["right_arm", "lower_back", "left_fsr", "right_fsr"]
    
    # bilateral arms and insoles
    scenario_SC3 = [
        "right_arm",
        "left_arm",
        "left_fsr",
        "right_fsr"
    ]
    
    # right (dominant?) arm and insoles
    scenario_SC4 = [
        "right_arm",
        "left_fsr",
        "right_fsr"
    ]
    
    # left arm and insoles
    scenario_SC5 = [
        "left_arm",
        "left_fsr",
        "right_fsr"
    ]
    
    # only IMU, new setup
    scenario_SC6 = [
        "right_arm",
        "left_arm",
        "lower_back",
        "upper_back",
    ]
    
    # only IMU, legacy setup
    scenario_SC7 = [
        "right_arm",
        "lower_back",
    ]
    
    # update this according to which files to run. some files (legacy) do not allow running certain scenarios...
    scenarios = [
        scenario_SC1,
        #scenario_SC2,
        # scenario_SC3,
        # scenario_SC4,
        # scenario_SC5,
        #scenario_SC6,
        # scenario_SC7
    ]

    # Define settings
    mode = "Repetition"              # "window" or "repetition"
    expanded_fsr = True
    imu_sampling_rate = 100 # original sampling rate of the files in this run
    resample_signal = False  # set True if these files need IMU resampling (OLD-RESAMPLING WAS DONE AT THE PREPROCESSING LEVEL IN THE FINAL VERSION!)
    target_fs = 100              # target IMU fs after resampling
    window_sec = 3.5
    use_rep_id = True # if False, use consecutive label runs instead

    # Run with selected settings!
    # NB IMU sampling rates must be consistent across all participants. Run different sampling rates in separate runs!
    
    # RUN A SINGLE SCENARIO
    # run_feature_extraction_for_multiple_tests(
    #     scenario=scenario_SC1,
    #     expanded_fsr=expanded_fsr,
    #     test_ids=run,
    #     stop_if_one_fails=True,
    #     IMU_sampling_rate=imu_sampling_rate,
    #     mode=mode,
    #     resample_signal=resample_signal,
    #     target_fs=target_fs,
    #     window_sec=window_sec,
    #     use_rep_id=use_rep_id,)
    
    # RUN ALL SCENARIOS
    # runs all scenarios with fixed settings. remember to set correct segmentation mode and window length for 
    # each run
    for s in scenarios:
        run_feature_extraction_for_multiple_tests(
                                            scenario=s,
                                            expanded_fsr=expanded_fsr,
                                            test_ids=run,
                                            stop_if_one_fails=True,
                                            IMU_sampling_rate=imu_sampling_rate,
                                            mode=mode,
                                            resample_signal=resample_signal,
                                            target_fs=target_fs,
                                            window_sec=window_sec,
                                            use_rep_id=use_rep_id,
                                            )
                                        
    