import pandas as pd

from feature_extraction.ExtractIMU_Features import ExtractIMU_Features, ExtractIMU_features_repetitions_based
from feature_extraction.ExtractPressure_Features import ExtractPressure_Features, ExtractPressure_Features_repetitions_based
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
):
    print("Starting...")
    if right_arm_df is None or lower_back_df is None or left_fsr_df is None or right_fsr_df is None:
        print("!!WARNING!! Data file [right arm, lower back, left sole or right sole] is not provided.")
        answer = input("Was this intentional? (y/n): ").strip().lower()
        if answer == "n":
            print("Stopping...")
            return
        else:
            print("Continuing...")
    
    if right_arm_df is not None:
        feat_muse_rarm, window_labels_rarm = ExtractIMU_features_repetitions_based(right_arm_df, "R_Arm", fs=800)
    else:
        feat_muse_rarm, window_labels_rarm = None, None
    
        # Left Arm
    if left_arm_df is not None:
        feat_muse_larm, window_labels_larm = ExtractIMU_features_repetitions_based(left_arm_df, "L_Arm", fs=800)
    else:
        feat_muse_larm, window_labels_larm = None, None

    # Lower Back
    if lower_back_df is not None:
        feat_muse_lback, window_labels_lback = ExtractIMU_features_repetitions_based(lower_back_df, "Lower_Back", fs=800)
    else:
        feat_muse_lback, window_labels_lback = None, None

    # Upper Back
    if upper_back_df is not None:
        feat_muse_uback, window_labels_uback = ExtractIMU_features_repetitions_based(upper_back_df, "Upper_Back", fs=800)
    else:
        feat_muse_uback, window_labels_uback = None, None

    # Left FSR
    if left_fsr_df is not None:
        if expanded_fsr == False:
            feat_fsr_left, window_labels_fsr_left = ExtractPressure_Features_repetitions_based(left_fsr_df, "Left", mean_fsr=True, fs=100, feature_space='baseline')
        elif expanded_fsr == True:
            feat_fsr_left, window_labels_fsr_left = ExtractPressure_Features_repetitions_based(left_fsr_df, "Left", mean_fsr=True, fs=100, feature_space='expanded+baseline')
    else:
        feat_fsr_left, window_labels_fsr_left = None, None

    # Right FSR
    if right_fsr_df is not None:
        if expanded_fsr == False:
            feat_fsr_right, window_labels_fsr_right = ExtractPressure_Features_repetitions_based(right_fsr_df, "Right", mean_fsr=True, fs=100, feature_space='baseline')
        elif expanded_fsr == True:
            feat_fsr_right, window_labels_fsr_right = ExtractPressure_Features_repetitions_based(right_fsr_df, "Right", mean_fsr=True, fs=100, feature_space='expanded+baseline')
    else:
        feat_fsr_right, window_labels_fsr_right = None, None
        
    features_dict = {
    "right_arm": feat_muse_rarm,
    "left_arm": feat_muse_larm,
    "lower_back": feat_muse_lback,
    "upper_back": feat_muse_uback,
    "left_fsr": feat_fsr_left,
    "right_fsr": feat_fsr_right
    }

    label_dict = {
        "right_arm": window_labels_rarm,
        "left_arm": window_labels_larm,
        "lower_back": window_labels_lback,
        "upper_back": window_labels_uback,
        "left_fsr": window_labels_fsr_left,
        "right_fsr": window_labels_fsr_right
    }
    
    # Filter out missing sensors
    available_sensors = [s for s in features_dict if features_dict[s] is not None]

    # Align labels across all available sensors
    all_labels = sorted(set(l for s in available_sensors for l in label_dict[s]))

    for label in all_labels:
        # find max count of this label among all sensors
        max_count = max(Counter(label_dict[s]).get(label, 0) for s in available_sensors)
        for sensor in available_sensors:
            current_count = Counter(label_dict[sensor]).get(label, 0)
            while current_count > max_count:
                # Drop last row for this label (you already have drop_last_for_label)
                features_dict[sensor], label_dict[sensor] = drop_last_for_label(
                    features_dict[sensor], label_dict[sensor], label
                )
                current_count -= 1

        # Check all lengths are equal
        lengths = [len(features_dict[s]) for s in available_sensors]
        if len(set(lengths)) > 1:
            print("STOPPING ... different number of windows across sensors!")
            for sensor in available_sensors:
                print(f"  {sensor:12}: {len(features_dict[sensor])}")
            return None

        # Combine features
        all_features = pd.concat([features_dict[s] for s in available_sensors], axis=1)
        all_features['label'] = label_dict[available_sensors[0]]  # first sensor's labels

        sensor_combo_scenario = "_".join(available_sensors)
            
        output_filename = f"Features_{test_id}_expanded{expanded_fsr}_{sensor_combo_scenario}.csv"
        
        all_features.to_csv(Path(output_dir) / output_filename)

        return all_features, available_sensors


def run_feature_extraction_for_multiple_tests(expanded_fsr=False, test_ids="All"):
    """
    Default behavior: test_ids is set to "All" - runs feature extraction for all test_ids found. Alternatively set test_ids to a list
    of str corresponding to the test_ids you want to run feature extraction for.
    """
    file_dict = get_test_file_paths()
    start = time.time()
    
    for test_id, paths in file_dict.items():
        if test_ids != "All" and test_id not in test_ids:
            continue
        print(f"\n--- Running feature extraction for {test_id} ---")
        try:
            # Map input files to standardized sensor names
            right_arm_path  = paths.get("right_arm") or paths.get("arm")
            left_arm_path   = paths.get("left_arm")
            lower_back_path = paths.get("lower_back") or paths.get("back")
            upper_back_path = paths.get("upper_back")
            left_fsr_path   = paths.get("left") or paths.get("left_fsr")
            right_fsr_path  = paths.get("right") or paths.get("right_fsr")
            
            # Load CSVs if files exist
            df_rarm = pd.read_csv(right_arm_path) if right_arm_path else None
            df_larm = pd.read_csv(left_arm_path) if left_arm_path else None
            df_lback = pd.read_csv(lower_back_path) if lower_back_path else None
            df_uback = pd.read_csv(upper_back_path) if upper_back_path else None
            df_lfsr = pd.read_csv(left_fsr_path) if left_fsr_path else None
            df_rfsr = pd.read_csv(right_fsr_path) if right_fsr_path else None
            
            feat_dir = get_one_foler_path(test_id)
            print(f"Directory to save features: {feat_dir}")
            
            # Drop rows where rep_id == None for files that exist
            for df in [df_rarm, df_larm, df_lback, df_uback, df_lfsr, df_rfsr]:
                if df is not None and "rep_id" in df.columns:
                    df.dropna(subset=["rep_id"], inplace=True)
            
            
            # Run feature extraction with flexible inputs
            # Features are saved to a .csv as a final step in run_feature_extraction.
            all_features = run_feature_extraction(
                output_dir=feat_dir,
                test_id = test_id,
                right_arm_df=df_rarm,
                left_arm_df=df_larm,
                lower_back_df=df_lback,
                upper_back_df=df_uback,
                left_fsr_df=df_lfsr,
                right_fsr_df=df_rfsr,
                expanded_fsr=expanded_fsr,
            )

            
            if all_features is None:
                print(f"Feature extraction failed for {test_id}")
                return None

        except Exception as e:
            print(f"Failed for {test_id}: {e}")
            return None  # stop everything if one test fails

    end = time.time()
    elapsed = end - start
    print(f"\n🕒 Done! Total time used: {elapsed:.2f} seconds")
    
    return True


if __name__ == '__main__':
    # Optionally define which test ids to run feature extraction for
    run = ['prelim_1']

    # Run with selected settings!
    run_feature_extraction_for_multiple_tests(expanded_fsr=True, test_ids=run)

    #TODO: configure multiple tests function to be able to account for different feature space configurations.
    

