# Module assumes that  labeling / repetitions segmentation has already been done.
# Label start stop times and (if applicable) repetetion start stop times should be available in their own respective .csv files. 
# Automatic preprocessing will not run if these are not present

# Labeling and segmentation workflow can be found iin the labeling jupyter notebooks. These are used for intitla preprocessing. However, at 
# the need for redoing the preprocessing without assigning new label or rep id start stop times, this script is more efficient.

# WORKFLOW OVERVIEW
# 1. Start with time aligning sensor files.
# 2. Apply labels and rep_ids
# 3. Resample 800Hz IMU files to 100Hz
# 5. Filter 100Hz data streams
# 6. Drop idle or unlabeled segments
# 7. Rotate back IMUs where necessary
# 8. Clean dataset columns and FSR mapping!
# 9. MANUAL store the resulting files in get_paths


from pre_processing.labeling import detect_spikes, init_label, plot_signal_peaks, extract_activity_windows, apply_corrected_labels, extract_pressure_data, remove_idle, sort_push_pull
from pre_processing.fiks_time_muse import make_df, check_samples, fix_time_muse_hz
from pre_processing.clean_mitch_timestamps import remove_end_duplicates, add_ReconstructedTime
from pre_processing.filter_sensor_data import median_filter_medfilt
from pre_processing.segment_repetition_vol2 import plot_activity_accelerations_peaks_and_magnitude, get_start_stop_times_from_peaks, assign_rep_ids
from plotting.plot_sensor_signals import plot_all_sensors, plot_fsr, plot_single_imu
from utils import get_imu_cols, get_fsr_cols
from feature_extraction.resample_old_imu_files import resample_imu_dataframe
from pre_processing.rotate_back_imus import rotate_axl_mag_gyr_180_deg_abt_z_axis
from pre_processing.clean_dataset_columns import clean_and_rename_columns

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def drop_label(df, label_to_drop):
    return df[df["label"] != label_to_drop].reset_index(drop=True)


def apply_time_labels(sensor_df, labels_df, time_col="ReconstructedTime"):
    """
    Apply interval labels to a dataframe based on time column.
    Returns a copy with an updated label column
    """

    df = sensor_df.copy()

    # Normalize column names
    rename_map = {
        "Start Time (s)": "start_time",
        "End Time (s)": "end_time",
        "Label": "label",
    }
    labels_df = labels_df.rename(columns=rename_map)

    for _, row in labels_df.iterrows():
        mask = (df[time_col] >= row["start_time"]) & (df[time_col] <= row["end_time"])
        df.loc[mask, "label"] = row["label"]

    return df



if __name__ == '__main__':
    # configure here at the top. Settings must work for all partcipants. i.e. they must have consistent sampling rates and all must be either protocol
    # or non rotocol, rep_id or no rep_id. no mixing and matching allowed. 
    base_dir = Path(r"C:\Users\johalot\msc_data")
    participants = [
        'akso_5'
    ]
    protocol = False # only relevant for the akso files. Set to true to find the prtocol files, false for work files.
    rep_id = False # whether or not to assign a rep_id column. if set to true the code assumes you have rep id start stop .csv files in the participant directory.
    IMU_original_fs = 100
    IMU_target_fs = 100
    resample_IMU = False # whether or not to resample IMU streams. Apply if original fs is 800! 
    rotate_back = False


    # --------------------------------------- Begin run --------------------------------------------------------------

    # run for each configured participant
    for participant in participants:
        print(f'PROCESSING PARTICIPANT {participant}')
        data_dir = base_dir / participant

        # First get files. This should run automatically, but may fail if files are not named consistently. 
        # This will be the case for the 'test' dataset for instance. 

        
        # PRELIM 1-6
        try:
            raw_rarm = data_dir / "Muse_E2511_RED-ARM_RIGHT.txt"
            raw_larm = data_dir / "Muse_E2511_GREY-ARM_LEFT.txt"
            raw_lb = data_dir / "muse_v3_3-LB.txt"
            raw_ub = data_dir / "muse_v3-UB.txt"
            raw_lsole = data_dir / "mitch_B0308-LEFT.txt"
            raw_rsole = data_dir / "mitch_B0510-RIGHT.txt"

            df_rarm = pd.read_csv(raw_rarm, delimiter="\t", skiprows=8, decimal=",")
            df_larm = pd.read_csv(raw_larm, delimiter="\t", skiprows=8, decimal=",")
            df_lb = pd.read_csv(raw_lb, delimiter="\t", skiprows=8, decimal=",")
            df_ub = pd.read_csv(raw_ub, delimiter="\t", skiprows=8, decimal=",")
            df_lsole = pd.read_csv(raw_lsole, delimiter="\t", skiprows=8, decimal=",")
            df_rsole = pd.read_csv(raw_rsole, delimiter="\t", skiprows=8, decimal=",")
        except Exception as e:
            print(f"Error trying to read raw files method 1: {e}. Moving on to method 2.")
            # PRELIM 7+
            try:
                raw_rarm = data_dir / "Muse_E2511_GREY-ARM_RIGHT.txt"
                raw_larm = data_dir / "Muse_E2511_RED-ARM_LEFT.txt"
                raw_lb = data_dir / "muse_v3_3-LB.txt"
                raw_ub = data_dir / "muse_v3-UB.txt"
                raw_lsole = data_dir / "mitch_B0308-LEFT.txt"
                raw_rsole = data_dir / "mitch_B0510-RIGHT.txt"

                df_rarm = pd.read_csv(raw_rarm, delimiter="\t", skiprows=8, decimal=",")
                df_larm = pd.read_csv(raw_larm, delimiter="\t", skiprows=8, decimal=",")
                df_lb = pd.read_csv(raw_lb, delimiter="\t", skiprows=8, decimal=",")
                df_ub = pd.read_csv(raw_ub, delimiter="\t", skiprows=8, decimal=",")
                df_lsole = pd.read_csv(raw_lsole, delimiter="\t", skiprows=8, decimal=",")
                df_rsole = pd.read_csv(raw_rsole, delimiter="\t", skiprows=8, decimal=",")
            except Exception as e:
                print(f"Error trying to read raw files method 2: {e}. Raw files will not be read.")

        
        # attempt finding files that already have reconstructed time
        time_rarm = None
        time_larm = None
        time_lb = None
        time_ub = None
        time_lsole = None
        time_rsole = None

        # AKSO PROTOCOL DATA
        if protocol == True:
            try:
                print('Trying to get akso protocol files...')
                time_rarm = data_dir / "right_arm_protocol_reconst_time.csv"
                time_larm = data_dir / "left_arm_protocol_reconst_time.csv"
                time_lb = data_dir / "lower_back_protocol_reconst_time.csv"
                time_ub = data_dir / "upper_back_protocol_reconst_time.csv"
                time_lsole = data_dir / "left_sole_protocol_reconst_time.csv"
                time_rsole = data_dir / "right_sole_protocol_reconst_time.csv"

                df_time_rarm = None
                df_time_larm = None
                df_time_lb = None
                df_time_ub = None
                df_time_left = None
                df_time_right = None

                df_time_rarm = pd.read_csv(time_rarm)
                df_time_larm = pd.read_csv(time_larm)
                df_time_lb = pd.read_csv(time_lb)
                df_time_ub = pd.read_csv(time_ub)
                df_time_lsole = pd.read_csv(time_lsole)
                df_time_rsole = pd.read_csv(time_rsole)
            except Exception as e:
                print(f"Could not retreive akso protocol files... Error {e}")

        # AKSO REAL WORK DATA
        else:
            try:
                time_rarm = data_dir / 'right_arm_hall_reconst_time.csv'
                time_larm = data_dir / 'left_arm_hall_reconst_time.csv'
                time_lb = data_dir / 'lower_back_hall_reconst_time.csv'
                time_ub = data_dir / 'upper_back_hall_reconst_time.csv'
                time_lsole = data_dir / 'left_sole_hall_reconst_time.csv'
                time_rsole = data_dir / 'right_sole_hall_reconst_time.csv'

                df_rarm = pd.read_csv(str(time_rarm))
                df_larm = pd.read_csv(str(time_larm))
                df_lb = pd.read_csv(str(time_lb))
                df_ub = pd.read_csv(str(time_ub))
                df_lsole = pd.read_csv(str(time_lsole))
                df_rsole = pd.read_csv(str(time_rsole))

                df_list = [df_rarm, df_larm, df_lsole, df_rsole, df_lb, df_ub]

                time_rarm = fix_time_muse_hz(df_list, 0, str(time_rarm), IMU_original_fs) 
                time_larm = fix_time_muse_hz(df_list, 1, str(time_larm), IMU_original_fs) 
                time_lb = fix_time_muse_hz(df_list, 4, str(time_lb), IMU_original_fs) 
                time_ub = fix_time_muse_hz(df_list, 5, str(time_ub), IMU_original_fs) 

                df_dupli_left, outpath_dupli_left = remove_end_duplicates(str(time_lsole))
                df_dupli_right, outpath_dupli_right = remove_end_duplicates(str(time_rsole))

                df_time_left, time_lsole = add_ReconstructedTime(outpath_dupli_left, df_dupli_left)
                df_time_right, time_rsole = add_ReconstructedTime(outpath_dupli_right, df_dupli_right)
                
                df_time_rarm = pd.read_csv(time_rarm)
                df_time_larm = pd.read_csv(time_larm)
                df_time_lb = pd.read_csv(time_lb)
                df_time_ub = pd.read_csv(time_ub)
                df_time_lsole = pd.read_csv(time_lsole)
                df_time_rsole = pd.read_csv(time_rsole)


            except Exception as e:
                print(f"Could not retreive akso true work files...")

        if df_time_rarm is None or df_time_larm is None or df_time_lb is None or df_time_ub is None or df_time_lsole is None or df_time_rsole is None:
            # do time alignment otherwise skip directly to applying labels and rep_ids
            # use raw files for this
            print('Beginning time recostruction....')
            df_list = [df_rarm, df_larm, df_lsole, df_rsole, df_lb, df_ub]

            time_rarm = fix_time_muse_hz(df_list, 0, str(raw_rarm), IMU_original_fs) 
            time_larm = fix_time_muse_hz(df_list, 1, str(raw_larm), IMU_original_fs) 
            time_lb = fix_time_muse_hz(df_list, 4, str(raw_lb), IMU_original_fs) 
            time_ub = fix_time_muse_hz(df_list, 5, str(raw_ub), IMU_original_fs) 

            df_dupli_left, outpath_dupli_left = remove_end_duplicates(str(raw_lsole))
            df_dupli_right, outpath_dupli_right = remove_end_duplicates(str(raw_rsole))

            df_time_left, time_lsole = add_ReconstructedTime(outpath_dupli_left, df_dupli_left)
            df_time_right, time_rsole = add_ReconstructedTime(outpath_dupli_right, df_dupli_right)
            
            df_time_rarm = pd.read_csv(time_rarm)
            df_time_larm = pd.read_csv(time_larm)
            df_time_lb = pd.read_csv(time_lb)
            df_time_ub = pd.read_csv(time_ub)
            df_time_lsole = pd.read_csv(time_lsole)
            df_time_rsole = pd.read_csv(time_rsole)

            print("Finsihed time alignment.")
        
        print('Starting apply labels...')
        # Apply labels
        if protocol == True:
            label_file_name = 'start_stop_labels.csv'
        else:
            label_file_name = 'hall_labels_start_stop.csv'

        label_df = pd.read_csv(data_dir / label_file_name)

        # Apply labeling based on timestamps
        df_label_rarm = apply_time_labels(df_time_rarm, labels_df=label_df)
        df_label_larm = apply_time_labels(df_time_larm, labels_df=label_df)
        df_label_lb = apply_time_labels(df_time_lb, labels_df=label_df)
        df_label_ub = apply_time_labels(df_time_ub, labels_df=label_df)
        df_label_left = apply_time_labels(df_time_lsole, labels_df=label_df)
        df_label_right = apply_time_labels(df_time_rsole, labels_df=label_df)

        


        if rep_id == True:
            # Assign rep ids if applicable.

            #create frst the rep_id column
            df_label_larm['rep_id'] = None
            df_label_rarm['rep_id'] = None
            df_label_ub['rep_id'] = None
            df_label_lb['rep_id'] = None
            df_label_left['rep_id'] = None
            df_label_right['rep_id'] = None

            labeled_dfs = {
            "left_arm" : df_label_larm,
            "right_arm" : df_label_rarm,
            "upper_back" : df_label_ub,
            "lower_back" : df_label_lb,
            "left_sole" : df_label_left,
            "right_sole" : df_label_right
            }

            numbered_labels = [
                    "push",
                    "pull",
                    "drag",
                    "stairs_up",
                    "stairs_down"
            ]
            rep_id_dfs = assign_rep_ids(sensor_dfs=labeled_dfs, output_dir=data_dir, numbered_labels=numbered_labels, save_files=False)

            # update the labeled dfs
            df_label_rarm = rep_id_dfs['right_arm']
            df_label_larm = rep_id_dfs['left_arm']
            df_label_lb = rep_id_dfs['lower_back']
            df_label_ub = rep_id_dfs['upper_back']
            df_label_left = rep_id_dfs['left_sole']
            df_label_right = rep_id_dfs['right_sole']
        
        if resample_IMU == True:
            # resample if resample setting is true
            df_label_rarm = resample_imu_dataframe(
            df_label_rarm,
            original_fs=IMU_original_fs,
            target_fs=IMU_target_fs,
            )
            df_label_larm = resample_imu_dataframe(
            df_label_larm,
            original_fs=IMU_original_fs,
            target_fs=IMU_target_fs,
            )
            df_label_lb = resample_imu_dataframe(
            df_label_lb,
            original_fs=IMU_original_fs,
            target_fs=IMU_target_fs,
            )
            df_label_ub = resample_imu_dataframe(
            df_label_ub,
            original_fs=IMU_original_fs,
            target_fs=IMU_target_fs,
            )

        outpath_rarm_labeled = f"{str(time_rarm).rstrip('.csv')}_labeled.csv"
        outpath_larm_labeled = f"{str(time_larm).rstrip('.csv')}_labeled.csv"
        outpath_lb_labeled = f"{str(time_lb).rstrip('.csv')}_labeled.csv"
        outpath_ub_labeled = f"{str(time_ub).rstrip('.csv')}_labeled.csv"
        outpath_lsole_labeled = f"{str(time_lsole).rstrip('.csv')}_labeled.csv"
        outpath_rsole_labeled = f"{str(time_rsole).rstrip('.csv')}_labeled.csv"


        # save labeled files
        df_label_rarm.to_csv(outpath_rarm_labeled)
        df_label_larm.to_csv(outpath_larm_labeled)
        df_label_lb.to_csv(outpath_lb_labeled)
        df_label_ub.to_csv(outpath_ub_labeled)
        df_label_left.to_csv(outpath_lsole_labeled)
        df_label_right.to_csv(outpath_rsole_labeled)
        
        
        # Apply median filter to labeled files
        imu_columns = get_imu_cols()
        fsr_columns = get_fsr_cols()
        df_filtered_rarm, path_rarm_filtered = median_filter_medfilt(outpath_rarm_labeled, imu_columns, kernel_size=9)
        df_filtered_larm, path_larm_filtered = median_filter_medfilt(outpath_larm_labeled, imu_columns, kernel_size=9)
        df_filtered_lb, path_lb_filtered = median_filter_medfilt(outpath_lb_labeled, imu_columns, kernel_size=9)
        df_filtered_ub, path_ub_filtered = median_filter_medfilt(outpath_ub_labeled, imu_columns, kernel_size=9)

        df_filtered_lsole, path_lsole_filtered = median_filter_medfilt(outpath_lsole_labeled, fsr_columns, kernel_size=9)
        df_filtered_rsole, path_rsole_filtered = median_filter_medfilt(outpath_rsole_labeled, fsr_columns, kernel_size=9)

        # drop IMU columns from sole data
        df_filtered_lsole = df_filtered_lsole.drop(columns=imu_columns)
        df_filtered_rsole = df_filtered_rsole.drop(columns=imu_columns)


      

        # drop idle / unlabeled segments
        df_left_no_idle, outpath_lsole_no_idle = remove_idle(correct_label_filepath=path_lsole_filtered, correct_label_df=df_filtered_lsole)
        df_right_no_idle, outpath_rsole_no_idle = remove_idle(correct_label_filepath=path_rsole_filtered, correct_label_df=df_filtered_rsole)
        df_rarm_no_idle, outpath_rarm_no_idle = remove_idle(correct_label_filepath=path_rarm_filtered, correct_label_df=df_filtered_rarm)
        df_larm_no_idle, outpath_larm_no_idle = remove_idle(correct_label_filepath=path_larm_filtered, correct_label_df=df_filtered_larm)
        df_lb_no_idle, outpath_lb_no_idle = remove_idle(correct_label_filepath=path_lb_filtered, correct_label_df=df_filtered_lb)
        df_ub_no_idle, outpath_ub_no_idle = remove_idle(correct_label_filepath=path_ub_filtered, correct_label_df=df_filtered_ub)


        if rotate_back == True:
            df_lb_no_idle = rotate_axl_mag_gyr_180_deg_abt_z_axis(outpath_lb_no_idle)
            df_ub_no_idle = rotate_axl_mag_gyr_180_deg_abt_z_axis(outpath_ub_no_idle)

            outpath_lb_no_idle = f"{outpath_lb_no_idle.rstrip('.csv')}_rotated.csv"
            outpath_ub_no_idle = f"{outpath_ub_no_idle.rstrip('.csv')}_rotated.csv"

            df_lb_no_idle.to_csv(outpath_lb_no_idle)
            df_ub_no_idle.to_csv(outpath_ub_no_idle)
        
        # clean dataset columns
        cleaned_dfs = clean_and_rename_columns(
                left_arm_path=outpath_larm_no_idle,
                right_arm_path=outpath_rarm_no_idle,
                upper_back_path=outpath_ub_no_idle,
                lower_back_path=outpath_lb_no_idle,
                left_fsr_path=outpath_lsole_no_idle,
                right_fsr_path=outpath_rsole_no_idle,
            )
        
        # finished!! 
        # store the cleaned files in data directory
        print(f'Completed preprocessing, storing final files in {data_dir}')

        cleaned_dfs['left_arm'].to_csv(f'{outpath_larm_no_idle.rstrip('.csv')}_cleaned.csv')
        cleaned_dfs['right_arm'].to_csv(f'{outpath_rarm_no_idle.rstrip('.csv')}_cleaned.csv')
        cleaned_dfs['upper_back'].to_csv(f'{outpath_ub_no_idle.rstrip('.csv')}_cleaned.csv')
        cleaned_dfs['lower_back'].to_csv(f'{outpath_lb_no_idle.rstrip('.csv')}_cleaned.csv')
        cleaned_dfs['left_fsr'].to_csv(f'{outpath_lsole_no_idle.rstrip('.csv')}_cleaned.csv')
        cleaned_dfs['right_fsr'].to_csv(f'{outpath_rsole_no_idle.rstrip('.csv')}_cleaned.csv')

        print('Done :))')



            






    
        





        








