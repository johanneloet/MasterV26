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
import time


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
    # configure here at the top. Settings must work for all participants. i.e. they must have consistent sampling rates and all must be either protocol
    # or non protocol, rep_id or no rep_id. no mixing and matching allowed. 
    base_dir = Path(r"C:\Users\Bruker\msc_data")
    participants = [
        'test_14'
        #'prelim_1',
        # 'prelim_2',
        # 'prelim_3',
        # 'prelim_4',
        # 'prelim_8',
        # 'akso_1',
        # 'akso_2',
        # 'akso_3',
        # 'akso_4',
        #'akso_5'
        
    ]
    protocol = True # only relevant for the akso files. Set to true to find the prtocol files, false for work files.
    rep_id = True # whether or not to assign a rep_id column. if set to true the code assumes you have rep id start stop .csv files in the participant directory.
    IMU_original_fs = 800
    IMU_target_fs = 100
    resample_IMU = True # whether or not to resample IMU streams. Apply if original fs is 800! 
    rotate_back = False



    # --------------------------------------- Begin run --------------------------------------------------------------
    # run for each configured participant
    for participant in participants:
        print(f'PROCESSING PARTICIPANT {participant}')
        data_dir = base_dir / participant

        # First get files. This should run automatically, but may fail if files are not named consistently. 
        # This will be the case for the 'test' dataset for instance. 
        # df_time_rarm = None
        # df_time_larm = None
        # df_time_lb = None
        # df_time_ub = None
        # df_time_lsole = None
        # df_time_rsole = None
 
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
                print(f"Error trying to read raw files method 2: {e}. Trying legacy format...")
                time.sleep(1)
                
                try:
                    #legacy setup
                    raw_rarm = data_dir / "Muse_E2511_RED-ARM.txt"
                    raw_lb = data_dir / "Muse_E2511_GREY-BACK.txt"
                    raw_lsole = data_dir / "mitch_B0308-LEFT.txt"
                    raw_rsole = data_dir / "mitch_B0510-RIGHT.txt"
                    
                    df_rarm = pd.read_csv(raw_rarm, delimiter="\t", skiprows=8, decimal=",")
                    df_larm = None
                    df_lb = pd.read_csv(raw_lb, delimiter="\t", skiprows=8, decimal=",")
                    df_ub = None
                    df_lsole = pd.read_csv(raw_lsole, delimiter="\t", skiprows=8, decimal=",")
                    df_rsole = pd.read_csv(raw_rsole, delimiter="\t", skiprows=8, decimal=",")
                    
                
                    
                    
                except Exception as e:
                    print(f"Error trying to read raw files first legacy method: {e}. Moving to final method.")
                    time.sleep(1)
                    try:
                        #legacy setup
                        raw_rarm = data_dir / "Muse_E2511_RED-ARM.txt"
                        raw_lb = data_dir / "Muse_E2511_GREY-BACK.txt"
                        raw_lsole = data_dir / "mitch_B0308-RIGHT.txt"
                        raw_rsole = data_dir / "mitch_B0510-LEFT.txt"
                        
                        df_rarm = pd.read_csv(raw_rarm, delimiter="\t", skiprows=8, decimal=",")
                        df_larm = None
                        df_lb = pd.read_csv(raw_lb, delimiter="\t", skiprows=8, decimal=",")
                        df_ub = None
                        df_lsole = pd.read_csv(raw_lsole, delimiter="\t", skiprows=8, decimal=",")
                        df_rsole = pd.read_csv(raw_rsole, delimiter="\t", skiprows=8, decimal=",")
                    except Exception as e:
                        print(f"Error trying to read raw files all test prelim methods: {e}. Moving to AKSO style.")
                        time.sleep(1)
                    

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
                # time_rarm = data_dir / "right_arm_protocol_reconst_time.csv"
                # time_larm = data_dir / "left_arm_protocol_reconst_time.csv"
                # time_lb = data_dir / "lower_back_protocol_reconst_time.csv"
                # time_ub = data_dir / "upper_back_protocol_reconst_time.csv"
                # time_lsole = data_dir / "left_sole_protocol_reconst_time.csv"
                # time_rsole = data_dir / "right_sole_protocol_reconst_time.csv"

                # special treatment for prelim 6. comment out this if running that file and run that file alone!
                # time_rarm = data_dir / "merged_ARM_RIGHT.csv"
                # time_larm = data_dir / "merged_ARM_LEFT.csv"
                # time_lb = data_dir / "merged_LB.csv"
                # time_ub = data_dir / "merged_UB.csv"
                # time_lsole = data_dir / "merged_lsole_time.csv"
                # time_rsole = data_dir / "merged_rsole_time.csv"
                
                
               
                # special handle for test 14
                time_rarm = data_dir / "rarm_merged.csv"
                time_lb = data_dir / "lback_merged.csv"
                time_lsole = data_dir / "lsole_merged.csv"
                time_rsole = data_dir / "rsole_merged.csv"
                
                time_ub = None
                time_larm = None
                
                

                df_time_rarm = None
                df_time_larm = None
                df_time_lb = None
                df_time_ub = None
                df_time_left = None
                df_time_right = None

                df_time_rarm = pd.read_csv(time_rarm)
                if time_larm is not None:
                    df_time_larm = pd.read_csv(time_larm)
                else:
                    df_time_larm = None
                df_time_lb = pd.read_csv(time_lb)
                if time_ub is not None:
                    df_time_ub = pd.read_csv(time_ub)
                else:
                    df_time_ub = None
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

        if df_time_rarm is None or df_time_lb is None or df_time_lsole is None or df_time_rsole is None:
            # do time alignment otherwise skip directly to applying labels and rep_ids
            # use raw files for this
            print("Beginning time reconstruction....")

            df_list = [df_rarm, df_larm, df_lsole, df_rsole, df_lb, df_ub]

            # Required IMUs
            time_rarm = fix_time_muse_hz(df_list, 0, str(raw_rarm), IMU_original_fs)
            time_lb = fix_time_muse_hz(df_list, 4, str(raw_lb), IMU_original_fs)

            # Optional IMUs
            if df_larm is not None and raw_larm is not None:
                time_larm = fix_time_muse_hz(df_list, 1, str(raw_larm), IMU_original_fs)
            else:
                print("WARNING: left_arm skipped during time reconstruction")
                time_larm = None
                df_time_larm = None

            if df_ub is not None and raw_ub is not None:
                time_ub = fix_time_muse_hz(df_list, 5, str(raw_ub), IMU_original_fs)
            else:
                print("WARNING: upper_back skipped during time reconstruction")
                time_ub = None
                df_time_ub = None

            # Required soles
            df_dupli_left, outpath_dupli_left = remove_end_duplicates(str(raw_lsole))
            df_dupli_right, outpath_dupli_right = remove_end_duplicates(str(raw_rsole))

            df_time_left, time_lsole = add_ReconstructedTime(outpath_dupli_left, df_dupli_left)
            df_time_right, time_rsole = add_ReconstructedTime(outpath_dupli_right, df_dupli_right)
            
            # special case test_9
            # print("\nDEBUG TIME CHECK")
            # print("LEFT sole end time :",
            #     df_time_left["ReconstructedTime"].iloc[-1])

            # print("RIGHT sole end time:",
            #     df_time_right["ReconstructedTime"].iloc[-1])

            # # overwrite left with right
            # df_time_left["ReconstructedTime"] = df_time_right["ReconstructedTime"]

            # print("\nAFTER OVERWRITE")
            # print("LEFT sole end time :",
            #     df_time_left["ReconstructedTime"].iloc[-1])

            # Read reconstructed files
            df_time_rarm = pd.read_csv(time_rarm)
            df_time_lb = pd.read_csv(time_lb)
            #df_time_lsole = pd.read_csv(time_lsole)
            df_time_rsole = pd.read_csv(time_rsole)
            
            df_time_lsole = df_time_left

            if time_larm is not None:
                df_time_larm = pd.read_csv(time_larm)

            if time_ub is not None:
                df_time_ub = pd.read_csv(time_ub)

            print("Finished time alignment.")
        
        # Plot left arm vs right arm
        
        
        # ---------------- DEBUG PLOT: left arm vs right arm ----------------
        import matplotlib.pyplot as plt
        import numpy as np

        def get_acc_cols(df):
            acc_cols = [
                c for c in df.columns
                if any(k in c.lower() for k in ["acc", "axl", "accelerometer"])
            ]
            return acc_cols[:3]

        if df_time_rarm is not None and df_time_larm is not None:

            r_acc = get_acc_cols(df_time_rarm)
            l_acc = get_acc_cols(df_time_larm)

            print("RIGHT ARM acc cols:", r_acc)
            print("LEFT ARM acc cols :", l_acc)

            # acceleration magnitude
            df_time_rarm["acc_mag"] = np.sqrt((df_time_rarm[r_acc] ** 2).sum(axis=1))
            df_time_larm["acc_mag"] = np.sqrt((df_time_larm[l_acc] ** 2).sum(axis=1))

            # select time range
            start_t = 629
            end_t = 680

            r_slice = df_time_rarm[
                (df_time_rarm["ReconstructedTime"] >= start_t) &
                (df_time_rarm["ReconstructedTime"] <= end_t)
            ]

            l_slice = df_time_larm[
                (df_time_larm["ReconstructedTime"] >= start_t) &
                (df_time_larm["ReconstructedTime"] <= end_t)
            ]

            plt.figure(figsize=(14,5))

            plt.plot(
                r_slice["ReconstructedTime"],
                r_slice["acc_mag"],
                label="RIGHT ARM"
            )

            plt.plot(
                l_slice["ReconstructedTime"],
                l_slice["acc_mag"],
                label="LEFT ARM"
            )

            plt.title(f"{participant}: acceleration magnitude")
            plt.xlabel("ReconstructedTime")
            plt.ylabel("Acceleration magnitude")
            plt.legend()

            plt.tight_layout()
            plt.show()

        else:
            print("DEBUG PLOT SKIPPED")
                
        
        
        ### END DEBUG PLOT
        
        
        print('Starting apply labels...')
        # Apply labels
        if protocol == True:
            label_file_name = 'start_stop_labels.csv'
        else:
            label_file_name = 'hall_labels_start_stop.csv'

        label_df = pd.read_csv(data_dir / label_file_name)

        # Apply labeling based on timestamps
        df_label_rarm = apply_time_labels(df_time_rarm, labels_df=label_df)
        df_label_larm = (
            apply_time_labels(df_time_larm, labels_df=label_df)
            if df_time_larm is not None else None
        )
        df_label_lb = apply_time_labels(df_time_lb, labels_df=label_df)
        df_label_ub = (
            apply_time_labels(df_time_ub, labels_df=label_df)
            if df_time_ub is not None else None
        )
        df_label_left = apply_time_labels(df_time_lsole, labels_df=label_df)
        df_label_right = apply_time_labels(df_time_rsole, labels_df=label_df)

        
        if rep_id == True:
            # Assign rep ids if applicable.
            #create frst the rep_id column
            for df in [df_label_rarm, df_label_larm, df_label_lb, df_label_ub, df_label_left, df_label_right]:
                if df is not None:
                    df["rep_id"] = None

            labeled_dfs = {
                "right_arm": df_label_rarm,
                "lower_back": df_label_lb,
                "left_sole": df_label_left,
                "right_sole": df_label_right,
            }

            if df_label_larm is not None:
                labeled_dfs["left_arm"] = df_label_larm

            if df_label_ub is not None:
                labeled_dfs["upper_back"] = df_label_ub

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
            df_label_larm = rep_id_dfs.get('left_arm') # use get for l arm and ub to not fail when not present
            df_label_lb = rep_id_dfs['lower_back']
            df_label_ub = rep_id_dfs.get('upper_back') # use get for l arm and ub to not fail when not present
            df_label_left = rep_id_dfs['left_sole']
            df_label_right = rep_id_dfs['right_sole']
        
        if resample_IMU == True:
            # resample if resample setting is true
            df_label_rarm = resample_imu_dataframe(
            df_label_rarm,
            original_fs=IMU_original_fs,
            target_fs=IMU_target_fs,
            )
            if df_label_larm is not None:
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
            if df_label_ub is not None:
                df_label_ub = resample_imu_dataframe(
                df_label_ub,
                original_fs=IMU_original_fs,
                target_fs=IMU_target_fs,
                )

        outpath_rarm_labeled = f"{str(time_rarm).rstrip('.csv')}_labeled.csv"
        if time_larm is not None:
            outpath_larm_labeled = f"{str(time_larm).rstrip('.csv')}_labeled.csv"
        else:
            outpath_larm_labeled = None
        outpath_lb_labeled = f"{str(time_lb).rstrip('.csv')}_labeled.csv"
        if time_ub is not None:
            outpath_ub_labeled = f"{str(time_ub).rstrip('.csv')}_labeled.csv"
        else:
            outpath_ub_labeled = None
        outpath_lsole_labeled = f"{str(time_lsole).rstrip('.csv')}_labeled.csv"
        outpath_rsole_labeled = f"{str(time_rsole).rstrip('.csv')}_labeled.csv"


        imu_columns = get_imu_cols()
        fsr_columns = get_fsr_cols()

        # save labeled files, apply median filer, drop idle and rotate (if applicable) per sensor
        # right arm, always included
        df_label_rarm.to_csv(outpath_rarm_labeled)
        df_filtered_rarm, path_rarm_filtered = median_filter_medfilt(outpath_rarm_labeled, imu_columns, kernel_size=9)
        df_rarm_no_idle, outpath_rarm_no_idle = remove_idle(correct_label_filepath=path_rarm_filtered, correct_label_df=df_filtered_rarm)
        
        # left arm, included in new files
        if df_label_larm is not None and outpath_larm_labeled is not None:
            df_label_larm.to_csv(outpath_larm_labeled)
            df_filtered_larm, path_larm_filtered = median_filter_medfilt(outpath_larm_labeled, imu_columns, kernel_size=9)
            df_larm_no_idle, outpath_larm_no_idle = remove_idle(correct_label_filepath=path_larm_filtered, correct_label_df=df_filtered_larm)
        else:
            outpath_larm_no_idle = None
                
        # lower back, always included
        df_label_lb.to_csv(outpath_lb_labeled)
        df_filtered_lb, path_lb_filtered = median_filter_medfilt(outpath_lb_labeled, imu_columns, kernel_size=9)
        df_lb_no_idle, outpath_lb_no_idle = remove_idle(correct_label_filepath=path_lb_filtered, correct_label_df=df_filtered_lb)
        if rotate_back == True:
            df_lb_no_idle = rotate_axl_mag_gyr_180_deg_abt_z_axis(outpath_lb_no_idle)
            outpath_lb_no_idle = f"{outpath_lb_no_idle.rstrip('.csv')}_rotated.csv"
            df_lb_no_idle.to_csv(outpath_lb_no_idle)
        
        # upper back, included in new files
        if df_label_ub is not None and outpath_ub_labeled is not None:
            df_label_ub.to_csv(outpath_ub_labeled)
            df_filtered_ub, path_ub_filtered = median_filter_medfilt(outpath_ub_labeled, imu_columns, kernel_size=9)
            df_ub_no_idle, outpath_ub_no_idle = remove_idle(correct_label_filepath=path_ub_filtered, correct_label_df=df_filtered_ub)
            if rotate_back == True:
                df_ub_no_idle = rotate_axl_mag_gyr_180_deg_abt_z_axis(outpath_ub_no_idle)
                outpath_ub_no_idle = f"{outpath_ub_no_idle.rstrip('.csv')}_rotated.csv"
                df_ub_no_idle.to_csv(outpath_ub_no_idle)
        else: 
            outpath_ub_no_idle = None
            
        # soles, always included
        df_label_left.to_csv(outpath_lsole_labeled)
        df_label_right.to_csv(outpath_rsole_labeled)
        df_filtered_lsole, path_lsole_filtered = median_filter_medfilt(outpath_lsole_labeled, fsr_columns, kernel_size=9)
        df_filtered_rsole, path_rsole_filtered = median_filter_medfilt(outpath_rsole_labeled, fsr_columns, kernel_size=9)
        # drop IMU columns from sole data
        df_filtered_lsole = df_filtered_lsole.drop(columns=imu_columns)
        df_filtered_rsole = df_filtered_rsole.drop(columns=imu_columns)
        # drop idle / unlabeled segments
        df_left_no_idle, outpath_lsole_no_idle = remove_idle(correct_label_filepath=path_lsole_filtered, correct_label_df=df_filtered_lsole)
        df_right_no_idle, outpath_rsole_no_idle = remove_idle(correct_label_filepath=path_rsole_filtered, correct_label_df=df_filtered_rsole)
    
        
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
        if outpath_larm_no_idle is not None:
            cleaned_dfs['left_arm'].to_csv(f"{outpath_larm_no_idle.rstrip('.csv')}_cleaned.csv")
        cleaned_dfs['right_arm'].to_csv(f"{outpath_rarm_no_idle.rstrip('.csv')}_cleaned.csv")
        if outpath_ub_no_idle is not None:
            cleaned_dfs['upper_back'].to_csv(f"{outpath_ub_no_idle.rstrip('.csv')}_cleaned.csv")
        cleaned_dfs['lower_back'].to_csv(f"{outpath_lb_no_idle.rstrip('.csv')}_cleaned.csv")
        cleaned_dfs['left_fsr'].to_csv(f"{outpath_lsole_no_idle.rstrip('.csv')}_cleaned.csv")
        cleaned_dfs['right_fsr'].to_csv(f"{outpath_rsole_no_idle.rstrip('.csv')}_cleaned.csv")

        # ---------------- DEBUG FINAL CLEANED SIGNALS ----------------
        import matplotlib.pyplot as plt
        import numpy as np

        def get_acc_cols(df):
            acc_cols = [
                c for c in df.columns
                if any(k in c.lower() for k in ["acc", "axl", "accelerometer"])
            ]
            return acc_cols[:3]

        if (
            "right_arm" in cleaned_dfs
           # and "left_arm" in cleaned_dfs
            and cleaned_dfs["right_arm"] is not None
            #and cleaned_dfs["left_arm"] is not None
        ):

            r_df = cleaned_dfs["right_arm"].copy()
            #l_df = cleaned_dfs["left_arm"].copy()

            r_acc = get_acc_cols(r_df)
           # l_acc = get_acc_cols(l_df)

            print("\nFINAL CLEANED DEBUG")
            print("RIGHT ARM acc cols:", r_acc)
            #print("LEFT ARM acc cols :", l_acc)

            r_df["acc_mag"] = np.sqrt((r_df[r_acc] ** 2).sum(axis=1))
            #l_df["acc_mag"] = np.sqrt((l_df[l_acc] ** 2).sum(axis=1))

            start_t = 0
            end_t = 722

            if "ReconstructedTime" in r_df.columns: #and "ReconstructedTime" in l_df.columns:

                r_slice = r_df[
                    (r_df["ReconstructedTime"] >= start_t)
                    & (r_df["ReconstructedTime"] <= end_t)
                ]

                # l_slice = l_df[
                #     (l_df["ReconstructedTime"] >= start_t)
                #     & (l_df["ReconstructedTime"] <= end_t)
                # ]

                plt.figure(figsize=(14,5))

                plt.plot(
                    r_slice["ReconstructedTime"],
                    r_slice["acc_mag"],
                    label="RIGHT ARM CLEANED"
                )

                # plt.plot(
                #     l_slice["ReconstructedTime"],
                #     l_slice["acc_mag"],
                #     label="LEFT ARM CLEANED"
                # )

                plt.title(f"{participant}: CLEANED acceleration magnitude")
                plt.xlabel("ReconstructedTime")
                plt.ylabel("Acceleration magnitude")
                plt.legend()

                plt.tight_layout()
                plt.show()

            else:
                print("ReconstructedTime column missing in cleaned dfs")

        else:
            print("Could not plot cleaned arm dfs")


        print('Done :))')