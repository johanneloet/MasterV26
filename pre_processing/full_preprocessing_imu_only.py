"""
IMU-only preprocessing workflow for NON-PROTOCOL / real work data only.

Uses:
- right arm IMU
- left arm IMU, if available
- lower back IMU
- upper back IMU, if available

Does NOT use:
- protocol files
- test-specific merged files
- soles
- FSR data

Expected files per participant:
- right_arm_hall_reconst_time.csv
- left_arm_hall_reconst_time.csv, optional
- lower_back_hall_reconst_time.csv
- upper_back_hall_reconst_time.csv, optional
- hall_labels_start_stop.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pre_processing.fiks_time_muse import fix_time_muse_hz
from pre_processing.filter_sensor_data import median_filter_medfilt
from pre_processing.labeling import remove_idle
from pre_processing.segment_repetition_vol2 import assign_rep_ids
from feature_extraction.resample_old_imu_files import resample_imu_dataframe
from pre_processing.rotate_back_imus import rotate_axl_mag_gyr_180_deg_abt_z_axis
from pre_processing.clean_dataset_columns import clean_and_rename_columns
from utils import get_imu_cols


def apply_time_labels(sensor_df, labels_df, time_col="ReconstructedTime"):
    df = sensor_df.copy()

    labels_df = labels_df.rename(
        columns={
            "Start Time (s)": "start_time",
            "End Time (s)": "end_time",
            "Label": "label",
        }
    )

    if "label" not in df.columns:
        df["label"] = None

    for _, row in labels_df.iterrows():
        mask = (
            (df[time_col] >= row["start_time"])
            & (df[time_col] <= row["end_time"])
        )
        df.loc[mask, "label"] = row["label"]

    return df


def safe_read_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def filter_remove_idle_and_optionally_rotate(
    df,
    outpath_labeled,
    imu_columns,
    rotate=False,
):
    df.to_csv(outpath_labeled, index=False)

    df_filtered, path_filtered = median_filter_medfilt(
        outpath_labeled,
        imu_columns,
        kernel_size=9,
    )

    df_no_idle, outpath_no_idle = remove_idle(
        correct_label_filepath=path_filtered,
        correct_label_df=df_filtered,
    )

    if rotate:
        df_no_idle = rotate_axl_mag_gyr_180_deg_abt_z_axis(outpath_no_idle)
        outpath_no_idle = f"{outpath_no_idle.rstrip('.csv')}_rotated.csv"
        df_no_idle.to_csv(outpath_no_idle, index=False)

    return df_no_idle, outpath_no_idle


def get_acc_cols(df):
    return [
        c for c in df.columns
        if any(k in c.lower() for k in ["acc", "axl", "accelerometer"])
    ][:3]


if __name__ == "__main__":

    base_dir = Path(r"C:\Users\Bruker\Data_msc26_prelim")

    participants = [
        "akso_8",
    ]

    rep_id = False

    IMU_original_fs = 100
    IMU_target_fs = 100
    resample_IMU = False

    rotate_back = False
    debug_plots = False

    for participant in participants:
        print(f"\nPROCESSING NON-PROTOCOL PARTICIPANT {participant}")

        data_dir = base_dir / participant

        time_rarm = data_dir / "right_arm_hall_reconst_time.csv"
        time_larm = data_dir / "left_arm_hall_reconst_time.csv"
        time_lb = data_dir / "lower_back_hall_reconst_time.csv"
        time_ub = data_dir / "upper_back_hall_reconst_time.csv"

        label_path = data_dir / "hall_labels_start_stop.csv"

        df_rarm = safe_read_csv(time_rarm)
        df_larm = safe_read_csv(time_larm)
        df_lb = safe_read_csv(time_lb)
        df_ub = safe_read_csv(time_ub)

        if df_rarm is None:
            raise FileNotFoundError(f"Missing required file: {time_rarm}")

        if df_lb is None:
            raise FileNotFoundError(f"Missing required file: {time_lb}")

        if df_larm is None:
            print(f"WARNING: left arm file not found. Skipping: {time_larm}")
            time_larm = None

        if df_ub is None:
            print(f"WARNING: upper back file not found. Skipping: {time_ub}")
            time_ub = None

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        print("Fixing IMU time for non-protocol data...")

        # Keep original index positions from your workflow:
        # 0 = right arm
        # 1 = left arm
        # 4 = lower back
        # 5 = upper back
        df_list = [df_rarm, df_larm, None, None, df_lb, df_ub]

        time_rarm = fix_time_muse_hz(
            df_list,
            0,
            str(time_rarm),
            IMU_original_fs,
        )

        time_lb = fix_time_muse_hz(
            df_list,
            4,
            str(time_lb),
            IMU_original_fs,
        )

        if df_larm is not None and time_larm is not None:
            time_larm = fix_time_muse_hz(
                df_list,
                1,
                str(time_larm),
                IMU_original_fs,
            )

        if df_ub is not None and time_ub is not None:
            time_ub = fix_time_muse_hz(
                df_list,
                5,
                str(time_ub),
                IMU_original_fs,
            )

        df_time_rarm = pd.read_csv(time_rarm)
        df_time_lb = pd.read_csv(time_lb)
        df_time_larm = pd.read_csv(time_larm) if time_larm is not None else None
        df_time_ub = pd.read_csv(time_ub) if time_ub is not None else None

        print("Finished IMU-only time alignment.")

        if debug_plots and df_time_rarm is not None and df_time_larm is not None:
            r_acc = get_acc_cols(df_time_rarm)
            l_acc = get_acc_cols(df_time_larm)

            print("RIGHT ARM acc cols:", r_acc)
            print("LEFT ARM acc cols :", l_acc)

            if len(r_acc) == 3 and len(l_acc) == 3:
                df_time_rarm["acc_mag"] = np.sqrt(
                    (df_time_rarm[r_acc] ** 2).sum(axis=1)
                )
                df_time_larm["acc_mag"] = np.sqrt(
                    (df_time_larm[l_acc] ** 2).sum(axis=1)
                )

                start_t = 629
                end_t = 680

                r_slice = df_time_rarm[
                    (df_time_rarm["ReconstructedTime"] >= start_t)
                    & (df_time_rarm["ReconstructedTime"] <= end_t)
                ]

                l_slice = df_time_larm[
                    (df_time_larm["ReconstructedTime"] >= start_t)
                    & (df_time_larm["ReconstructedTime"] <= end_t)
                ]

                plt.figure(figsize=(14, 5))
                plt.plot(
                    r_slice["ReconstructedTime"],
                    r_slice["acc_mag"],
                    label="RIGHT ARM",
                )
                plt.plot(
                    l_slice["ReconstructedTime"],
                    l_slice["acc_mag"],
                    label="LEFT ARM",
                )
                plt.title(f"{participant}: acceleration magnitude")
                plt.xlabel("ReconstructedTime")
                plt.ylabel("Acceleration magnitude")
                plt.legend()
                plt.tight_layout()
                plt.show()

        print("Applying non-protocol labels...")

        label_df = pd.read_csv(label_path)

        df_label_rarm = apply_time_labels(df_time_rarm, label_df)
        df_label_lb = apply_time_labels(df_time_lb, label_df)

        df_label_larm = (
            apply_time_labels(df_time_larm, label_df)
            if df_time_larm is not None
            else None
        )

        df_label_ub = (
            apply_time_labels(df_time_ub, label_df)
            if df_time_ub is not None
            else None
        )

        if rep_id:
            print("Assigning rep ids...")

            for df in [df_label_rarm, df_label_larm, df_label_lb, df_label_ub]:
                if df is not None:
                    df["rep_id"] = None

            labeled_dfs = {
                "right_arm": df_label_rarm,
                "lower_back": df_label_lb,
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
                "stairs_down",
            ]

            rep_id_dfs = assign_rep_ids(
                sensor_dfs=labeled_dfs,
                output_dir=data_dir,
                numbered_labels=numbered_labels,
                save_files=False,
            )

            df_label_rarm = rep_id_dfs["right_arm"]
            df_label_lb = rep_id_dfs["lower_back"]
            df_label_larm = rep_id_dfs.get("left_arm")
            df_label_ub = rep_id_dfs.get("upper_back")

        if resample_IMU:
            print("Resampling IMUs...")

            df_label_rarm = resample_imu_dataframe(
                df_label_rarm,
                original_fs=IMU_original_fs,
                target_fs=IMU_target_fs,
            )

            df_label_lb = resample_imu_dataframe(
                df_label_lb,
                original_fs=IMU_original_fs,
                target_fs=IMU_target_fs,
            )

            if df_label_larm is not None:
                df_label_larm = resample_imu_dataframe(
                    df_label_larm,
                    original_fs=IMU_original_fs,
                    target_fs=IMU_target_fs,
                )

            if df_label_ub is not None:
                df_label_ub = resample_imu_dataframe(
                    df_label_ub,
                    original_fs=IMU_original_fs,
                    target_fs=IMU_target_fs,
                )

        print("Filtering IMU streams and removing idle segments...")

        imu_columns = get_imu_cols()

        outpath_rarm_labeled = f"{str(time_rarm).rstrip('.csv')}_labeled.csv"
        outpath_lb_labeled = f"{str(time_lb).rstrip('.csv')}_labeled.csv"

        outpath_larm_labeled = (
            f"{str(time_larm).rstrip('.csv')}_labeled.csv"
            if time_larm is not None
            else None
        )

        outpath_ub_labeled = (
            f"{str(time_ub).rstrip('.csv')}_labeled.csv"
            if time_ub is not None
            else None
        )

        _, outpath_rarm_no_idle = filter_remove_idle_and_optionally_rotate(
            df_label_rarm,
            outpath_rarm_labeled,
            imu_columns,
            rotate=False,
        )

        _, outpath_lb_no_idle = filter_remove_idle_and_optionally_rotate(
            df_label_lb,
            outpath_lb_labeled,
            imu_columns,
            rotate=rotate_back,
        )

        if df_label_larm is not None and outpath_larm_labeled is not None:
            _, outpath_larm_no_idle = filter_remove_idle_and_optionally_rotate(
                df_label_larm,
                outpath_larm_labeled,
                imu_columns,
                rotate=False,
            )
        else:
            outpath_larm_no_idle = None

        if df_label_ub is not None and outpath_ub_labeled is not None:
            _, outpath_ub_no_idle = filter_remove_idle_and_optionally_rotate(
                df_label_ub,
                outpath_ub_labeled,
                imu_columns,
                rotate=rotate_back,
            )
        else:
            outpath_ub_no_idle = None

        print("Cleaning and renaming IMU columns...")

        cleaned_dfs = clean_and_rename_columns(
            left_arm_path=outpath_larm_no_idle,
            right_arm_path=outpath_rarm_no_idle,
            upper_back_path=outpath_ub_no_idle,
            lower_back_path=outpath_lb_no_idle,
        )

        print(f"Saving final IMU-only non-protocol files in {data_dir}")

        if outpath_larm_no_idle is not None and "left_arm" in cleaned_dfs:
            cleaned_dfs["left_arm"].to_csv(
                f"{outpath_larm_no_idle.rstrip('.csv')}_cleaned.csv",
                index=False,
            )

        if "right_arm" in cleaned_dfs:
            cleaned_dfs["right_arm"].to_csv(
                f"{outpath_rarm_no_idle.rstrip('.csv')}_cleaned.csv",
                index=False,
            )

        if outpath_ub_no_idle is not None and "upper_back" in cleaned_dfs:
            cleaned_dfs["upper_back"].to_csv(
                f"{outpath_ub_no_idle.rstrip('.csv')}_cleaned.csv",
                index=False,
            )

        if "lower_back" in cleaned_dfs:
            cleaned_dfs["lower_back"].to_csv(
                f"{outpath_lb_no_idle.rstrip('.csv')}_cleaned.csv",
                index=False,
            )

        print("Done.")