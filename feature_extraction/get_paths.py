import os
from pathlib import Path
# File handling methodology was devloped by Maria (Sylte 2025), the available files have been exteded through this work
BASE_DIR = Path(r"")

def get_test_file_paths():
    tests = {
        "test_1": {
            "right_arm": BASE_DIR / "test_1" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_1" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_1" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_1" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_2": {
            "right_arm": BASE_DIR / "test_2" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_2" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_2" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_2" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_3": {
            "right_arm": BASE_DIR / "test_3" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_3" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_3" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_3" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_4": {
            "right_arm": BASE_DIR / "test_4" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_4" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_4" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_4" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_5": {
            "right_arm": BASE_DIR / "test_5" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_5" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_5" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_5" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_6": {
            "right_arm": BASE_DIR / "test_6" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_6" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_6" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_6" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_7": {
            "arm": BASE_DIR / "test_7" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "back": BASE_DIR / "test_7" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left": BASE_DIR / "test_7" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right": BASE_DIR / "test_7" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_8": {
            "right_arm": BASE_DIR / "test_8" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_8" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_8" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_8" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_9": {
            "right_arm": BASE_DIR / "test_9" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_9" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_9" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_9" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_10": {
            "right_arm": BASE_DIR / "test_10" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_10" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_10" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_10" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_11": {
            "right_arm": BASE_DIR / "test_11" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_11" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_11" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_11" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_12": {
            "right_arm": BASE_DIR / "test_12" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_12" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_12" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_12" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_13": {
            "right_arm": BASE_DIR / "test_13" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_13" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_13" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_13" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_14": {
            "arm": BASE_DIR / "test_14" / "rarm_merged_labeled_median_filter_no_idle_cleaned.csv",
            "back": BASE_DIR / "test_14" / "lback_merged_labeled_median_filter_no_idle_cleaned.csv",
            "left": BASE_DIR / "test_14" / "lsole_merged_labeled_median_filter_no_idle_cleaned.csv",
            "right": BASE_DIR / "test_14" / "rsole_merged_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_15": {
            "right_arm": BASE_DIR / "test_15" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_15" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_15" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_15" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_16": {
            "right_arm": BASE_DIR / "test_16" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_16" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_16" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_16" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_17": {
            "right_arm": BASE_DIR / "test_17" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_17" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_17" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_17" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_18": {
            "right_arm": BASE_DIR / "test_18" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_18" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_18" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_18" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_19": {
            "right_arm": BASE_DIR / "test_19" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_19" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_19" / "mitch_B0510-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_19" / "mitch_B0308-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "test_20": {
            "right_arm": BASE_DIR / "test_20" / "Muse_E2511_RED-ARM_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "test_20" / "Muse_E2511_GREY-BACK_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "test_20" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "test_20" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "prelim_2": {
            "left_arm": BASE_DIR / "prelim_2" / "Muse_E2511_GREY-ARM_LEFT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "prelim_2" / "Muse_E2511_RED-ARM_RIGHT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "prelim_2" / "muse_v3-UB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "prelim_2" / "muse_v3_3-LB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "prelim_2" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "prelim_2" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "prelim_3": {
            "left_arm": BASE_DIR / "prelim_3" / "Muse_E2511_GREY-ARM_LEFT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "prelim_3" / "Muse_E2511_RED-ARM_RIGHT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "prelim_3" / "muse_v3-UB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "prelim_3" / "muse_v3_3-LB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "prelim_3" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "prelim_3" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "prelim_4": {
            "left_arm": BASE_DIR / "prelim_4" / "Muse_E2511_GREY-ARM_LEFT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "prelim_4" / "Muse_E2511_RED-ARM_RIGHT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "prelim_4" / "muse_v3-UB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "prelim_4" / "muse_v3_3-LB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "prelim_4" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "prelim_4" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "prelim_5": {
            "left_arm": BASE_DIR / "prelim_5" / "Muse_E2511_GREY-ARM_LEFT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "prelim_5" / "Muse_E2511_RED-ARM_RIGHT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "prelim_5" / "muse_v3-UB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "prelim_5" / "muse_v3_3-LB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "prelim_5" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "prelim_5" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "prelim_6": {
            "left_arm": BASE_DIR / "prelim_6" / "merged_ARM_LEFT_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "prelim_6" / "merged_ARM_RIGHT_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "prelim_6" / "merged_UB_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "prelim_6" / "merged_LB_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "prelim_6" / "merged_lsole_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "prelim_6" / "merged_rsole_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "prelim_7": {
            "left_arm": BASE_DIR / "prelim_7" / "Muse_E2511_RED-ARM_LEFT_new_time_hz_labeled_median_filter_no_idle_cleaned_prelim_only.csv",
            "right_arm": BASE_DIR / "prelim_7" / "Muse_E2511_GREY-ARM_RIGHT_new_time_hz_labeled_median_filter_no_idle_cleaned_prelim_only.csv",
            "upper_back": BASE_DIR / "prelim_7" / "muse_v3-UB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned_prelim_only.csv",
            "lower_back": BASE_DIR / "prelim_7" / "muse_v3_3-LB_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned_prelim_only.csv",
            "left_fsr": BASE_DIR / "prelim_7" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned_prelim_only.csv",
            "right_fsr": BASE_DIR / "prelim_7" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned_prelim_only.csv"
        },
        "prelim_8": {
            "left_arm": BASE_DIR / "prelim_8" / "Muse_E2511_RED-ARM_LEFT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "prelim_8" / "Muse_E2511_GREY-ARM_RIGHT_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "prelim_8" / "muse_v3-UB_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "prelim_8" / "muse_v3_3-LB_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "prelim_8" / "mitch_B0308-LEFT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "prelim_8" / "mitch_B0510-RIGHT_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksoprotocol_1": {
            "left_arm": BASE_DIR / "akso_1" / "left_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_1" / "right_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_1" / "upper_back_protocol_reconst_time_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "akso_1" / "lower_back_protocol_reconst_time_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_1" / "left_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_1" / "right_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksoprotocol_2": {
            "left_arm": BASE_DIR / "akso_2" / "left_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_2" / "right_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_2" / "upper_back_protocol_reconst_time_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "akso_2" / "lower_back_protocol_reconst_time_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_2" / "left_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_2" / "right_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksoprotocol_3": {
            "left_arm": BASE_DIR / "akso_3" / "left_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_3" / "right_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_3" / "upper_back_protocol_reconst_time_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "akso_3" / "lower_back_protocol_reconst_time_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_3" / "left_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_3" / "right_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksoprotocol_4": {
            "left_arm": BASE_DIR / "akso_4" / "left_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_4" / "right_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_4" / "upper_back_protocol_reconst_time_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "akso_4" / "lower_back_protocol_reconst_time_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_4" / "left_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_4" / "right_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksoprotocol_5": {
            "left_arm": BASE_DIR / "akso_5" / "left_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_5" / "right_arm_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_5" / "upper_back_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "akso_5" / "lower_back_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_5" / "left_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_5" / "right_sole_protocol_reconst_time_labeled_median_filter_no_idle_cleaned.csv"
            }, 
        "aksowork_1": {
            "left_arm": BASE_DIR / "akso_1" / "left_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_1" / "right_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_1" / "upper_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "akso_1" / "lower_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_1" / "left_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_1" / "right_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksowork_2": {
            "left_arm": BASE_DIR / "akso_2" / "left_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_2" / "right_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_2" / "upper_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "akso_2" / "lower_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_2" / "left_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_2" / "right_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksowork_3": {
            "left_arm": BASE_DIR / "akso_3" / "left_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_3" / "right_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_3" / "upper_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "akso_3" / "lower_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_3" / "left_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_3" / "right_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksowork_4": {
            "left_arm": BASE_DIR / "akso_4" / "left_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_4" / "right_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_4" / "upper_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "lower_back": BASE_DIR / "akso_4" / "lower_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_rotated_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_4" / "left_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_4" / "right_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"
        },
        "aksowork_5": {
            "left_arm": BASE_DIR / "akso_5" / "left_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "right_arm": BASE_DIR / "akso_5" / "right_arm_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "upper_back": BASE_DIR / "akso_5" / "upper_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "lower_back": BASE_DIR / "akso_5" / "lower_back_hall_reconst_time_new_time_hz_labeled_median_filter_no_idle_cleaned.csv",
            "left_fsr": BASE_DIR / "akso_5" / "left_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv",
            "right_fsr": BASE_DIR / "akso_5" / "right_sole_hall_reconst_time_del_end_dupli_new_time_labeled_median_filter_no_idle_cleaned.csv"},
        
        
    }
    return tests

def get_test_folder_paths():
    folders = {
        "test_1": BASE_DIR / "test_1",
        "test_2": BASE_DIR / "test_2",
        "test_3": BASE_DIR / "test_3",
        "test_4": BASE_DIR / "test_4",
        "test_5": BASE_DIR / "test_5",
        "test_6": BASE_DIR / "test_6",
        "test_7": BASE_DIR / "test_7",
        "test_8": BASE_DIR / "test_8",
        "test_9": BASE_DIR / "test_9",
        "test_10": BASE_DIR / "test_10",
        "test_11": BASE_DIR / "test_11",
        "test_12": BASE_DIR / "test_12",
        "test_13": BASE_DIR / "test_13",
        "test_14": BASE_DIR / "test_14",
        "test_15": BASE_DIR / "test_15",
        "test_16": BASE_DIR / "test_16",
        "test_17": BASE_DIR / "test_17",
        "test_18": BASE_DIR / "test_18",
        "test_19": BASE_DIR / "test_19",
        "test_20": BASE_DIR / "test_20",
        "prelim_2": BASE_DIR / "prelim_2",
        "prelim_3": BASE_DIR / "prelim_3",
        "prelim_4": BASE_DIR / "prelim_4",
        "prelim_5": BASE_DIR / "prelim_5",
        "prelim_6": BASE_DIR / "prelim_6",
        "prelim_7": BASE_DIR / "prelim_7",
        "prelim_8": BASE_DIR / "prelim_8",
        "aksoprotocol_1": BASE_DIR / "akso_1",
        "aksoprotocol_2": BASE_DIR / "akso_2",
        "aksoprotocol_3": BASE_DIR / "akso_3",
        "aksoprotocol_4": BASE_DIR / "akso_4",
        "aksoprotocol_5": BASE_DIR / "akso_5",
        "aksowork_1": BASE_DIR / "akso_1",
        "aksowork_2": BASE_DIR / "akso_2",
        "aksowork_3": BASE_DIR / "akso_3",
        "aksowork_4": BASE_DIR / "akso_4",
        "aksowork_5": BASE_DIR / "akso_5",
    }
    return folders


def get_one_test(test_number):
    test_to_get = "test_" + str(test_number)
    return get_test_file_paths()[test_to_get]


def get_one_file(test_number, sensor):
    test_nr_to_get = "test_" + str(test_number)
    return get_test_file_paths()[test_nr_to_get][sensor]


def get_one_foler_path(test_number):
    if isinstance(test_number, int):
        test_number = "test_" + str(test_number)
    return get_test_folder_paths()[test_number]


def get_feture_paths(
    window_length_sec=4,
    norm_IMU=True,
    mean_fsr=False,
    hdr=False,
    feature_space="baseline",
):
    folders = get_test_folder_paths()
    feature_files = {}

    for test_id, folder in folders.items():
        if mean_fsr is None:
            filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_no_fsr_hdr{'T' if hdr else 'F'}_{feature_space}.csv"
        else:
            filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_mean{'T' if mean_fsr else 'F'}_hdr{'T' if hdr else 'F'}_{feature_space}.csv"
        full_path = os.path.join(folder, filename)
        feature_files[test_id] = full_path

    return feature_files


def get_feature_paths_for_multiple_spaces(
    window_length_sec=4,
    norm_IMU=True,
    mean_fsr=False,
    hdr=False,
    feature_spaces=[
        "baseline",
        "expanded+baseline",
        "expanded_only",
        "time_only",
        "time_only+exp_FSR",
        "freq_only",
        "freq_only+exp_FSR",
        "FSR_only",
        "arm_only",
        "back_only",
    ],
):
    folders = get_test_folder_paths()
    feature_files = {}

    for test_id, folder in folders.items():
        paths = []
        for feature_space in feature_spaces:
            if mean_fsr is None:
                filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_no_fsr_hdr{'T' if hdr else 'F'}_{feature_space}.csv"
            else:
                filename = f"{test_id}_features_4sensors_window{window_length_sec}_norm{'T' if norm_IMU else 'F'}_mean{'T' if mean_fsr else 'F'}_hdr{'T' if hdr else 'F'}_{feature_space}.csv"
            full_path = os.path.join(folder, filename)
            paths.append(full_path)
        feature_files[test_id] = paths

    return feature_files