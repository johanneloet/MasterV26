# File for creating consistent sensor coordinate frame setups. Rotating the back IMU sensors 180 degrees about the z-axis. 
# Applicable and must be done for prelim 1-7 and akso 1-4!!

# imports
import pandas as pd
from feature_extraction.get_paths import get_test_file_paths
from utils import get_imu_cols

# define rotation function
def rotate_axl_mag_gyr_180_deg_abt_z_axis(imu_file : str):
    df = pd.read_csv(imu_file)
    imu_cols = get_imu_cols()
    rotate_cols = [c for c in imu_cols if 'Z' not in c]
    # flip sign (same as 180 degree rotation)
    df[rotate_cols] = -df[rotate_cols]

    return df


# script for rotating all relevant tests
if __name__ == '__main__':
    participants_to_rotate = [
        'prelim_1',
        'prelim_2',
        'prelim_3',
        'prelim_4',
        'prelim_5',
        # 'prelim_6',
        # 'prelim_7',
        'akso_1',
        'akso_2',
        'akso_3',
        'akso_4',
    ]

    test_files_per_participant = get_test_file_paths()

    for p in participants_to_rotate:
        file_paths = test_files_per_participant[p]

        lb_df_rotated = rotate_axl_mag_gyr_180_deg_abt_z_axis(file_paths['lower_back'])
        ub_df_rotated = rotate_axl_mag_gyr_180_deg_abt_z_axis(file_paths['upper_back'])

        lb_df_rotated.to_csv(f'{file_paths['lower_back'].rstrip('.csv')}_rotated.csv', index=False)
        ub_df_rotated.to_csv(f'{file_paths['upper_back'].rstrip('.csv')}_rotated.csv', index=False)

        # WHEN DONE UPDATE GET PATHS WITH NEW FILE NAMES !! (add the _rotated suffix)
    print("Done! Remember to update get_paths with the rotated filenames:)")

