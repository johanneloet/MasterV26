def get_imu_cols():
    """
    Standard IMU columns (accelerometer, gyroscope, magnetometer)
    """
    return [
        "Axl.X",
        "Axl.Y",
        "Axl.Z",  # accelerometer
        "Gyr.X",
        "Gyr.Y",
        "Gyr.Z",  # gyroscope
        "Mag.X",
        "Mag.Y",
        "Mag.Z",  # magnetometer
    ]


def get_fsr_cols():
    """
    Standard 16 FSR channels
    """
    return [f"Fsr.{i:02d}" for i in range(1, 17)]


def get_nearest_row_by_time(df, t, time_col="ReconstructedTime"):
    i = (df[time_col] - t).abs().idxmin()
    return df.loc[i]
