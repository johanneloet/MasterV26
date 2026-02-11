"""
Code to remove undesired columns such as 'timestamp' and 'unnamed'. Also rename the Fsr-columns to the corrects ones.
"""

from feature_extraction.get_paths import get_test_file_paths
import pandas as pd

FILES = get_test_file_paths()

FSR_MAPPING_LEFT = {
    "Fsr.01": "Fsr.11",
    "Fsr.02": "Fsr.10",
    "Fsr.03": "Fsr.09",
    "Fsr.04": "Fsr.14",
    "Fsr.05": "Fsr.12",
    "Fsr.06": "Fsr.16",
    "Fsr.07": "Fsr.13",
    "Fsr.08": "Fsr.15",
    "Fsr.09": "Fsr.01",
    "Fsr.10": "Fsr.02",
    "Fsr.11": "Fsr.06",
    "Fsr.12": "Fsr.07",
    "Fsr.13": "Fsr.03",
    "Fsr.14": "Fsr.05",
    "Fsr.15": "Fsr.08",
    "Fsr.16": "Fsr.04",
}

FSR_MAPPING_RIGHT = {
    "Fsr.01": "Fsr.08",
    "Fsr.02": "Fsr.04",
    "Fsr.03": "Fsr.05",
    "Fsr.04": "Fsr.01",
    "Fsr.05": "Fsr.03",
    "Fsr.06": "Fsr.06",
    "Fsr.07": "Fsr.07",
    "Fsr.08": "Fsr.02",
    "Fsr.09": "Fsr.14",
    "Fsr.10": "Fsr.15",
    "Fsr.11": "Fsr.16",
    "Fsr.12": "Fsr.13",
    "Fsr.13": "Fsr.12",
    "Fsr.14": "Fsr.09",
    "Fsr.15": "Fsr.11",
    "Fsr.16": "Fsr.10",
}


def clean_and_rename_columns(
    left_arm_path=None,
    right_arm_path=None,
    upper_back_path=None,
    lower_back_path=None,
    left_fsr_path=None,
    right_fsr_path=None,
):
    """
    Removes columns containing 'timestamp' or 'unnamed' (case-insensitive),
    and renames FSR columns for left/right insoles using mapping dictionaries.
    Returns the cleaned DataFrames.
    """

    # Helper to drop unwanted columns
    def _drop_unwanted(df):
        drop_cols = [
            c for c in df.columns if "timestamp" in c.lower() or "unnamed" in c.lower()
        ]
        return df.drop(columns=drop_cols, errors="ignore")

    # Map paths to sensor names
    sensor_paths = {
        "left_arm": left_arm_path,
        "right_arm": right_arm_path,
        "upper_back": upper_back_path,
        "lower_back": lower_back_path,
        "left_fsr": left_fsr_path,
        "right_fsr": right_fsr_path,
    }

    cleaned_dfs = {}

    for sensor, path in sensor_paths.items():
        if path is None:
            print(f"⚠️ No file provided for {sensor}, skipping...")
            continue

        # Load CSV
        df = pd.read_csv(path)
        df = _drop_unwanted(df)

        # Rename FSR columns if applicable
        if sensor == "left_fsr":
            df = df.rename(columns=FSR_MAPPING_LEFT)
        elif sensor == "right_fsr":
            df = df.rename(columns=FSR_MAPPING_RIGHT)

        cleaned_dfs[sensor] = df

    return cleaned_dfs


if __name__ == "__main__":
    for test_id, paths in FILES.items():
        if test_id == "prelim_1":  # redo only a single test - fill in the name here
            # clean files
            cleaned_dfs = clean_and_rename_columns(
                left_arm_path=paths.get("left_arm"),
                right_arm_path=paths.get("right_arm"),
                upper_back_path=paths.get("upper_back"),
                lower_back_path=paths.get("lower_back"),
                left_fsr_path=paths.get("left_fsr"),
                right_fsr_path=paths.get("right_fsr"),
            )

            # save any cleaned DataFrames that were returned, ignore nonetypes
            for sensor, df in cleaned_dfs.items():
                orig_path = paths.get(sensor)
                if orig_path is None:
                    continue
                cleaned_path = orig_path.replace(".csv", "_cleaned.csv")
                df.to_csv(cleaned_path, index=False)
                print(f"💾 Saved cleaned {sensor} to {cleaned_path}")
