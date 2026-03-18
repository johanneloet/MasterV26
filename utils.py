import re
import pandas as pd

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


def map_label_hierarchical(label):
    """
    Map detailed labels into broader groups using ordered rules.
    More specific rules must come before broader ones.
    """
    if pd.isna(label):
        return None

    label = str(label).lower().strip()
    label = re.sub(r"[\s\-]+", "_", label)

    def has(*terms):
        return all(term in label for term in terms)

    def has_any(*terms):
        return any(term in label for term in terms)

    # ---- very specific static / welding cases first ----
    if has("welding", "upright"):
        return "static_upright"

    if has("welding") and has_any("lean", "left", "right", "twist"):
        return "static_lean_twist"

    if has("welding", "arm"):
        return "static_arms"
    if has("welding"):
        return "static"

    if has_any("static_arms", "resting_on_right_knee"):
        return "static_arms"

    if has_any("static_upright", "sitting", "sit", "stand", "standing"):
        return "static_upright"

    if has_any("static_lean_forward", "static_lean"):
        return "static_lean"

    # ---- posture / lying ----
    if "lying" in label:
        return "lying"

    if "break" in label:
        return "break"

    # ---- locomotion ----
    if has_any("stairs", "stair", "ladder"):
        return "stairs_ladder"

    if has_any("walk", "walking"):
        return "walking"

    # ---- carrying / handling ----
    if "carry" in label:
        return "carry"

    if has_any("pull", "drag"):
        return "drag"

    if has_any("neutral_load_left", "neutral_load_right", "load_left", "load_right", "remove_gear"):
        return "load_handle"

    # ---- trunk / posture changes ----
    if "twist" in label:
        return "twist"

    if has_any("lean", "bend"):
        return "lean_bend"

    # ---- upper-body / arm dominant ----
    if has("arm", "forward"):
        return "arms_forward"
    if has("shoulder"):
        return "shoulder_load"

    # ---- vibration / tool use ----
    if has_any("drill", "grind", "vibration"):
        return "vibration"

    # ---- squat / low posture ----
    if has_any("squat", "sit_squat", "crawl", "climb"):
        return "low_movement"

    return label

def drop_label(df, label_to_drop):
    return df[df["label"] != label_to_drop].reset_index(drop=True)