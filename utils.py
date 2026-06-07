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



def contains_any(label, keywords):
    return any(k in label for k in keywords)

def drop_label(df, label_to_drop):
    return df[df["label"] != label_to_drop].reset_index(drop=True)


def map_taxonomy_candidate_1(label, static_label=None):

    if contains_any(label, ["lying"]):
        return "other"

    if contains_any(label, ["walk", "walking", "stairs", "stair", "ladder", "climbing", "climb"]):
        return "locomotion"

    if contains_any(label, ["push", "pull", "drag", "carry", "lift", "lifting"]):
        return "material_handling"

    if contains_any(label, [
        "lean", "leaning", "forward_lean", "sideways_lean",
        "backward_lean", "torso", "twist", "trunk", "squat", "bend", "bending"
    ]):
        return "trunk_movement"

    if contains_any(label, [
        "hand", "hands", "arm", "arms", "shoulder", 
        "reach", "reaching"
    ]):
        return "arm_movement"

    if contains_any(label, ["standing", "sitting", "neutral", "idle", "resting"]):
        return "neutral"
    return "other"


def map_taxonomy_candidate_2(label, static_label=None):

    if contains_any(label, ["lying"]):
        return "other"

    if contains_any(label, ["walk", "walking", "stairs", "stair", "ladder", "climb", "climbing"]):
        return "locomotion"

    if contains_any(label, [
        "push", "pull", "drag", "carry", "lift", "lifting",
        "load", "squat", "kneel", "bend", "bending"
    ]):
        return "handling_or_lower_body_movement"

    if contains_any(label, [
        "hand", "hands", "arm", "arms", "shoulder", "elbow",
        "reach", "reaching", "overhead",
        "lean", "leaning", "torso", "twist", "trunk"
    ]):
        return "upper_body_movement"

    if contains_any(label, ["standing", "sitting", "neutral", "idle", "resting"]):
        return "neutral"

    return "other"


def map_taxonomy_candidate_3(label, static_label=None):

    if contains_any(label, ["lying"]):
        return "other"

    locomotion_handling_keywords = [
        "walk", "walking",
        "stairs", "stair",
        "climb", "climbing",
        "push", "pull", "drag",
        "carry", "carrying",
    ]

    full_body_engagement_keywords = [
        "squat", "squatting",
        "lift", "lifting",
        "lean", "leaning",
        "forward_lean",
        "sideways_lean",
        "backward_lean",
        "torso",
        "twist", "twisting",
        "trunk",
        "awkward",
    ]

    arm_engagement_keywords = [
        "hand", "hands",
        "arm", "arms",
        "shoulder",
        "elbow",
        "reach", "reaching",
        "overhead",
    ]

    neutral_keywords = [
        "standing",
        "sitting",
        "neutral",
        "neutral_load",
        "idle",
        "resting",
        "rest",
    ]

    # Priority matters
    if contains_any(label, locomotion_handling_keywords):
        return "locomotion_handling"

    if contains_any(label, full_body_engagement_keywords):
        return "full_body_engagement"

    if contains_any(label, arm_engagement_keywords):
        return "arm_engagement"

    if contains_any(label, neutral_keywords):
        return "neutral"

    return "other"


def map_taxonomy_candidate_4(label, static_label=None):
    """
    Candidate 4:
    static / intermediate / locomotion
    """

    if contains_any(label, ["lying"]):
        return "other"

    # label based on static label detection
    if static_label == "static" or contains_any(label, ["static", "standing", "sitting", "idle", "resting", "neutral_load"]):
        return "static"

    if contains_any(label, ["walk", "walking", "stairs", "stair", "climb", "ladder", "climbing"]):
        return "locomotion"

    return "intermediate"


