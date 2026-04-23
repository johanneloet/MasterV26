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
    if pd.isna(label):
        return None

    label = str(label).lower().strip()
    label = re.sub(r"[\s\-]+", "_", label)

    def has(*terms):
        return all(term in label for term in terms)

    def has_any(*terms):
        return any(term in label for term in terms)

    if "break" in label:
        return "break"

    if "lying" in label:
        return "lying"

    if has("welding", "upright"):
        return "welding_upright"

    if has("welding") and has_any("lean", "left", "right", "twist"):
        return "welding_lean_twist"

    if has("welding", "arm"):
        return "welding_arms"

    if has("welding"):
        return "welding"

    if has_any("stairs", "stair", "ladder"):
        return "stairs_ladder"

    if has_any("walk", "walking"):
        return "walking"

    if "carry" in label:
        return "carry"

    if has_any("pull", "drag", "push"):
        return "push_pull_drag"

    if has_any("neutral_load_left", "neutral_load_right"):
        return "neutral_load"

    if "twist" in label:
        return "twist"

    if has_any("lean_forward", "forward_lean", "bend_forward", "forward_bend"):
        return "forward_lean"

    if has_any("lean_left", "left_lean", "bend_left", "left_bend"):
        return "left_lean"

    if has_any("lean_right", "right_lean", "bend_right", "right_bend"):
        return "right_lean"

    if has_any("lean", "bend"):
        return "trunk_flexion_unspecified"

    if "left_arm" in label:
        return "left_arm"

    if has_any("right_arm", "shoulder_load"):
        return "right_arm"

    if re.search(r"\barms?\b", label):
        return "both_arms_movement"

    if has("arm", "forward"):
        return "arms_forward"

    if has("shoulder"):
        return "shoulder_load"

    if has_any("drill", "grind", "vibration"):
        return "vibration"

    if has_any("squat", "sit_squat", "crawl", "climb"):
        return "low_movement"

    return label

def map_label_coarse(hier, static_label):
    if hier is None:
        return None

    if static_label == "static":
        return "static"

    if hier in {"walking", "stairs_ladder"}:
        return "locomotion"

    return "intermediate"

def drop_label(df, label_to_drop):
    return df[df["label"] != label_to_drop].reset_index(drop=True)


def map_label_taxonomy_v1(label):
    """
    Experimental finer-grained taxonomy.
    Only maps semantic/task labels.
    Does NOT infer static.
    """
    if pd.isna(label):
        return None

    label = str(label).lower().strip()
    label = re.sub(r"[\s\-]+", "_", label)

    def has(*terms):
        return all(term in label for term in terms)

    def has_any(*terms):
        return any(term in label for term in terms)

    # ---- break / equipment ----
    if has_any(
        "break",
        "break_put_on_equipment",
        "break_putting_on_equipment",
        "remove_gear",
    ):
        return "break"

    # ---- stairs / ladder ----
    if has_any("stairs", "stair", "ladder"):
        return "stairs_ladder"

    # ---- walking / carrying ----
    if has_any("walk", "walking", "carry"):
        return "walk_carry"

    # ---- standing ----
    if has_any("standing", "stand", "break_stand"):
        return "standing"

    # ---- kneeling ----
    if "knee" in label or has_any("kneel", "kneeling"):
        return "kneel"

    # ---- trunk flexion ----
    if has_any("lean", "bend"):
        return "trunk_flexion"

    # ---- arm movement ----
    if has_any("arms_90", "arm_90", "arms_up", "arm_up"):
        return "arm_movement"

    if "left_arm" in label:
        return "arm_movement"

    if "right_arm" in label:
        return "arm_movement"

    if has("arm", "forward") or has_any("arms_forward", "forward_arm", "forward_arms"):
        return "arm_movement"

    # ---- shoulder load ----
    if has_any("shoulder_load", "shoulder"):
        return "shoulder_load"

    # ---- push / pull / drag ----
    if has_any("push", "pull", "drag"):
        return "push_pull_drag"

    # ---- twist ----
    if "twist" in label:
        return "twist"

    # ---- vibration / tools ----
    if has_any("drill", "grind", "vibration"):
        return "vibration"

    # ---- low movement ----
    if has_any("squat", "sit_squat", "crawl", "climb"):
        return "low_movement"

    # ---- lying ----
    if "lying" in label:
        return "lying"

    # ---- neutral load ----
    if has_any("neutral_load_left", "neutral_load_right", "neutral_load"):
        return "neutral_load"

    return label


# def map_label_taxonomy_posture_focus(label):
#     if pd.isna(label):
#         return None

#     label = str(label).lower().strip()
#     label = re.sub(r"[\s\-]+", "_", label)

#     def has_any(*terms):
#         return any(term in label for term in terms)

#     # static posture semantics
#     if has_any("lean", "flex", "bend", "trunk_flexion", "lean_forward", "forward_lean", "sideways_lean"):
#         return "trunk_flexion"

#     if has_any("neutral_load", "standing", "stand", "sitting", "sit", "upright"):
#         return "neutral"

#     # dynamic locomotion
#     if has_any("walk", "walking", "stairs", "stair", "ladder"):
#         return "locomotion"

#     # dynamic arm-dominant
#     if has_any(
#         "left_arm", "right_arm", "arms_forward", "arms_up", "arms_90",
#         "arm", "shoulder_load", "shoulder"
#     ):
#         return "arm_movement"

#     # dynamic handling
#     if has_any("carry", "push", "pull", "drag"):
#         return "load_handling"

#     if has_any("twist", "vibration", "drill", "grind", "kneel", "crawl", "squat", "climb"):
#         return "other_dynamic"

#     if has_any("break", "remove_gear", "break_put_on_equipment", "break_putting_on_equipment"):
#         return "break"

#     return "other_dynamic"


def map_label_taxonomy_posture_focus(label):
    if pd.isna(label):
        return None

    if label == "lying_arms_up":
        return label

    label = str(label).lower().strip()
    label = re.sub(r"[\s\-]+", "_", label)

    def has_any(*terms):
        return any(term in label for term in terms)

    # ---- trunk posture / flexion semantics ----
    if has_any("arm_up", "arms_up", "shoulder_load"):
        return("arm_elevation")

    if has_any(
        "lean", "flex", "bend", "trunk_flexion",
        "lean_forward", "forward_lean", "sideways_lean", "twist", "grinding_forward"
    ):
        return "trunk_engaged"

    # ---- neutral / upright posture semantics ----
    if has_any(
        "standing", "stand", "sitting", "sit",
        "upright", "neutral_load"
    ):
        return "trunk_neutral"

    # ---- locomotion ----
    
    if has_any( "push", "pull", "drag", "carry", "squat"):
        return "handling"
    if has_any("walk", "walking"):
        return "walking"
    if has_any("stairs", "stair", "ladder"):
        return "stairs_ladder"
    if has_any("climb"):
        return "climbing"

    if has_any(
        "left_arm", "right_arm",
        "arms_forward", "arms_90",
        "arm", "shoulder",
    ):
        return "arm_motion"

    if has_any(
        "break",
        "remove_gear",
        "break_put_on_equipment",
        "break_putting_on_equipment"
    ):
        return "break"
    if has_any("crawl", "kneel", "knee"):
        return "drop"

    return label

def map_label_coarse_posture_focus(mapped_label, static_label):
    if mapped_label is None:
        return None

    if static_label == "static":
        if "trunk_neutral" in mapped_label:
            return "upper_body_neutral_static"
        elif "trunk" in mapped_label:
            return "upper_body_static"
        elif "lying_arms_up" in mapped_label:
            return "lying_arms_up"
        elif "arm" in mapped_label:
            return "upper_body_static"
    else:
        if "trunk_neutral" in mapped_label:
            return "upper_body_neutral_motion"
        elif "trunk" in mapped_label:
            return "upper_body_motion"
        elif "lying_arms_up" in mapped_label:
            return "lying_arms_up"
        elif "arm" in mapped_label:
            return "upper_body_motion"

    return mapped_label


# def map_label_coarse_posture_focus(mapped_label, static_label):
#     if mapped_label is None:
#         return None

#     if static_label == "static":
#         return "static"
#     else:
#         if mapped_label == "walking" or mapped_label == "stairs_ladder" or mapped_label == "handling":
#             return "locomotion"
#         else:
#             return "intermediate"

#     return mapped_label
