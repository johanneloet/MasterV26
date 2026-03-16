import os
import pandas as pd

def add_label(data_dir, label, start_time, end_time, filename="hall_labels_start_stop.csv"):
    """
    Create or append to a label CSV.

    Parameters
    ----------
    data_dir : str or Path
        Directory where the label file should live
    label : str
        Name of the label/activity
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
    filename : str
        Name of the label file
    """

    filepath = os.path.join(data_dir, filename)

    # Create file if it doesn't exist
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=["label", "start_time", "end_time"])
    else:
        df = pd.read_csv(filepath)

    # Add new row
    new_row = {
        "label": label,
        "start_time": start_time,
        "end_time": end_time
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Save
    df.to_csv(filepath, index=False)

    print(f"Label '{label}' added ({start_time}s → {end_time}s)")