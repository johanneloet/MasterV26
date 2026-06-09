import os
import pandas as pd

#function for adding  hall label to a hall_labels_start_stop.csv file

def add_label(data_dir, label, start_time, end_time, filename="hall_labels_start_stop.csv"):
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