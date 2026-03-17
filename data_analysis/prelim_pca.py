import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np

from feature_extraction.get_paths import get_test_folder_paths, get_one_foler_path
from plotting.pca_plots import plot_pca_scores, plot_pca_subplots, plot_scree


def run_pca_on_dataset(
    right_arm=True,
    left_arm=True,
    lower_back=True,
    upper_back=True,
    left_fsr=True,
    right_fsr=True,
    expanded_fsr=False,
    prefixes=["prelim"],
):
    test_folder_dict = get_test_folder_paths()

    # Define sensor combination string to identify correct files.
    sensor_flags = {
        "right_arm": right_arm,
        "left_arm": left_arm,
        "lower_back": lower_back,
        "upper_back": upper_back,
        "left_fsr": left_fsr,
        "right_fsr": right_fsr,
    }
    sensor_order = [
        "right_arm",
        "left_arm",
        "lower_back",
        "upper_back",
        "left_fsr",
        "right_fsr",
    ]

    sensor_config = [s for s in sensor_order if sensor_flags[s]]

    sensor_combo_scenario = "_".join(sensor_config)

    # Find all files corresponding to the prefixes to be included (e.g. only the preliminary tests or all currently existing testfiles...)
    include_csvs = []
    for test_id, folder_path in test_folder_dict.items():
        if test_id.split("_")[0] in prefixes:
            feature_filename = (
                f"Features_{test_id}_expanded{expanded_fsr}_{sensor_combo_scenario}.csv"
            )
            include_csvs.append(Path(folder_path) / feature_filename)

    feature_dfs = []
    for p in include_csvs:
        df = pd.read_csv(p)
        # Extract test_id from filename
        filename = p.name  # e.g. "Features_prelim_01_expandedFalse_..."
        test_id = filename.split("Features_")[1].split("_expanded")[0]

        prefix = test_id.split("_")[0]  # "prelim" or "test"

        df["prefix"] = prefix  # add prefix column

        feature_dfs.append(df)

    combined_features = pd.concat(feature_dfs, ignore_index=True)

    y = combined_features["label"]
    prefix = combined_features["prefix"]

    X = combined_features.drop(columns=["label", "prefix"])
    # Remove any non-numeric columns
    X = X.select_dtypes(include="number")

    # Remove rows with NaNs
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=10)
    scores = pca.fit_transform(X_scaled)

    scores_df = pd.DataFrame(
        scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])]
    )

    # add labels and prefixes back in to be able to plot color coded by label and dataset later
    scores_df["label"] = y.values
    scores_df["prefix"] = prefix.values

    return scores_df, pca


if __name__ == "__main__":
    scores, pca = run_pca_on_dataset(
        left_arm=False,
        upper_back=False,
        expanded_fsr=True,
        prefixes=["akso", "prelim"],
    )
    plot_scree(pca)
    plot_pca_scores(scores, pc_x=1, pc_y=2)
    plot_pca_subplots(scores, color_by="label", style_by=None)
