import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

from feature_extraction.get_paths import get_test_folder_paths, get_one_foler_path

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
        if test_id.split('_')[0] in prefixes:
            feature_filename = f"Features_{test_id}_expanded{expanded_fsr}_{sensor_combo_scenario}.csv"
            include_csvs.append(Path(folder_path) / feature_filename)

    feature_dfs = []
    for p in include_csvs:
        df = pd.read_csv(p)
        feature_dfs.append(df)
    
    combined_features = pd.concat(feature_dfs, ignore_index=True)

    y = combined_features["label"]
    X = combined_features.drop(columns=["label"])
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
    scores,
    columns=[f"PC{i+1}" for i in range(scores.shape[1])]
    )

    # add labels back in to be able to plot color coded by label later. 
    scores_df["label"] = y.values

    return scores_df


def plot_pca_scores(scores_df, pc_x=1, pc_y=2):
    x_col = f"PC{pc_x}"
    y_col = f"PC{pc_y}"

    plt.figure(figsize=(8,6))

    for label in scores_df["label"].unique():
        subset = scores_df[scores_df["label"] == label]
        plt.scatter(subset[x_col], subset[y_col], label=label, alpha=0.7)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"PCA Score Plot ({x_col} vs {y_col})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pca_subplots(scores_df, pairs=None, color_by="label", ncols=3, alpha=0.7):
    """
    Plot multiple PCA score plots as subplots.

    pairs: list of tuples like [(1,2), (1,3), ...]
           If None, defaults to [(1,2), (1,3), (1,4), (1,5), (1,6)]  (5 plots)
    """
    if pairs is None:
        pairs = [(1, i) for i in range(2, 7)]  # PC1 vs PC2..PC6 (5 plots)

    nplots = len(pairs)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)

    labels = scores_df[color_by].unique()

    for idx, (pc_x, pc_y) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        x_col = f"PC{pc_x}"
        y_col = f"PC{pc_y}"

        for lab in labels:
            sub = scores_df[scores_df[color_by] == lab]
            ax.scatter(sub[x_col], sub[y_col], label=lab, alpha=alpha)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")

    # Hide any unused axes
    for idx in range(nplots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    # One shared legend (cleaner than repeating)
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for legend
    plt.show()

if __name__ == '__main__': 
    scores = run_pca_on_dataset(expanded_fsr=True)
    plot_pca_scores(scores)
    plot_pca_subplots(scores)







    
    

    


