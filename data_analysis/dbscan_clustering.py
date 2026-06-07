import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN

from feature_extraction.get_paths import get_test_folder_paths

from utils import drop_label

from plotting.cluster_plots import save_cluster_label_heatmap, plot_pca_clusters


def plot_dbscan_k_distance(
    right_arm=True,
    left_arm=True,
    lower_back=True,
    upper_back=True,
    left_fsr=True,
    right_fsr=True,
    expanded_fsr=False,
    prefixes=["prelim"],
    feature_mode="Window",
    k=20,
    use_pca=True,
    n_pca=0.95,
):
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt

    test_folder_dict = get_test_folder_paths()

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

    include_csvs = []
    for test_id, folder_path in test_folder_dict.items():
        if test_id.split("_")[0] in prefixes:
            feature_filename = (
                f"Features_{feature_mode}_{test_id}_expanded{expanded_fsr}_{sensor_combo_scenario}.csv"
            )
            include_csvs.append(Path(folder_path) / feature_filename)

    feature_dfs = [pd.read_csv(p) for p in include_csvs]
    if not feature_dfs:
        raise ValueError("No feature files found for the requested configuration.")

    combined_features = pd.concat(feature_dfs, ignore_index=True)

    metadata_cols = [
        "label",
        "prefix",
        "test_id",
        "rep_id",
        "container_id",
        "window_id",
        "start_idx",
        "end_idx",
        "rep_id.1"
    ]

    X = combined_features.drop(
        columns=[c for c in metadata_cols if c in combined_features.columns],
        errors="ignore",
    )
    X = X.select_dtypes(include="number")

    mask = ~X.isna().any(axis=1)
    X = X.loc[mask].reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_pca:
        pca = PCA(n_components=n_pca)
        X_model = pca.fit_transform(X_scaled)
    else:
        X_model = X_scaled

    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X_model)
    distances, _ = nbrs.kneighbors(X_model)

    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(7, 4))
    plt.plot(k_distances)
    plt.xlabel("Points sorted")
    plt.ylabel(f"{k}-NN distance")
    plt.title("DBSCAN k-distance plot")
    plt.tight_layout()
    plt.show()

    return k_distances

def run_dbscan_on_dataset(
    right_arm=True,
    left_arm=True,
    lower_back=True,
    upper_back=True,
    left_fsr=True,
    right_fsr=True,
    expanded_fsr=False,
    prefixes=["prelim"],
    feature_mode="Window",
    feature_window_sec=3.5,
    map_label_fn=None,
    min_samples=3,
    use_pca=True,
    n_pca=0.95,
    min_cluster_size=50,
):
    test_folder_dict = get_test_folder_paths()

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

    include_csvs = []
    for test_id, folder_path in test_folder_dict.items():
        if test_id.split("_")[0] in prefixes:
            feature_mode_original = feature_mode
            if test_id.split("_")[0] == 'aksowork' and feature_mode == 'Repetition':
                # since a rep mode does nto exist for the work files, use window with 3.5 sec lenght instead.
                feature_mode = 'Window'
            feature_filename = (
                f"Features_{feature_mode}_{test_id}_expanded{expanded_fsr}_SEG{feature_mode}{feature_window_sec}_{sensor_combo_scenario}.csv"
            )
            include_csvs.append(Path(folder_path) / feature_filename)
            feature_mode = feature_mode_original

    feature_dfs = []
    for p in include_csvs:
        df = pd.read_csv(p)
        for col in df.columns:
            if "upper_back" in col.lower():
                print(col)
        
        filename = p.name
        test_id = filename.split(f"Features_")[1].split("_expanded")[0]
        test_id = "_".join(test_id.split("_")[1:])

        parts = test_id.split("_")
        prefix = "_".join(parts[:-1]) if len(parts) > 1 else test_id
        df["prefix"] = prefix
        df["test_id"] = test_id
        feature_dfs.append(df)

    if not feature_dfs:
        raise ValueError("No feature files found for the requested configuration.")

    combined_features = pd.concat(feature_dfs, ignore_index=True)
    combined_features['original_label'] = combined_features['label']
    # combined_features['label'] = combined_features.apply(
    # lambda row: map_label_fn(row["label"], row.get("static_label", None)),
    # axis=1
    # )
    combined_features= drop_label(combined_features, "lying_arms_up")
    # drop all labels containing "break"
    combined_features = combined_features[
        ~combined_features["label"].astype(str).str.contains(
            "break",
            case=False,
            na=False
        )
    ].copy()

    labels = combined_features["label"]
    print("LEN LABELS", len(labels))
    import time
    time.sleep(5)
    static_labels = combined_features["static_label"]
    # combined_features['label'] = combined_features['original_label']
    combined_features.drop(columns=['original_label'])
    metadata_cols = [
        "label",
        "prefix",
        "test_id",
        "rep_id",
        "container_id",
        "window_id",
        "start_idx",
        "end_idx",
        "rep_id.1"
    ]

    meta = combined_features[[c for c in metadata_cols if c in combined_features.columns]].copy()

    X = combined_features.drop(
        columns=[c for c in metadata_cols if c in combined_features.columns],
        errors="ignore",
    )
    X = X.select_dtypes(include="number")
    #print("X before dropping Nan", X)

    mask = ~X.isna().any(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)

    scaler = StandardScaler()
    #print("X is.....")
    #print(X)
    X_scaled = scaler.fit_transform(X)

    if use_pca:
        pca = PCA(n_components=n_pca)
        X_model = pca.fit_transform(X_scaled)
        print(f"PCA explained variance before DBSCAN: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        pca = None
        X_model = X_scaled

    db = HDBSCAN(
        min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=-1,
        cluster_selection_epsilon=0.1,
    )
    clusters = db.fit_predict(X_model)

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = int((clusters == -1).sum())

    print(f"DBSCAN clusters found: {n_clusters}")
    print(f"Noise points: {n_noise}")

    out_df = pd.DataFrame(X_model[:, :2], columns=["PC1", "PC2"])
    out_df["cluster"] = clusters
    out_df["static_label"] = static_labels

    for col in meta.columns:
        out_df[col] = meta[col].values
    
    out_df["label"] = labels

    return out_df, db, pca, scaler

if __name__ == "__main__":
    plot_dbscan_k_distance(
        right_arm=True,
        left_arm=True,
        lower_back=True,
        upper_back=True,
        left_fsr=True,
        right_fsr=True,
        expanded_fsr=True,
        prefixes=["aksowork", "aksoprotocol", "prelim"],
        feature_mode="Window",
        k=20,
        use_pca=True,
        n_pca=0.95,
    )

 
    dbscan_df, db_model, pca_model, scaler = run_dbscan_on_dataset(
        right_arm=True,
        left_arm=True,
        lower_back=True,
        upper_back=True,
        left_fsr=True,
        right_fsr=True,
        expanded_fsr=True,
        prefixes=["aksoprotocol", "prelim", "aksowork"],
        feature_mode="Window",
        min_samples=5,
        n_pca=10,
        min_cluster_size=33
    )


    plot_pca_clusters(dbscan_df, save_path="cluster_plots/TESTPLOT.pdf",static_by="static_label")

