import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from feature_extraction.get_paths import get_test_folder_paths
import matplotlib.pyplot as plt

from utils import drop_label
from plotting.cluster_plots import save_cluster_label_heatmap

from matplotlib.colors import LinearSegmentedColormap

def run_kmeans_model_selection_on_dataset(
    right_arm=True,
    left_arm=True,
    lower_back=True,
    upper_back=True,
    left_fsr=True,
    right_fsr=True,
    expanded_fsr=False,
    prefixes=["prelim"],
    feature_mode="Window",
    feature_window_length=3.5,
    k_values=range(2, 13),
    use_pca=True,
    n_pca=0.95,
    random_state=42,
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
            feature_filename = (
                f"Features_{feature_mode}_{test_id}_expanded{expanded_fsr}_SEG{feature_mode}{feature_window_length}_{sensor_combo_scenario}.csv"
            )
            include_csvs.append(Path(folder_path) / feature_filename)

    feature_dfs = [pd.read_csv(p) for p in include_csvs]
    if not feature_dfs:
        raise ValueError("No feature files found for the requested configuration.")

    combined_features = pd.concat(feature_dfs, ignore_index=True)

    combined_features = drop_label(combined_features, "lying_arms_up")

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
        print(f"PCA explained variance before K-means: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        X_model = X_scaled

    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=50, random_state=random_state)
        labels = km.fit_predict(X_model)
        sil = silhouette_score(X_model, labels)
        results.append({"k": k, "silhouette": sil})
        print(f"k={k}: silhouette={sil:.4f}")

    return pd.DataFrame(results)

def plot_kmeans_model_selection(results_df, filename):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))

    plt.plot(
        results_df["k"],
        results_df["silhouette"],
        marker="o"
    )

    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("K-means model selection")

    plt.tight_layout()

    plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close()

def run_kmeans_on_dataset(
    right_arm=True,
    left_arm=True,
    lower_back=True,
    upper_back=True,
    left_fsr=True,
    right_fsr=True,
    expanded_fsr=False,
    prefixes=["prelim"],
    feature_mode="Window",
    feature_window_length=3.5,
    map_label_fn = None,
    n_clusters=8,
    use_pca=True,
    n_pca=0.95,
    random_state=42,
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
            if test_id.split("_")[0] == 'aksowork' and feature_mode == 'Repetition':
                # since a rep mode does nto exist for the work files, use window with 3.5 sec lenght instead.
                feature_mode = 'Window'
                feature_window_length = 3.5
            feature_filename = (
            f"Features_{feature_mode}_{test_id}_expanded{expanded_fsr}_SEG{feature_mode}{feature_window_length}_{sensor_combo_scenario}.csv"
            )
            include_csvs.append(Path(folder_path) / feature_filename)

    feature_dfs = []
    for p in include_csvs:
        df = pd.read_csv(p)

        filename = p.name
        test_id = filename.split("Features_")[1].split("_expanded")[0]
        parts = test_id.split("_")
        prefix = "_".join(parts[:-1]) if len(parts) > 1 else test_id

        df["prefix"] = prefix
        df["test_id"] = test_id
        feature_dfs.append(df)

    if not feature_dfs:
        raise ValueError("No feature files found for the requested configuration.")

    combined_features = pd.concat(feature_dfs, ignore_index=True)
    #combined_features['label'] = combined_features['label'].apply(map_label_fn)
    # combined_features['label'] = combined_features.apply(
    # lambda row: map_label_fn(row["label"], row.get("static_label", None)),
    # axis=1
    # )
    
    combined_features = drop_label(combined_features, "lying_arms_up")
    
    # drop all labels containing "break"
    combined_features = combined_features[
        ~combined_features["label"].astype(str).str.contains(
            "break",
            case=False,
            na=False
        )
    ].copy()

    labels = combined_features["label"]


    static_labels = combined_features["static_label"]

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

    print(X.columns)

    mask = ~X.isna().any(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_pca:
        pca = PCA(n_components=n_pca)
        X_model = pca.fit_transform(X_scaled)
        print(f"PCA explained variance before K-means: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        pca = None
        X_model = X_scaled

    km = KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
    clusters = km.fit_predict(X_model)

    out_df = pd.DataFrame(X_model[:, :2], columns=["PC1", "PC2"])
    out_df["cluster"] = clusters
    out_df["static_label"] = static_labels
    for col in meta.columns:
        out_df[col] = meta[col].values
    
    out_df["label"] = labels

    return out_df, km, pca, scaler


if __name__ == '__main__':
    results = run_kmeans_model_selection_on_dataset(
    right_arm=True,
    left_arm=True,
    lower_back=True,
    upper_back=True,
    left_fsr=True,
    right_fsr=True,
    expanded_fsr=True,
    prefixes=["aksowork", "prelim", "aksoprotocol"],
    feature_mode="Window",
    k_values=range(2, 13),
    use_pca=True,
    n_pca=0.95,
    random_state=343)

    plot_kmeans_model_selection(results)


    clustered_df, km, pca, scaler = run_kmeans_on_dataset(
    right_arm=True,
    left_arm=True,
    lower_back=True,
    upper_back=True,
    left_fsr=True,
    right_fsr=True,
    expanded_fsr=True,
    prefixes=["aksowork", "prelim", "aksoprotocol"],
    feature_mode="Window",
    n_clusters=4,
    use_pca=True,
    n_pca=0.95,
    random_state=343,
    )