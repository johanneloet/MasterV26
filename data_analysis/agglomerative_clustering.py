import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from feature_extraction.get_paths import get_test_folder_paths
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

from utils import drop_label
from plotting.cluster_plots import save_cluster_label_heatmap


def run_agglomerative_on_dataset(
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
    n_clusters=4,
    linkage_method="ward",
    metric="euclidean",
    use_pca=True,
    n_pca=0.95,
    map_label_fn=None
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
                feature_window_sec = 3.5
            feature_filename = (
                f"Features_{feature_mode}_{test_id}_expanded{expanded_fsr}_SEG{feature_mode}{feature_window_sec}_{sensor_combo_scenario}.csv"
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
    #combined_features["label"] = combined_features["label"].apply(map_label_fn)
    # combined_features['label'] = combined_features.apply(
    # lambda row: map_label_fn(row["label"], row.get("static_label", None)),
    # axis=1
    # )
    
    # drop all labels containing "break"
    combined_features = combined_features[
        ~combined_features["label"].astype(str).str.contains(
            "break",
            case=False,
            na=False
        )
    ].copy()

    labels = combined_features["label"]
        
    combined_features = drop_label(combined_features, "lying_arms_up")

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
        "rep_id.1",
    ]

    meta = combined_features[[c for c in metadata_cols if c in combined_features.columns]].copy()

    X = combined_features.drop(
        columns=[c for c in metadata_cols if c in combined_features.columns],
        errors="ignore",
    )
    X = X.select_dtypes(include="number")

    mask = ~X.isna().any(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_pca:
        pca = PCA(n_components=n_pca)
        X_model = pca.fit_transform(X_scaled)
        print(f"PCA explained variance before Agglomerative: {pca.explained_variance_ratio_.sum():.4f}")
        print(f"PCA shape: {X_model.shape}")
    else:
        pca = None
        X_model = X_scaled
        print(f"Feature shape: {X_model.shape}")

    if linkage_method == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
            metric="euclidean",
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=metric,
        )

    clusters = model.fit_predict(X_model)

    print(f"Agglomerative clusters found: {len(np.unique(clusters))}")
    print(pd.Series(clusters).value_counts().sort_index())

    out_df = pd.DataFrame(X_model[:, :2], columns=["PC1", "PC2"])
    out_df["cluster"] = clusters
    out_df["static_label"] = static_labels
    
    for col in meta.columns:
        out_df[col] = meta[col].values
    
    out_df["label"] = labels

    for col in meta.columns:
        out_df[col] = meta[col].values

    return out_df, model, pca, scaler, X_model


def plot_pca_agglomerative_clusters(scores_df, pc_x="PC1", pc_y="PC2", cluster_col="cluster"):
    unique_clusters = sorted(scores_df[cluster_col].unique())
    cmap = plt.get_cmap("Set3")

    # pick pastel region (avoid very dark end)
    colors = cmap(np.linspace(0.35, 0.75, 8))
    cluster_to_color = {cl: colors[i] for i, cl in enumerate(unique_clusters)}

    plt.figure(figsize=(10, 8))

    for cl in unique_clusters:
        sub = scores_df[scores_df[cluster_col] == cl]

        plt.scatter(
            sub[pc_x],
            sub[pc_y],
            color=cluster_to_color[cl],
            s=30,
            alpha=0.75,
            label=str(cl),
        )

    plt.title(f"PCA colored by {cluster_col}")
    plt.xlabel(pc_x)
    plt.ylabel(pc_y)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_agglomerative_dendrogram(
    X,
    filename,
    method="ward",
    truncate_mode="level",
    p=5,
    figsize=(12, 6),
):
    Z = linkage(X, method=method)

    plt.figure(figsize=figsize)

    dendrogram(
        Z,
        truncate_mode=truncate_mode,
        p=p
    )

    plt.title(f"Agglomerative dendrogram ({method})")
    plt.xlabel("Samples / merged clusters")
    plt.ylabel("Distance")

    plt.tight_layout()

    plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close()

    return Z


def summarize_labels_in_clusters(scores_df, cluster_col="cluster", label_col="label"):
    plot_df = scores_df.copy()
    plot_df[label_col] = plot_df[label_col].apply(map_label_hierarchical)

    counts = (
        plot_df.groupby([cluster_col, label_col])
        .size()
        .reset_index(name="count")
        .sort_values([cluster_col, "count"], ascending=[True, False])
    )

    counts["cluster_total"] = counts.groupby(cluster_col)["count"].transform("sum")
    counts["pct_within_cluster"] = 100 * counts["count"] / counts["cluster_total"]

    return counts


def cluster_label_crosstab(scores_df, cluster_col="cluster", label_col="label"):
    plot_df = scores_df.copy()
    plot_df[label_col] = plot_df[label_col].apply(map_label_hierarchical)

    ct = pd.crosstab(plot_df[cluster_col], plot_df[label_col])
    ct_pct = pd.crosstab(
        plot_df[cluster_col],
        plot_df[label_col],
        normalize="index"
    ) * 100
    return ct, ct_pct


if __name__ == "__main__":
    aggl_df, aggl_model, pca_model, scaler, X_model = run_agglomerative_on_dataset(
        right_arm=True,
        left_arm=True,
        lower_back=True,
        upper_back=True,
        left_fsr=True,
        right_fsr=True,
        expanded_fsr=True,
        prefixes=["aksowork", "aksoprotocol", "prelim"],
        feature_mode="Window",
        n_clusters=7,
        linkage_method="ward",
        metric="euclidean",
        use_pca=True,
        n_pca=0.85,
    )

    plot_pca_agglomerative_clusters(aggl_df)

    plot_agglomerative_dendrogram(
        X_model,
        method="ward",
        truncate_mode="level",
        p=5,
    )

    # counts = summarize_labels_in_clusters(aggl_df)
    # ct_counts, ct_pct = cluster_label_crosstab(aggl_df)

    save_cluster_label_heatmap(
        aggl_df,
        filename="agglomerative_cluster_heatmap.png",
        cluster_col="cluster",
        label_col="label",
        title="Agglomerative: Cluster vs Label (%)",
        map_label_fn=None,
        drop_noise=False,
        min_total_label_count=0,
        sort_labels=False,
        sort_clusters=False,
    )