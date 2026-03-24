import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN

from feature_extraction.get_paths import get_test_folder_paths
import matplotlib.pyplot as plt

from utils import map_label_hierarchical, drop_label
from matplotlib.colors import LinearSegmentedColormap

from plotting.cluster_plots import save_cluster_label_heatmap


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
    eps=2.5,
    min_samples=20,
    use_pca=True,
    n_pca=0.95,
    min_cluster_size=10,
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
                f"Features_{feature_mode}_{test_id}_expanded{expanded_fsr}_{sensor_combo_scenario}.csv"
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
    combined_features['original_label'] = combined_features['label']
    combined_features['label'] = combined_features['label'].apply(map_label_hierarchical)

    combined_features= drop_label(combined_features, 'lying')
    #combined_features= drop_label(combined_features, 'break')

    static_labels = combined_features["static_label"]
    combined_features['label'] = combined_features['original_label']
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

    mask = ~X.isna().any(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    meta = meta.loc[mask].reset_index(drop=True)

    scaler = StandardScaler()
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
        n_jobs=-1 
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

    return out_df, db, pca, scaler

def plot_pca_dbscan_clusters(
    scores_df,
    pc_x="PC1",
    pc_y="PC2",
    cluster_col="cluster",
    static_by=None,
    static_values=("static",),
):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    scores_df = scores_df.copy()

    if static_by is not None:
        scores_df["_static_state"] = scores_df[static_by].isin(static_values)
        print(scores_df[[static_by, "_static_state"]].drop_duplicates().sort_values(static_by))

    unique_clusters = sorted(scores_df[cluster_col].dropna().unique())

    non_noise = [c for c in unique_clusters if c != -1]
    cmap = plt.get_cmap("Set3")

    colors = cmap(np.linspace(0, 1, max(len(non_noise), 1)))
    cluster_to_color = {cl: colors[i] for i, cl in enumerate(non_noise)}
    cluster_to_color[-1] = (0.75, 0.75, 0.75, 0.6)  # grey noise

    plt.figure(figsize=(10, 8))

    for cl in unique_clusters:
        sub = scores_df[scores_df[cluster_col] == cl]

        if static_by is not None:
            sub_static = sub[sub["_static_state"] == True]
            sub_nonstatic = sub[sub["_static_state"] == False]

            if not sub_static.empty:
                plt.scatter(
                    sub_static[pc_x],
                    sub_static[pc_y],
                    color=cluster_to_color[cl],
                    marker="o",
                    s=30,
                    alpha=0.75,
                    label=None,
                )

            if not sub_nonstatic.empty:
                plt.scatter(
                    sub_nonstatic[pc_x],
                    sub_nonstatic[pc_y],
                    color=cluster_to_color[cl],
                    marker="X",
                    s=30,
                    alpha=0.75,
                    label=None,
                )
        else:
            plt.scatter(
                sub[pc_x],
                sub[pc_y],
                color=cluster_to_color[cl],
                s=30,
                alpha=0.75,
                label=None,
            )

    # cluster legend
    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cluster_to_color[cl],
            markersize=8,
            label="noise" if cl == -1 else str(cl),
        )
        for cl in unique_clusters
    ]

    plt.gca().add_artist(
        plt.legend(handles=cluster_handles, bbox_to_anchor=(1.02, 1), loc="upper left", title=cluster_col)
    )

    # static/non-static legend
    if static_by is not None:
        marker_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="None",
                markersize=8,
                label="Static",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color="black",
                linestyle="None",
                markersize=8,
                label="Non-static",
            ),
        ]

        plt.legend(handles=marker_handles, bbox_to_anchor=(1.02, 0.55), loc="upper left", title=static_by)

    plt.title("PCA colored by HDBSCAN clusters")
    plt.xlabel(pc_x)
    plt.ylabel(pc_y)
    plt.tight_layout()
    plt.show()

def summarize_labels_in_clusters(scores_df, cluster_col="cluster", label_col="label"):

    scores_df['label'] = scores_df['label'].apply(map_label_hierarchical)
    # Counts of labels inside each cluster
    counts = (
        scores_df.groupby([cluster_col, label_col])
        .size()
        .reset_index(name="count")
        .sort_values([cluster_col, "count"], ascending=[True, False])
    )

    # Percent composition within each cluster
    counts["cluster_total"] = counts.groupby(cluster_col)["count"].transform("sum")
    counts["pct_within_cluster"] = 100 * counts["count"] / counts["cluster_total"]

    return counts

# def save_cluster_label_heatmap(
#     ct_pct,
#     ct_counts,
#     filename,
#     title="Cluster vs Label (%)"
# ):
#     import matplotlib.pyplot as plt
#     import numpy as np

#     fig_w = max(8, 0.6 * ct_pct.shape[1])
#     fig_h = max(6, 0.6 * ct_pct.shape[0])

#     plt.figure(figsize=(fig_w, fig_h))

#     pastel_bupu = LinearSegmentedColormap.from_list(
#         "pastel_bupu",
#         ["#f7fcfd", "#e0ecf4", "#bfd3e6", "#9ebcda", "#c994c7", "#ddcce6"]
#     )

#     im = plt.imshow(
#         ct_pct.values,
#         aspect="auto",
#         cmap=pastel_bupu,
#         vmin=0,
#         vmax=60   # better contrast than 100
#     )

#     plt.colorbar(im, label="% inside cluster")

#     plt.xticks(
#         np.arange(ct_pct.shape[1]),
#         ct_pct.columns,
#         rotation=45,
#         ha="right",
#         fontsize=8
#     )

#     plt.yticks(
#         np.arange(ct_pct.shape[0]),
#         ct_pct.index,
#         fontsize=8
#     )

#     # ⭐ annotate BOTH percentage + counts
#     for i in range(ct_pct.shape[0]):
#         for j in range(ct_pct.shape[1]):

#             pct = ct_pct.iloc[i, j]
#             count = ct_counts.iloc[i, j]

#             if count > 0:
#                 plt.text(
#                     j,
#                     i,
#                     f"{pct:.0f}%\n(n={count})",
#                     ha="center",
#                     va="center",
#                     fontsize=7
#                 )

#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()

def cluster_label_crosstab(scores_df, cluster_col="cluster", label_col="label"):
    ct = pd.crosstab(scores_df[cluster_col], scores_df[label_col])
    ct_pct = pd.crosstab(
        scores_df[cluster_col],
        scores_df[label_col],
        normalize="index"
    ) * 100
    return ct, ct_pct

import matplotlib.pyplot as plt
import mplcursors

def interactive_pca_plot(X_pca, clusters, labels):

    fig, ax = plt.subplots(figsize=(8,6))

    sc = ax.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=clusters,
        cmap="tab20",
        s=20
    )

    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        sel.annotation.set_text(
            f"idx: {i}\ncluster: {clusters[i]}\nlabel: {labels[i]}"
        )

    plt.title("Interactive PCA scatter")
    plt.show()



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


    interactive_pca_plot(
    dbscan_df[["PC1", "PC2"]].to_numpy(),
    dbscan_df["cluster"].to_numpy(),
    dbscan_df["label"].to_numpy(),
)
    plot_pca_dbscan_clusters(dbscan_df, static_by="static_label")

    counts = summarize_labels_in_clusters(dbscan_df)
    ct_counts, ct_pct = cluster_label_crosstab(dbscan_df)
    save_cluster_label_heatmap(
    dbscan_df,
    filename="dbscan_cluster_heatmap.png",
    cluster_col="cluster",
    label_col="label",
    title="HDBSCAN: Cluster vs Label (%)",
    map_label_fn=map_label_hierarchical,
    drop_noise=False,   # keep -1 in plot
    min_total_label_count=0,
    sort_labels=False,
    sort_clusters=False,
)

  