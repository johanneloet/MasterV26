import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

from feature_extraction.get_paths import get_test_folder_paths
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils import map_label_hierarchical, drop_label
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
                f"Features_{feature_mode}_{test_id}_expanded{expanded_fsr}_{sensor_combo_scenario}.csv"
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

def plot_kmeans_model_selection(results_df):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    plt.plot(results_df["k"], results_df["silhouette"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("K-means model selection")
    plt.tight_layout()
    plt.show()

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
    combined_features['label'] = combined_features['label'].apply(map_label_hierarchical)

    combined_features = drop_label(combined_features, "lying")
    combined_features = drop_label(combined_features, "break")

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

    return out_df, km, pca, scaler



def plot_pca_clusters(scores_df, pc_x="PC1", pc_y="PC2", cluster_col="cluster"):
    import matplotlib.pyplot as plt
    import numpy as np
    cmap = plt.get_cmap("Set3")
    unique_clusters = sorted(scores_df[cluster_col].unique())
    colors = cmap(np.linspace(0.35, 0.75, 5))
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



def summarize_clusters_long(df, cluster_col="cluster", label_col="label", top_n=None):
    import pandas as pd

    rows = []

    df['label'] = df['label'].apply(map_label_hierarchical)

    for cl in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == cl]
        counts = sub[label_col].value_counts()
        total = counts.sum()

        for label, count in counts.items():
            rows.append({
                "cluster": cl,
                "label": label,
                "count": int(count),
                "percentage": count / total,
            })

    out = pd.DataFrame(rows).sort_values(
        ["cluster", "count"], ascending=[True, False]
    )

    if top_n is not None:
        out = (
            out.groupby("cluster", group_keys=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    return out

def save_cluster_summaries_png(df, cluster_col="cluster", label_col="label", top_n=10, out_dir="cluster_tables"):
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_clusters_long(df, cluster_col=cluster_col, label_col=label_col, top_n=top_n)

    for cl in sorted(summary["cluster"].unique()):
        sub = summary[summary["cluster"] == cl].copy()
        sub["percentage"] = (100 * sub["percentage"]).round(1).astype(str) + "%"

        fig_h = max(2.5, 0.45 * len(sub) + 1)
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=sub[["label", "count", "percentage"]].values,
            colLabels=["label", "count", "%"],
            loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.3)

        plt.title(f"Cluster {cl}: top labels")
        plt.savefig(out_dir / f"cluster_{cl}_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def cluster_label_crosstab(df, cluster_col="cluster", label_col="label", map_hierarchical=True):
    plot_df = df.copy()

    if map_hierarchical:
        plot_df[label_col] = plot_df[label_col].apply(map_label_hierarchical)

    ct_counts = pd.crosstab(plot_df[cluster_col], plot_df[label_col])
    ct_pct = pd.crosstab(
        plot_df[cluster_col],
        plot_df[label_col],
        normalize="index"
    ) * 100

    return ct_counts, ct_pct

def save_kmeans_cluster_label_heatmap(
    df,
    filename="kmeans_cluster_label_heatmap.png",
    cluster_col="cluster",
    label_col="label",
    map_hierarchical=True,
    sort_labels=True,
    annotate=True,
):

    _, ct_pct = cluster_label_crosstab(
        df,
        cluster_col=cluster_col,
        label_col=label_col,
        map_hierarchical=map_hierarchical,
    )

    # optional: order labels by overall frequency for cleaner plot
    if sort_labels:
        label_order = ct_pct.sum(axis=0).sort_values(ascending=False).index
        ct_pct = ct_pct[label_order]

    # pastel BuPu-style colormap
    pastel_bupu = LinearSegmentedColormap.from_list(
        "pastel_bupu",
        ["#f7fcfd", "#e0ecf4", "#bfd3e6", "#9ebcda", "#c994c7", "#ddcce6"]
    )

    fig_w = max(8, 0.55 * ct_pct.shape[1])
    fig_h = max(4.5, 0.7 * ct_pct.shape[0])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        ct_pct.values,
        aspect="auto",
        cmap=pastel_bupu, # caps the darkest shade before 100%
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("% inside cluster")

    ax.set_xticks(np.arange(ct_pct.shape[1]))
    ax.set_xticklabels(ct_pct.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(ct_pct.shape[0]))
    ax.set_yticklabels(ct_pct.index)

    ax.set_xlabel("Label")
    ax.set_ylabel("K-means cluster")
    ax.set_title("K-means cluster composition by label (%)")

    if annotate:
        for i in range(ct_pct.shape[0]):
            for j in range(ct_pct.shape[1]):
                val = ct_pct.iloc[i, j]
                if val >= 5:
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return ct_pct

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
    random_state=88,)

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
    random_state=42,
    )

    plot_pca_clusters(clustered_df)
    
    summary_df = summarize_clusters_long(clustered_df, top_n=10)
    print(summary_df)
    save_cluster_summaries_png(clustered_df, top_n=50)

    ct_pct = save_cluster_label_heatmap(
    clustered_df,
    filename="kmeans_cluster_label_heatmap.png",
    map_label_fn=map_label_hierarchical,
    drop_noise=False,   # keep -1 in plot
    min_total_label_count=0,
    sort_labels=False,
    sort_clusters=False,
    )

    print(ct_pct.round(1))
