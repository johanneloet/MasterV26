import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from feature_extraction.get_paths import get_test_folder_paths
from pathlib import Path

from utils import map_label_hierarchical


def run_tsne_on_dataset(
    right_arm=True,
    left_arm=True,
    lower_back=True,
    upper_back=True,
    left_fsr=True,
    right_fsr=True,
    expanded_fsr=False,
    prefixes=["prelim"],
    feature_mode="Window",   # or "Repetition"
    n_pca=0.95, # define in terms of expl. variance ratio
    perplexity=40,
    random_state=42,
):
    test_folder_dict = get_test_folder_paths()

    # Define sensor combination string to identify correct files
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

    # Find relevant feature files
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

        # Safer prefix extraction from test_id directly
        prefix = test_id.split("_")[1]

        df["prefix"] = prefix
        df["test_id"] = test_id

        feature_dfs.append(df)

    combined_features = pd.concat(feature_dfs, ignore_index=True)

    y = combined_features["label"]
    prefix = combined_features["prefix"]
    test_id = combined_features["test_id"]

    # Drop non-feature columns
    drop_cols = ["label", "prefix", "test_id", "rep_id"]
    X = combined_features.drop(columns=[c for c in drop_cols if c in combined_features.columns])

    # Keep only numeric columns
    X = X.select_dtypes(include="number")

    # Remove rows with NaNs
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    prefix = prefix.loc[mask]
    test_id = test_id.loc[mask]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA before t-SNE
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA explained variance before t-SNE: {pca.explained_variance_ratio_.sum():.4f}")

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    tsne_scores = tsne.fit_transform(X_pca)

    tsne_df = pd.DataFrame(tsne_scores, columns=["TSNE1", "TSNE2"])
    tsne_df["label"] = y.values
    tsne_df["prefix"] = prefix.values
    tsne_df["test_id"] = test_id.values

    print(tsne_df['prefix'].unique())

    return tsne_df, tsne, pca

# def plot_tsne_scores(tsne_df):
#     import matplotlib.pyplot as plt
#     import numpy as np

#     plt.figure(figsize=(10, 8))

#     markers = {"prelim": "X", "akso": "o"}

#     unique_labels = sorted(tsne_df["label"].unique())

#     # create color map
#     cmap = plt.cm.get_cmap("tab20", len(unique_labels))
#     label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

#     for label in unique_labels:
#         subset = tsne_df[tsne_df["label"] == label]

#         for pref in subset["prefix"].unique():
#             sub2 = subset[subset["prefix"] == pref]

#             plt.scatter(
#                 sub2["TSNE1"],
#                 sub2["TSNE2"],
#                 color=label_to_color[label],
#                 label=f"{label} ({pref})",
#                 alpha=0.7,
#                 s=25,
#                 marker=markers.get(pref, "o"),
#             )

#     plt.title("t-SNE embedding")
#     plt.xlabel("TSNE1")
#     plt.ylabel("TSNE2")

#     # remove duplicate legend entries
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(),
#                bbox_to_anchor=(1.05, 1), loc="upper left")

#     plt.tight_layout()
#     plt.show()


def get_many_distinct_colors(n):
    """
    Build a long list of visually distinct qualitative colors.
    """

    cmaps = [
        plt.cm.tab20,
        plt.cm.tab20b,
        plt.cm.tab20c,
        plt.cm.Set3,
        plt.cm.Dark2,
        plt.cm.Paired,
    ]

    colors = []

    for cmap in cmaps:
        colors.extend(cmap(np.linspace(0, 1, cmap.N)))

    if n > len(colors):
        raise ValueError("Too many labels — need different strategy")

    return colors[:n]

def plot_tsne_scores(tsne_df, legend_cols=3):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    # ---------- DATASET DOMAIN ----------
    tsne_df = tsne_df.copy()
    tsne_df["domain"] = np.where(
        tsne_df["prefix"] == "aksowork",
        "akso_work",
        "other"
    )

    markers = {
        "akso_work": "X",
        "other": "o"
    }

    # ---------- COLORS FOR LABELS ----------

    tsne_df["label_grouped"] = tsne_df["label"].apply(map_label_hierarchical)
    unique_labels = sorted(tsne_df["label_grouped"].unique())

    colors = get_many_distinct_colors(len(unique_labels))
    label_to_color = {lab: colors[i] for i, lab in enumerate(unique_labels)}

    # ---------- TSNE FIGURE ----------
    plt.figure(figsize=(10, 8))

    for lab in unique_labels:
        sub_lab = tsne_df[tsne_df["label"] == lab]

        for dom in sub_lab["domain"].unique():
            sub = sub_lab[sub_lab["domain"] == dom]

            plt.scatter(
                sub["TSNE1"],
                sub["TSNE2"],
                color=label_to_color[lab],
                marker=markers[dom],
                alpha=0.7,
                s=30,
            )

    plt.title("t-SNE embedding")
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.tight_layout()
    plt.show()

    # ---------- LEGEND 1: LABEL COLORS ----------
    label_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=lab,
            markerfacecolor=label_to_color[lab],
            markersize=8,
        )
        for lab in unique_labels
    ]

    n_rows = int(np.ceil(len(unique_labels) / legend_cols))
    fig_height = max(2, 0.4 * n_rows)

    plt.figure(figsize=(4 * legend_cols, fig_height))
    plt.legend(
        handles=label_handles,
        loc="center",
        ncol=legend_cols,
        frameon=False
    )
    plt.axis("off")
    plt.title("Movement labels")
    plt.tight_layout()
    plt.show()

    # ---------- LEGEND 2: DOMAIN MARKERS ----------
    domain_handles = [
        Line2D([0], [0], marker="o", color="black",
               linestyle="None", label="Prelim + Akso protocol"),
        Line2D([0], [0], marker="X", color="black",
               linestyle="None", label="Akso real work"),
    ]

    plt.figure(figsize=(4, 2))
    plt.legend(handles=domain_handles, loc="center", frameon=False)
    plt.axis("off")
    plt.title("Dataset domain")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    SEEDS = [42, 363, 60, 51]
    for s in SEEDS:
        tsne_df, _, _ = run_tsne_on_dataset(
        right_arm=True,
        left_arm=True,
        lower_back=True,
        upper_back=True,
        left_fsr=True,
        right_fsr=True,
        expanded_fsr=True,
        prefixes=["prelim", "aksoprotocol", "aksowork"],
        feature_mode="Window",   
        n_pca=0.95,
        perplexity=40,
        random_state=s,
        )
        plot_tsne_scores(tsne_df=tsne_df)