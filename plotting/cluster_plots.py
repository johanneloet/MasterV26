import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import os

def save_cluster_label_heatmap(
    df,
    filename,
    cluster_col="cluster",
    label_col="label",
    title="Cluster vs Label (%)",
    map_label_fn=None,
    drop_noise=False,
    noise_label=-1,
    min_total_label_count=0,
    sort_labels=True,
    sort_clusters=False,
    annotate_pct_threshold=10,
    annotate_count_threshold=10,
    vmax=60,
):
    plot_df = df.copy()

    if map_label_fn is not None:
        plot_df[label_col] = plot_df[label_col].apply(map_label_fn)

    if drop_noise:
        plot_df = plot_df[plot_df[cluster_col] != noise_label].copy()

    if min_total_label_count > 0:
        total_counts = plot_df[label_col].value_counts()
        rare_labels = total_counts[total_counts < min_total_label_count].index
        plot_df[label_col] = plot_df[label_col].where(
            ~plot_df[label_col].isin(rare_labels),
            "other"
        )

    ct_counts = pd.crosstab(plot_df[cluster_col], plot_df[label_col])
    ct_pct = pd.crosstab(
        plot_df[cluster_col],
        plot_df[label_col],
        normalize="index"
    ) * 100

    if sort_labels:
        label_order = ct_pct.max(axis=0).sort_values(ascending=False).index
        ct_counts = ct_counts[label_order]
        ct_pct = ct_pct[label_order]

    if sort_clusters:
        cluster_order = ct_counts.sum(axis=1).sort_values(ascending=False).index
        ct_counts = ct_counts.loc[cluster_order]
        ct_pct = ct_pct.loc[cluster_order]

    cluster_sizes = ct_counts.sum(axis=1)
    ct_pct_ext = ct_pct.copy()
    ct_counts_ext = ct_counts.copy()
    ct_pct_ext["TOTAL"] = 0
    ct_counts_ext["TOTAL"] = cluster_sizes

    pastel_bupu = LinearSegmentedColormap.from_list(
        "pastel_bupu",
        ["#f7fcfd", "#e0ecf4", "#bfd3e6", "#9ebcda", "#c994c7", "#ddcce6"]
    )

    fig_w = max(10, 0.58 * ct_pct_ext.shape[1])
    fig_h = max(4.5, 0.8 * ct_pct_ext.shape[0])

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        ct_pct_ext.values,
        aspect="auto",
        cmap=pastel_bupu,
        vmin=0,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("% inside cluster")

    ax.set_xticks(np.arange(ct_pct_ext.shape[1]))
    ax.set_xticklabels(ct_pct_ext.columns, rotation=45, ha="right", fontsize=9)

    ax.set_yticks(np.arange(ct_pct_ext.shape[0]))
    ax.set_yticklabels(ct_pct_ext.index, fontsize=9)

    ax.set_xlabel("Label")
    ax.set_ylabel("Cluster")
    ax.set_title(title)

    total_col_idx = ct_pct_ext.shape[1] - 1
    ax.axvline(total_col_idx - 0.5, color="black", linewidth=1.5)

    for i in range(ct_pct_ext.shape[0]):
        for j in range(ct_pct_ext.shape[1]):
            if ct_pct_ext.columns[j] == "TOTAL":
                n = int(ct_counts_ext.iloc[i, j])
                ax.text(
                    j, i, f"n={n}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold"
                )
            else:
                pct = ct_pct_ext.iloc[i, j]
                count = int(ct_counts_ext.iloc[i, j])

                if (pct >= annotate_pct_threshold) or (count >= annotate_count_threshold):
                    ax.text(
                        j, i,
                        f"{pct:.0f}%\n(n={count})",
                        ha="center", va="center",
                        fontsize=7
                    )
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return ct_counts, ct_pct

def plot_pca_clusters(
    scores_df,
    save_path=None,
    pc_x="PC1",
    pc_y="PC2",
    cluster_col="cluster",
    style_by=None,
    static_values=("static",),
    special_labels=(-1,),
    special_label_name_map=None,
    cmap_name="tab20",
    figsize=(10, 8),
    point_size=30,
    alpha=0.75,
    title=None,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = scores_df.copy()

    if cluster_col not in df.columns:
        raise ValueError(f"'{cluster_col}' not found in DataFrame")
    if pc_x not in df.columns or pc_y not in df.columns:
        raise ValueError(f"'{pc_x}' and/or '{pc_y}' not found in DataFrame")

    if style_by is not None and style_by not in df.columns:
        raise ValueError(f"'{style_by}' not found in DataFrame")

    unique_clusters = [c for c in pd.unique(df[cluster_col]) if pd.notna(c)]
    unique_clusters = sorted(unique_clusters, key=lambda x: str(x))

    special_labels = set(special_labels or [])
    regular_clusters = [c for c in unique_clusters if c not in special_labels]

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, max(len(regular_clusters), 1)))
    cluster_to_color = {cl: colors[i] for i, cl in enumerate(regular_clusters)}

    for cl in unique_clusters:
        if cl in special_labels:
            cluster_to_color[cl] = (0.75, 0.75, 0.75, 0.6)

    if special_label_name_map is None:
        special_label_name_map = {}

    marker_list = ["o", "X", "s", "D", "^", "v", "P", "*"]
    marker_map = None
    style_labels = None
    use_static_style = False

    if style_by is not None:
        # If this column should be treated as static/non-static
        # then collapse to boolean styling
        if len(static_values) > 0:
            static_mask = df[style_by].isin(static_values)

            # Only treat as static-style if column is not already a general multi-class style
            # You can remove this condition if you always want static_values to apply.
            if static_mask.any():
                df["_style_group"] = static_mask
                marker_map = {
                    True: "o",
                    False: "X",
                }
                style_labels = [True, False]
                use_static_style = True
            else:
                style_labels = sorted(df[style_by].dropna().unique(), key=lambda x: str(x))
                marker_map = {
                    lab: marker_list[i % len(marker_list)] for i, lab in enumerate(style_labels)
                }
        else:
            style_labels = sorted(df[style_by].dropna().unique(), key=lambda x: str(x))
            marker_map = {
                lab: marker_list[i % len(marker_list)] for i, lab in enumerate(style_labels)
            }

    fig, ax = plt.subplots(figsize=figsize)

    for cl in unique_clusters:
        sub = df[df[cluster_col] == cl]

        if style_by is not None:
            if use_static_style:
                for style_lab in style_labels:
                    subset = sub[sub["_style_group"] == style_lab]
                    if not subset.empty:
                        ax.scatter(
                            subset[pc_x],
                            subset[pc_y],
                            color=cluster_to_color[cl],
                            marker=marker_map[style_lab],
                            s=point_size,
                            alpha=alpha,
                            label=None,
                        )
            else:
                for style_lab in style_labels:
                    subset = sub[sub[style_by] == style_lab]
                    if not subset.empty:
                        ax.scatter(
                            subset[pc_x],
                            subset[pc_y],
                            color=cluster_to_color[cl],
                            marker=marker_map[style_lab],
                            s=point_size,
                            alpha=alpha,
                            label=None,
                        )
        else:
            ax.scatter(
                sub[pc_x],
                sub[pc_y],
                color=cluster_to_color[cl],
                s=point_size,
                alpha=alpha,
                label=None,
            )

    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cluster_to_color[cl],
            markersize=8,
            label=special_label_name_map.get(cl, str(cl)),
        )
        for cl in unique_clusters
    ]

    cluster_legend = ax.legend(
        handles=cluster_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title=cluster_col,
    )
    ax.add_artist(cluster_legend)

    if style_by is not None:
        if use_static_style:
            marker_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[True],
                    color="black",
                    linestyle="None",
                    markersize=8,
                    label="Static",
                ),
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[False],
                    color="black",
                    linestyle="None",
                    markersize=8,
                    label="Non-static",
                ),
            ]
        else:
            marker_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_map[lab],
                    color="black",
                    linestyle="None",
                    markersize=8,
                    label=str(lab),
                )
                for lab in style_labels
            ]

        ax.legend(
            handles=marker_handles,
            bbox_to_anchor=(1.02, 0.55),
            loc="upper left",
            title=style_by,
        )

    ax.set_title(title or f"PCA colored by {cluster_col}")
    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    plt.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
