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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plot_df = df.copy()

    # optional label mapping
    if map_label_fn is not None:
        plot_df[label_col] = plot_df[label_col].apply(map_label_fn)

    # optional noise removal (useful for DBSCAN/HDBSCAN)
    if drop_noise:
        plot_df = plot_df[plot_df[cluster_col] != noise_label].copy()

    # optional rare-label collapsing
    if min_total_label_count > 0:
        total_counts = plot_df[label_col].value_counts()
        rare_labels = total_counts[total_counts < min_total_label_count].index
        plot_df[label_col] = plot_df[label_col].where(
            ~plot_df[label_col].isin(rare_labels),
            "other"
        )

    # build tables
    ct_counts = pd.crosstab(plot_df[cluster_col], plot_df[label_col])
    ct_pct = pd.crosstab(
        plot_df[cluster_col],
        plot_df[label_col],
        normalize="index"
    ) * 100

    # optional label sorting
    if sort_labels:
        label_order = ct_pct.max(axis=0).sort_values(ascending=False).index
        ct_counts = ct_counts[label_order]
        ct_pct = ct_pct[label_order]

    # optional cluster sorting by size
    if sort_clusters:
        cluster_order = ct_counts.sum(axis=1).sort_values(ascending=False).index
        ct_counts = ct_counts.loc[cluster_order]
        ct_pct = ct_pct.loc[cluster_order]

    # add total column
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

    # separator before TOTAL column
    total_col_idx = ct_pct_ext.shape[1] - 1
    ax.axvline(total_col_idx - 0.5, color="black", linewidth=1.5)

    # annotations
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