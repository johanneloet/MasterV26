import matplotlib.pyplot as plt
import math
import numpy as np

from matplotlib.lines import Line2D


# Code developed during the preliminary PCA to study where the activities fall in different PC spaces.
# The code was developed with the aid of chatgpt


def plot_pca_scores(
    scores_df,
    pca=None,
    pc_x=1,
    pc_y=2,
    color_by="label",
    style_by=None,
    title=None,
    save_path=None,
    figsize=(12, 8),
    alpha=0.75,
):
    x_col = f"PC{pc_x}"
    y_col = f"PC{pc_y}"

    if x_col not in scores_df.columns or y_col not in scores_df.columns:
        raise ValueError(f"{x_col} or {y_col} not found in scores_df.")

    fig, ax = plt.subplots(figsize=figsize)

    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    labels = sorted(scores_df[color_by].dropna().unique())

    # pastel neon pink -> cream -> turquoise
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_pastel",
         ["#ff8fc1", "#ffe680", "#63d8d1"]
    )

    colors = custom_cmap(
        np.linspace(0, 1, len(labels))
    )

    color_map = {
        lab: colors[i]
        for i, lab in enumerate(labels)
    }

    if style_by is not None:
        styles = sorted(scores_df[style_by].dropna().unique())
        markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
        marker_map = {
            style: markers[i % len(markers)]
            for i, style in enumerate(styles)
        }

        for label in labels:
            for style in styles:
                subset = scores_df[
                    (scores_df[color_by] == label) &
                    (scores_df[style_by] == style)
                ]

                if subset.empty:
                    continue

                ax.scatter(
                    subset[x_col],
                    subset[y_col],
                    label=f"{label} | {style}",
                    alpha=alpha,
                    color=color_map[label],
                    marker=marker_map[style],
                    s=20,
                )
    else:
        for label in labels:
            subset = scores_df[scores_df[color_by] == label]

            ax.scatter(
                subset[x_col],
                subset[y_col],
                label=label,
                alpha=alpha,
                color=color_map[label],
                s=20,
            )

    if pca is not None:
        x_var = pca.explained_variance_ratio_[pc_x - 1] * 100
        y_var = pca.explained_variance_ratio_[pc_y - 1] * 100
        ax.set_xlabel(f"{x_col} ({x_var:.1f}% variance)", fontsize=16)
        ax.set_ylabel(f"{y_col} ({y_var:.1f}% variance)", fontsize=16)
    else:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    if title is None:
        title = f"PCA score plot colored by {color_by}"

   # ax.set_title(title)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.0, -0.21, 1.0, 0.1),
        ncol=4,
        title="Label",
        title_fontsize=15,
        fontsize=15,
        frameon=True,
        mode="expand",
        markerscale=2.5,
        handletextpad=0.1,
    )
    #plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# def plot_pca_scores(scores_df, pc_x=1, pc_y=2):
#     x_col = f"PC{pc_x}"
#     y_col = f"PC{pc_y}"

#     plt.figure(figsize=(8, 6))

#     labels = sorted(scores_df["label"].unique())
#     cmap = plt.get_cmap("tab20b", len(labels))
#     color_map = {lab: cmap(i) for i, lab in enumerate(labels)}

#     for label in labels:
#         subset = scores_df[scores_df["label"] == label]
#         plt.scatter(
#             subset[x_col], subset[y_col], label=label, alpha=0.7, color=color_map[label]
#         )

#     plt.xlabel(x_col)
#     plt.ylabel(y_col)
#     plt.title(f"PCA Score Plot ({x_col} vs {y_col})")
#     plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#     plt.tight_layout()
#     plt.show()


def plot_pca_subplots(
    scores_df,
    pairs=None,
    color_by="label",
    style_by=None,
    static_by=None,
    static_values=("static",),
    ncols=3,
    alpha=0.6,
    point_size=20,
):
    """
    Plot multiple PCA scatter subplots.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Must contain PC columns (PC1, PC2, ...) and grouping columns.
    pairs : list of tuples
        [(1,2), (1,3), ...] for PC combinations.
    color_by : str
        Column name used for color grouping.
    style_by : str or None
        Column name used for marker style. Optional.
    static_by : str or None
        Column name used to mark static vs non-static windows.
    static_values : tuple
        Values in static_by that should be treated as static.
    ncols : int
        Number of subplot columns.
    alpha : float
        Point transparency.
    point_size : int
        Scatter marker size.
    """

    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if pairs is None:
        pairs = [(1, i) for i in range(2, 8)]

    scores_df = scores_df.copy()

    nplots = len(pairs)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    fig.subplots_adjust(right=0.82)

    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    labels = sorted(scores_df[color_by].dropna().unique())

    # pastel neon pink -> cream -> turquoise
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_pastel",
        [
            "#f4a3c4",  # soft pink
            "#f3e6a3",  # pastel cream/yellow
            "#7fd3d0",  # turquoise
        ]
    )

    colors = custom_cmap(
        np.linspace(0, 1, len(labels))
    )

    color_map = {
        lab: colors[i]
        for i, lab in enumerate(labels)
    }

    # ---- MARKER SETUP ----
    marker_list = ["o", "X", "s", "D", "^", "v", "P", "*"]

    marker_map = None
    style_labels = None

    if style_by is not None and static_by is not None:
        raise ValueError("Use either style_by or static_by, not both at the same time.")

    if style_by:
        style_labels = sorted(scores_df[style_by].dropna().unique())
        marker_map = {
            lab: marker_list[i % len(marker_list)] for i, lab in enumerate(style_labels)
        }

    if static_by:
        scores_df["_static_state"] = scores_df[static_by].isin(static_values)
        marker_map = {
            True: "o",   # static
            False: "X",  # non-static
        }

        # optional sanity check
        print(scores_df[[static_by, "_static_state"]].drop_duplicates().sort_values(static_by))

    # ---- PLOTTING LOOP ----
    for idx, (pc_x, pc_y) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        x_col = f"PC{pc_x}"
        y_col = f"PC{pc_y}"

        for color_lab in color_labels:
            subset_color = scores_df[scores_df[color_by] == color_lab]

            if style_by:
                for style_lab in style_labels:
                    subset = subset_color[subset_color[style_by] == style_lab]

                    ax.scatter(
                        subset[x_col],
                        subset[y_col],
                        color=color_map[color_lab],
                        marker=marker_map[style_lab],
                        alpha=alpha,
                        s=point_size,
                        edgecolor="none",
                    )

            elif static_by:
                subset_static = subset_color[subset_color["_static_state"] == True]
                subset_nonstatic = subset_color[subset_color["_static_state"] == False]

                if not subset_static.empty:
                    ax.scatter(
                        subset_static[x_col],
                        subset_static[y_col],
                        color=color_map[color_lab],
                        marker=marker_map[True],
                        alpha=alpha,
                        s=point_size,
                        edgecolor="none",
                    )

                if not subset_nonstatic.empty:
                    ax.scatter(
                        subset_nonstatic[x_col],
                        subset_nonstatic[y_col],
                        color=color_map[color_lab],
                        marker=marker_map[False],
                        alpha=alpha,
                        s=point_size,
                        edgecolor="none",
                    )

            else:
                ax.scatter(
                    subset_color[x_col],
                    subset_color[y_col],
                    color=color_map[color_lab],
                    alpha=alpha,
                    s=point_size,
                    edgecolor="none",
                )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")

    # Hide unused subplots
    for idx in range(nplots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    # ---- LEGENDS ----
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[lab],
            markersize=8,
            label=str(lab),
        )
        for lab in color_labels
    ]

    fig.legend(handles=color_handles, loc="center right", title=color_by, ncol=1)

    if style_by:
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
        fig.legend(handles=marker_handles, loc="lower right", title=style_by, ncol=1)

    elif static_by:
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
        fig.legend(handles=marker_handles, loc="lower right", title=static_by, ncol=1)

    plt.show()

def plot_scree(pca, max_pcs=None):
    """
    Plot scree plot (explained variance and cumulative variance).

    Args:
        pca: fitted sklearn PCA object
        max_pcs (int or None): 
            number of PCs to show.
            If None → show all PCs.
    """

    explained_var = pca.explained_variance_ratio_

    # limit only if max_pcs given
    if max_pcs is not None:
        explained_var = explained_var[:max_pcs]

    pcs = np.arange(1, len(explained_var) + 1)
    cumulative_var = np.cumsum(explained_var)

    plt.figure(figsize=(10, 5))

    plt.bar(pcs, explained_var, alpha=0.7)
    plt.plot(pcs, cumulative_var, marker="o", color="black")

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.ylim(0, 1.05)

    # only show every nth tick if many PCs
    if len(pcs) > 30:
        step = len(pcs) // 15
        plt.xticks(pcs[::step])
    else:
        plt.xticks(pcs)

    plt.tight_layout()
    plt.show()
