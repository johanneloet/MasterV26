import matplotlib.pyplot as plt
import math
import numpy as np

# Code developed during the preliminary PCA to study where the activities fall in different PC spaces.


def plot_pca_scores(scores_df, pc_x=1, pc_y=2):
    x_col = f"PC{pc_x}"
    y_col = f"PC{pc_y}"

    plt.figure(figsize=(8, 6))

    labels = sorted(scores_df["label"].unique())
    cmap = plt.get_cmap("tab20b", len(labels))
    color_map = {lab: cmap(i) for i, lab in enumerate(labels)}

    for label in labels:
        subset = scores_df[scores_df["label"] == label]
        plt.scatter(
            subset[x_col], subset[y_col], label=label, alpha=0.7, color=color_map[label]
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"PCA Score Plot ({x_col} vs {y_col})")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


# def plot_pca_subplots(scores_df, pairs=None, color_by="label",
#                       ncols=3, alpha=0.6):
#     if pairs is None:
#         pairs = [(1, i) for i in range(2, 8)]

#     nplots = len(pairs)
#     nrows = math.ceil(nplots / ncols)

#     fig, axes = plt.subplots(
#         nrows,
#         ncols,
#         figsize=(5* ncols, 4 * nrows),  # extra width for legend
#         squeeze=False,
#         constrained_layout=True
#     )
#     fig.subplots_adjust(right=0.83)

#     labels = sorted(scores_df[color_by].unique())

#     # Use large distinct palette
#     cmap = (
#         list(plt.cm.tab20.colors)
#         + list(plt.cm.tab20b.colors)
#         + list(plt.cm.tab20c.colors)
#     )
#     color_map = {lab: cmap[i % len(cmap)] for i, lab in enumerate(labels)}

#     for idx, (pc_x, pc_y) in enumerate(pairs):
#         r, c = divmod(idx, ncols)
#         ax = axes[r][c]

#         x_col = f"PC{pc_x}"
#         y_col = f"PC{pc_y}"

#         for lab in labels:
#             sub = scores_df[scores_df[color_by] == lab]
#             ax.scatter(
#                 sub[x_col],
#                 sub[y_col],
#                 alpha=alpha,
#                 color=color_map[lab],
#                 s=8,
#                 edgecolor="none"
#             )

#         ax.set_xlabel(x_col)
#         ax.set_ylabel(y_col)
#         ax.set_title(f"{x_col} vs {y_col}")

#     # Create shared legend manually
#     handles = [
#         plt.Line2D([0], [0], marker='o', color='w',
#                    markerfacecolor=color_map[lab],
#                    markersize=8, label=lab)
#         for lab in labels
#     ]

#     fig.legend(
#         handles=handles,
#         loc="center right",           # anchor the legend to the left-center
#         #bbox_to_anchor=(1.02, 0.5),  # place it just outside the figure
#         ncol=1
#     )

#     plt.show()


import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_pca_subplots(
    scores_df,
    pairs=None,
    color_by="label",
    style_by=None,  # e.g. "prefix"
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
        Column name used for color grouping (e.g., movement label).
    style_by : str or None
        Column name used for marker style (e.g., prefix). Optional.
    ncols : int
        Number of subplot columns.
    alpha : float
        Point transparency.
    point_size : int
        Scatter marker size.
    """

    if pairs is None:
        pairs = [(1, i) for i in range(2, 8)]

    nplots = len(pairs)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    # Leave space for legends
    fig.subplots_adjust(right=0.82)

    # ---- COLOR SETUP ----
    color_labels = sorted(scores_df[color_by].unique())

    cmap = (
        list(plt.cm.tab20.colors)
        + list(plt.cm.tab20b.colors)
        + list(plt.cm.tab20c.colors)
    )

    color_map = {lab: cmap[i % len(cmap)] for i, lab in enumerate(color_labels)}

    # ---- MARKER SETUP (optional) ----
    if style_by:
        style_labels = sorted(scores_df[style_by].unique())
        marker_list = ["o", "X", "s", "D", "^", "v", "P", "*"]
        marker_map = {
            lab: marker_list[i % len(marker_list)] for i, lab in enumerate(style_labels)
        }

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

    # Color legend (movement)
    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[lab],
            markersize=8,
            label=lab,
        )
        for lab in color_labels
    ]

    fig.legend(handles=color_handles, loc="center right", title=color_by, ncol=1)

    # Marker legend (prefix)
    if style_by:
        marker_handles = [
            Line2D(
                [0],
                [0],
                marker=marker_map[lab],
                color="black",
                linestyle="None",
                markersize=8,
                label=lab,
            )
            for lab in style_labels
        ]

        fig.legend(handles=marker_handles, loc="lower right", title=style_by, ncol=1)

    plt.show()


def plot_scree(pca, max_pcs=None):
    """
    Plot scree plot (explained variance and cumulative variance).

    Args:
        pca: fitted sklearn PCA object
        max_pcs (int, optional): limit number of PCs shown
    """
    explained_var = pca.explained_variance_ratio_

    if max_pcs is not None:
        explained_var = explained_var[:max_pcs]

    pcs = np.arange(1, len(explained_var) + 1)
    cumulative_var = np.cumsum(explained_var)

    plt.figure(figsize=(8, 5))

    plt.bar(pcs, explained_var, alpha=0.7, label="Individual explained variance")
    plt.plot(
        pcs,
        cumulative_var,
        marker="o",
        color="black",
        label="Cumulative explained variance",
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.xticks(pcs)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()
