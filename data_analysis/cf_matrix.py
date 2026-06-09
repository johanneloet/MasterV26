import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# This file is inspired by plotting code from Roya
# some adjustaments to the plots was aided by chatgpt

def make_confusion_matrix(
    cf,
    categories,
    figsize=(35, 35),
    cmap="Blues",
    title=None,
    fontsize=25,
    savepath="confusion_matrix_colored.pdf",
    color_code=True,
):

    cf = np.asarray(cf)
    n = cf.shape[0]

    row_sums = cf.sum(axis=1)  # predicted totals (rows)
    col_sums = cf.sum(axis=0)  # true totals (columns)
    total = cf.sum()
    diag = np.diag(cf)

    precision = np.divide(
        diag, row_sums, out=np.zeros_like(row_sums, float), where=row_sums != 0
    )
    recall = np.divide(
        diag, col_sums, out=np.zeros_like(col_sums, float), where=col_sums != 0
    )
    accuracy = np.trace(cf) / total if total > 0 else 0

    cf_percent = np.divide(
    cf,
    row_sums[:, np.newaxis],
    out=np.zeros_like(cf, dtype=float),
    where=row_sums[:, np.newaxis] != 0,
) * 100
    
    # initialize full matrix
    color_data = np.zeros((n + 1, n + 1))

    # put normalized confusion matrix in main block
    color_data[:-1, :-1] = cf_percent


    # set ext row/column to constant background value
    constant_bg = 15  # light blue in "Blues" colormap

    color_data[:-1, -1] = constant_bg
    color_data[-1, :-1] = constant_bg
    color_data[-1, -1] = constant_bg

    # ---- EXTENDED GRID ----
    # ext = np.zeros((n + 1, n + 1), float)
    # ext[:-1, :-1] = cf_percent
    # ext[:-1, -1] = row_sums  # Precision column
    # ext[-1, :-1] = col_sums  # Recall row
    # ext[-1, -1] = total

    # ---- LABELS FOR HEATMAP (only main n×n block) ----
    labels = np.full_like(color_data, "", dtype=object)  # start with empty strings

    # main block counts
    for i in range(n):
        for j in range(n):
            labels[i, j] = f"{cf[i, j]:.0f}"

    # we leave last row/column labels as "" so seaborn doesn't draw counts there

    base_cmap = plt.cm.Blues

    truncated_blues = LinearSegmentedColormap.from_list(
        "truncated_blues",
        base_cmap(np.linspace(0, 0.40, 256))
    )

    # ---- PLOTTING ----
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        color_data,
        annot=labels,
        fmt="",
        cmap=truncated_blues,
        cbar=False,
        xticklabels=list(categories) + ["Precision"],
        yticklabels=list(categories) + ["Recall"],
        annot_kws={"fontsize": fontsize+6, "fontweight": "bold"},
    )

    # ---- ADD ROW-WISE PERCENTAGES (per predicted class / row) ----
    for i in range(n):
        for j in range(n):
            count = cf[i, j]
            row_total = row_sums[i]
            pct = (count / row_total) * 100 if row_total > 0 else 0

            ax.text(
                j + 0.5,
                i + 0.80,  # lower in cell so it does not overlap count
                f"{pct:.2f}%",
                color="black",
                ha="center",
                va="center",
                fontsize=fontsize - 2,
                # fontweight='bold'
            )

    # ---- PRECISION COLUMN (row-based) ----
    # for i in range(n):
    #     p  = precision[i] * 100
    #     fp = (1 - precision[i]) * 100

    #     x = n + 0.5
    #     # three lines: total (black), correct (green), error (red)
    #     ax.text(x, i + 0.35,
    #             f"{row_sums[i]:.0f}",
    #             color="black", ha='center', va='center',
    #             fontsize=fontsize-1)
    #     ax.text(x, i + 0.50,
    #             f"{p:.2f}%",
    #             color="green", ha='center', va='center',
    #             fontsize=fontsize-1)
    #     ax.text(x, i + 0.65,
    #             f"{fp:.2f}%",
    #             color="red", ha='center', va='center',
    #             fontsize=fontsize-1)
    for i in range(n):
        p = precision[i] * 100
        fp = (1 - precision[i]) * 100

        x = n + 0.5
        ax.text(
            x,
            i + 0.2,
            f"{row_sums[i]:.0f}",
            color="black",
            ha="center",
            va="center",
            fontsize=fontsize - 4,
            fontweight="bold",
        )  # total
        ax.text(
            x,
            i + 0.55,
            f"{p:.2f}%",
            color="green",
            ha="center",
            va="center",
            fontsize=fontsize - 4,
            fontweight="bold",
        )  # correct%
        ax.text(
            x,
            i + 0.80,
            f"{fp:.2f}%",
            color="red",
            ha="center",
            va="center",
            fontsize=fontsize - 4,
            fontweight="bold",
        )  # wrong%

    for j in range(n):
        r = recall[j] * 100
        fn = (1 - recall[j]) * 100

        x = j + 0.5
        ax.text(
            x,
            n + 0.2,
            f"{col_sums[j]:.0f}",
            color="black",
            ha="center",
            va="center",
            fontsize=fontsize - 4,
            fontweight="bold",
        )
        ax.text(
            x,
            n + 0.3,
            f"\n{r:.2f}%",
            color="green",
            ha="center",
            va="center",
            fontsize=fontsize - 4,
            fontweight="bold",
        )
        ax.text(
            x,
            n + 0.45,
            f"\n\n{fn:.2f}%",
            color="red",
            ha="center",
            va="center",
            fontsize=fontsize - 4,
            fontweight="bold",
        )

    x = n + 0.5
    y = n + 0.5
    if color_code:
        # Create row and column colors based on keywords
        categories = ["arm", "lean"]
        row_colors = []
        col_colors = []
        for label in categories:
            l = label.lower()
            # default color
            color = "black"
            # check if any keyword matches
        for kw, kw_color in color_code.items():
            if kw in l:
                color = kw_color
                break
        row_colors.append(color)
        col_colors.append(color)

        # Apply colors
        for tick_label, color in zip(ax.get_yticklabels()[:-1], row_colors):
            tick_label.set_color(color)
        for tick_label, color in zip(ax.get_xticklabels()[:-1], col_colors):
            tick_label.set_color(color)

    ax.text(
        x,
        y - 0.3,
        f"{total:.0f}",
        color="Black",
        ha="center",
        va="center",
        fontsize=fontsize - 4,
        fontweight="bold",
    )
    ax.text(
        x,
        y,
        f"{accuracy*100:.2f}%",
        color="green",
        ha="center",
        va="center",
        fontsize=fontsize - 4,
        fontweight="bold",
    )
    ax.text(
        x,
        y + 0.3,
        f"{(1-accuracy)*100:.2f}%",
        color="red",
        ha="center",
        va="center",
        fontsize=fontsize - 4,
        fontweight="bold",
    )

    # labels
    ax.set_xlabel("True", fontsize=fontsize)
    ax.set_ylabel("Predicted", fontsize=fontsize)

    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    plt.xticks(rotation=45, ha="right", fontsize=fontsize)
    plt.yticks(rotation=45, ha="right", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(savepath, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
