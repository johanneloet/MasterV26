import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def build_sensor_ablation_participant_df(
    csv_path="./results/loocv_summary_results_sensor_ablation.csv"
):
    df = pd.read_csv(csv_path)

    rows = []

    for _, row in df.iterrows():
        participant_metrics = row["participant_metrics"]

        if pd.isna(participant_metrics):
            continue

        if isinstance(participant_metrics, str):
            participant_metrics = ast.literal_eval(participant_metrics)

        for p in participant_metrics:
            rows.append({
                "dataset_scenario": row["dataset_scenario"],
                "sensor_scenario": row["sensor_scenario"],
                "classifier": row["classifier"],
                "segmentation": row["segmentation"],
                "taxonomy": row["taxonomy"],
                "participant": p["participant"],
                "leave_out_group": p.get("leave_out_group", p["participant"]),
                "accuracy": p["accuracy"],
                "f1": p["f1"],
                "precision": p["precision"],
                "recall": p["recall"],
                "n_test_samples": p.get("n_test_samples", None),
            })

    return pd.DataFrame(rows)

def plot_sensor_ablation_boxplots_by_dc(
    csv_path="./results/loocv_summary_results_sensor_ablation.csv",
    output_dir="./results/sensor_ablation_boxplots",
    metric="f1",
):
    os.makedirs(output_dir, exist_ok=True)

    perf = build_sensor_ablation_participant_df(csv_path)

    sensor_order = ["SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8", "SC9"]
    classifier_order = ["NN", "SVC"]

    mean_legend = mlines.Line2D(
        [], [],
        color="grey",
        marker="^",
        markersize=6,
        linestyle="None",
        label="Mean",
    )

    for dc in sorted(perf["dataset_scenario"].unique()):
        dc_df = perf[perf["dataset_scenario"] == dc].copy()

        # Create combined plotting label: NN_SC1, NN_SC2, ..., SVC_SC1, ...
        dc_df["plot_group"] = (
            dc_df["classifier"] + "_" + dc_df["sensor_scenario"]
        )

        plot_order = []
        separator_x = None

        for clf in classifier_order:
            available_sc = [
                sc for sc in sensor_order
                if ((dc_df["classifier"] == clf) & (dc_df["sensor_scenario"] == sc)).any()
            ]

            for sc in available_sc:
                plot_order.append(f"{clf}_{sc}")

            if clf == "NN":
                separator_x = len(plot_order) - 0.5

        plt.figure(figsize=(13, 5))

        sns.boxplot(
            data=dc_df,
            x="plot_group",
            y=metric,
            order=plot_order,
            color="white",
            showfliers=False,
            width=0.35,
            showmeans=True,
            meanprops={
                "marker": "^",
                "markerfacecolor": "grey",
                "markeredgecolor": "grey",
                "markersize": 8,
            },
        )

        sns.stripplot(
            data=dc_df,
            x="plot_group",
            y=metric,
            order=plot_order,
            hue="participant",
            jitter=0.25,
            size=4,
            dodge=False,
            alpha=0.8,
        )

        if separator_x is not None:
            plt.axvline(
                x=separator_x,
                color="gray",
                linestyle="--",
                linewidth=1,
            )

        ax = plt.gca()

        # Cleaner x labels: show only SC labels, classifier shown as group text
        ax.set_xticklabels(
            [label.split("_", 1)[1] for label in plot_order],
            rotation=45,
            ha="right",
        )

        plt.xlabel("Sensor scenario")
        plt.ylabel(metric.upper() if metric != "f1" else "F1 score")
        plt.title(f"{dc}: Sensor ablation performance by classifier")

        plt.ylim(0, 1.02)

        # Add classifier group labels under x-axis
        nn_count = sum(label.startswith("NN_") for label in plot_order)
        svc_count = sum(label.startswith("SVC_") for label in plot_order)

        if nn_count > 0:
            ax.text(
                (nn_count - 1) / 2,
                -0.22,
                "NN",
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=12,
                fontweight="bold",
            )

        if svc_count > 0:
            ax.text(
                nn_count + (svc_count - 1) / 2,
                -0.22,
                "SVC",
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=12,
                fontweight="bold",
            )

        handles, labels = ax.get_legend_handles_labels()
        handles = [mean_legend] + handles
        labels = ["Mean"] + labels

        plt.legend(
            handles=handles,
            labels=labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            fontsize=8,
            markerscale=0.8,
            labelspacing=0.35,
            handletextpad=0.4,
            handlelength=1.2,
            frameon=True,
            title="Participant",
        )

        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"{dc}_sensor_ablation_{metric}_boxplot.pdf",
        )

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

def plot_sensor_ablation_summary_heatmaps(
    csv_path="./results/loocv_summary_results_sensor_ablation.csv",
    output_dir="./results/sensor_ablation_heatmaps",
):
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.colors as mcolors

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    dc_order = ["DC1", "DC2", "DC3", "DC4", "DC5", "DC6", "DC7"]
    sc_order = ["SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8", "SC9"]

    base_cmap = plt.cm.BuPu
    colors = base_cmap(np.linspace(0.0, 0.56, 256))
    cmap = mcolors.LinearSegmentedColormap.from_list("truncated_BuPu", colors)

    for clf in sorted(df["classifier"].unique()):
        clf_df = df[df["classifier"] == clf].copy()

        heatmap_values = clf_df.pivot(
            index="dataset_scenario",
            columns="sensor_scenario",
            values="mean_f1",
        )

        heatmap_std = clf_df.pivot(
            index="dataset_scenario",
            columns="sensor_scenario",
            values="std_f1",
        )

        heatmap_values = heatmap_values.reindex(index=dc_order, columns=sc_order)

        annot = heatmap_values.copy().astype(object)

        for row in annot.index:
            for col in annot.columns:
                val = annot.loc[row, col]

                if pd.isna(val):
                    annot.loc[row, col] = ""
                else:
                    annot.loc[row, col] = f"{val:.2f}"

        plt.figure(figsize=(9, 5))

        sns.heatmap(
            heatmap_values,
            annot=annot,
            fmt="",
            cmap=cmap,
            linewidths=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Mean F1"},
            annot_kws={"size": 12, "color": "black"},
        )

        plt.title(f"{clf}: Mean F1 by dataset and sensor scenario", fontsize=16)
        plt.xlabel("Sensor scenario", fontsize=13)
        plt.ylabel("Dataset scenario", fontsize=13)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11, rotation=0)

        plt.tight_layout()

        save_path = os.path.join(
            output_dir,
            f"{clf}_sensor_ablation_summary_heatmap.pdf",
        )

        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_sensor_ablation_boxplots_by_dc(
        csv_path="./results/loocv_summary_results_sensor_ablation.csv",
        metric="f1",
    )
    plot_sensor_ablation_summary_heatmaps()