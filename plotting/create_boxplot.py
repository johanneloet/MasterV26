import os
import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def build_participant_df(
    csv_path="./results/loocv_summary_results_sensor_ablation.csv"
):
    import ast
    import pandas as pd

    df = pd.read_csv(csv_path)

    rows = []

    for _, row in df.iterrows():
        participant_metrics = row["participant_metrics"]

        if pd.isna(participant_metrics):
            continue

        if isinstance(participant_metrics, str):
            participant_metrics = ast.literal_eval(participant_metrics)

        for p in participant_metrics:

            row_dict = {
                # Common fields
                "classifier": row.get("classifier", None),
                "segmentation": row.get("segmentation", None),
                "taxonomy": row.get("taxonomy", None),

                # Scenario fields
                "dataset_scenario": row.get("dataset_scenario", None),
                "sensor_scenario": row.get("sensor_scenario", None),
                "DC_id": row.get("DC_id", None),

                # Participant info
                "participant": p["participant"],
                "leave_out_group": p.get(
                    "leave_out_group",
                    p["participant"]
                ),

                # Metrics
                "accuracy": p.get("accuracy", None),
                "f1": p.get("f1", None),
                "precision": p.get("precision", None),
                "recall": p.get("recall", None),
                "n_test_samples": p.get("n_test_samples", None),
            }

            # Backwards compatibility:
            # if dataset_scenario missing but DC_id exists
            if (
                row_dict["dataset_scenario"] is None
                and row_dict["DC_id"] is not None
            ):
                row_dict["dataset_scenario"] = row_dict["DC_id"]

            rows.append(row_dict)

    return pd.DataFrame(rows)

import re

def map_participant_for_plot(pid: str) -> str:
    """
    Mapping:

    test_1-20           -> P01-P20 legacy
    prelim_2-8          -> P21-P27 protocol
    aksoprotocol_1-5    -> P28-P32 protocol
    aksowork_1-5        -> P28-P32 work
    """

    match = re.match(
        r"^(test|prelim|aksoprotocol|aksowork)_(\d+)$",
        str(pid),
    )

    if not match:
        return str(pid)

    group, num = match.groups()
    num = int(num)

    # Legacy cohort
    if group == "test":
        return f"P{num:02d}-Legacy"

    # prelim_2 -> P21
    # prelim_8 -> P27
    if group == "prelim":
        mapped_num = 19 + num
        return f"P{mapped_num:02d}-Protocol"

    # aksoprotocol_1 -> P28
    # aksoprotocol_5 -> P32
    if group == "aksoprotocol":
        mapped_num = 27 + num
        return f"P{mapped_num:02d}-Protocol"

    # aksowork_1 -> P28
    # aksowork_5 -> P32
    if group == "aksowork":
        mapped_num = 27 + num
        return f"P{mapped_num:02d}-AKSO Real Work"

    return str(pid)

def plot_sensor_ablation_boxplots_by_dc(
    csv_path="./results/loocv_summary_results_sensor_ablation.csv",
    output_dir="./results/sensor_ablation_boxplots",
    metric="f1",
):
    os.makedirs(output_dir, exist_ok=True)

    perf = build_participant_df(csv_path)
    perf["participant"] = perf["participant"].apply(
    map_participant_for_plot
)

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
        #plt.title(f"{dc}: Sensor ablation performance by classifier")

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
            fontsize=10,
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
        
        
def plot_nn_sensor_ablation_dc4_dc6_dc7(
    csv_path="./results/loocv_summary_results_sensor_ablation.csv",
    output_dir="./results/sensor_ablation_boxplots",
    metric="f1",
):
    os.makedirs(output_dir, exist_ok=True)

    perf = build_participant_df(csv_path)

    perf["participant_plot"] = perf["participant"].apply(
        map_participant_for_plot
    )

    sensor_order = ["SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8", "SC9"]
    dc_order = ["DC4", "DC6", "DC7"]

    plot_df = perf[
        (perf["classifier"] == "NN")
        & (perf["dataset_scenario"].isin(dc_order))
    ].copy()

    plot_df["plot_group"] = (
        plot_df["dataset_scenario"] + "_" + plot_df["sensor_scenario"]
    )

    plot_order = []
    for dc in dc_order:
        available_sc = [
            sc for sc in sensor_order
            if (
                (plot_df["dataset_scenario"] == dc)
                & (plot_df["sensor_scenario"] == sc)
            ).any()
        ]

        for sc in available_sc:
            plot_order.append(f"{dc}_{sc}")

    def participant_legend_sort(label):
        match = re.match(r"P(\d+)-(Legacy|Protocol|AKSO Real Work)", str(label))

        if not match:
            return (999, 999)

        participant_num = int(match.group(1))
        stage = match.group(2)

        stage_order = {
            "Legacy": 0,
            "Protocol": 1,
            "AKSO Real Work": 2,
        }

        return (
        stage_order[stage],
        participant_num,
    )

    mean_legend = mlines.Line2D(
        [],
        [],
        color="grey",
        marker="^",
        markersize=6,
        linestyle="None",
        label="Mean",
    )

    plt.figure(figsize=(13, 7))

    sns.boxplot(
        data=plot_df,
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
        data=plot_df,
        x="plot_group",
        y=metric,
        order=plot_order,
        hue="participant_plot",
        jitter=0.25,
        size=4,
        dodge=False,
        alpha=0.8,
    )

    ax = plt.gca()

    ax.set_xticklabels(
        [label.split("_", 1)[1] for label in plot_order],
        rotation=45,
        ha="right",
    )

    plt.xlabel("Sensor scenario")
    plt.ylabel(metric.upper() if metric != "f1" else "F1 score")
    plt.ylim(0, 1.02)

    dc_counts = [
        sum(label.startswith(f"{dc}_") for label in plot_order)
        for dc in dc_order
    ]

    cumulative = 0
    for count in dc_counts[:-1]:
        cumulative += count
        plt.axvline(
            x=cumulative - 0.5,
            color="gray",
            linestyle="--",
            linewidth=1,
        )
    dc_to_experiment = {
        'DC4':'LOOCV11',
        'DC6': 'LOOCV13',
        'DC7':'LOOCV14'
    }

    start = 0
    for dc, count in zip(dc_order, dc_counts):
        if count > 0:
            ax.text(
                start + (count - 1) / 2,
                -0.22,
                f"{dc_to_experiment[dc]} (NN)",
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=12,
                fontweight="bold",
            )
        start += count

    handles, labels = ax.get_legend_handles_labels()

    unique = dict(zip(labels, handles))
    unique.pop("", None)

    sorted_items = sorted(
        unique.items(),
        key=lambda x: participant_legend_sort(x[0]),
    )

    sorted_labels = [x[0] for x in sorted_items]
    sorted_handles = [x[1] for x in sorted_items]

    handles = [mean_legend] + sorted_handles
    labels = ["Mean"] + sorted_labels

    plt.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.33),
        ncol=6,
        fontsize=10,
        markerscale=0.8,
        labelspacing=0.4,
        handletextpad=0.3,
        handlelength=1.0,
        frameon=True,
        title="Participant",
        title_fontsize=8,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    save_path = os.path.join(
        output_dir,
        f"DC4_DC6_DC7_SVC_sensor_ablation_{metric}_boxplot.pdf",
    )

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_cross_domain_generalization_boxplot(
    csv_path="./results/loocv_summary_train_proto_test_real_window3.5.csv",
    output_dir="./results/cross_domain_generalization_boxplots",
    metric="f1",
):
    os.makedirs(output_dir, exist_ok=True)

    perf = build_participant_df(csv_path).copy()
    
    perf["participant"] = perf["participant"].apply(map_participant_for_plot)

    dc_name_map = {
        "DC4": "CDG1",
        "DC6": "CDG2",
        "DC7": "CDG3",
    }

    perf["training_dataset"] = perf["DC_id"].map(dc_name_map)
    perf = perf.dropna(subset=["training_dataset", "classifier", metric])

    experiment_order = ["CDG1", "CDG2", "CDG3"]
    classifier_order = ["NN", "SVC", "RFC"]

    mean_legend = mlines.Line2D(
        [], [],
        color="grey",
        marker="^",
        markersize=6,
        linestyle="None",
        label="Mean",
    )

    # Same structure as sensor ablation:
    # NN_CDG1, NN_CDG2, ..., SVC_CDG1, ..., RFC_CDG1, ...
    perf["plot_group"] = (
        perf["classifier"] + "_" + perf["training_dataset"]
    )
    
    

    plot_order = []
    separator_xs = []

    for clf in classifier_order:
        available_exp = [
            exp for exp in experiment_order
            if ((perf["classifier"] == clf) & (perf["training_dataset"] == exp)).any()
        ]

        for exp in available_exp:
            plot_order.append(f"{clf}_{exp}")

        if len(plot_order) > 0:
            separator_xs.append(len(plot_order) - 0.5)

    # Remove last separator
    separator_xs = separator_xs[:-1]

    plt.figure(figsize=(13, 5))

    sns.boxplot(
        data=perf,
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
        data=perf,
        x="plot_group",
        y=metric,
        order=plot_order,
        hue="participant",
        jitter=0.25,
        size=4,
        dodge=False,
        alpha=0.8,
    )

    for separator_x in separator_xs:
        plt.axvline(
            x=separator_x,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    ax = plt.gca()

    ax.set_xticklabels(
        [label.split("_", 1)[1] for label in plot_order],
        rotation=45,
        ha="right",
    )

    plt.xlabel("Training dataset scenario")
    plt.ylabel(metric.upper() if metric != "f1" else "F1 score")
    #plt.title("Cross-domain generalization to real-world movement data")
    plt.ylim(0, 1.02)

    # Add classifier group labels under x-axis
    start_idx = 0

    for clf in classifier_order:
        clf_count = sum(label.startswith(f"{clf}_") for label in plot_order)

        if clf_count > 0:
            ax.text(
                start_idx + (clf_count - 1) / 2,
                -0.22,
                clf,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=12,
                fontweight="bold",
            )

            start_idx += clf_count

    handles, labels = ax.get_legend_handles_labels()
    handles = [mean_legend] + handles
    labels = ["Mean"] + labels

    plt.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=10,
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
        f"cross_domain_generalization_{metric}_boxplot_mixed_datasets.pdf",
    )

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    
def write_cross_domain_generalization_latex_table(
    csv_path="./results/loocv_summary_train_proto_test_real_window3.5.csv",
    output_path="./results/cross_domain_generalization_table.tex",
):
    import os
    import pandas as pd

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(csv_path)

    dc_name_map = {
        "DC4": "CDG1",
        "DC6": "CDG2",
        "DC7": "CDG3",
    }

    df["Experiment"] = df["DC_id"].map(dc_name_map)

    classifier_order = ["NN", "SVC", "RFC"]
    experiment_order = ["CDG1", "CDG2", "CDG3"]

    rows = []

    for exp in experiment_order:
        for clf in classifier_order:

            row = df[
                (df["Experiment"] == exp)
                & (df["classifier"] == clf)
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            rows.append({
                "Experiment": exp,
                "Classifier": clf,
                "Accuracy": (
                    f"{row['mean_accuracy']:.3f} "
                    f"$\\pm$ "
                    f"{row['std_accuracy']:.3f}"
                ),
                "F1": (
                    f"{row['mean_f1']:.3f} "
                    f"$\\pm$ "
                    f"{row['std_f1']:.3f}"
                ),
                "Precision": (
                    f"{row['mean_precision']:.3f} "
                    f"$\\pm$ "
                    f"{row['std_precision']:.3f}"
                ),
                "Recall": (
                    f"{row['mean_recall']:.3f} "
                    f"$\\pm$ "
                    f"{row['std_recall']:.3f}"
                ),
            })

    table_df = pd.DataFrame(rows)

    latex = table_df.to_latex(
        index=False,
        escape=False,
        column_format="llcccc",
        caption=(
            "Cross-domain generalization performance from protocol-based "
            "training datasets to real-world movement data."
        ),
        label="tab:cross_domain_generalization_results",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # plot_sensor_ablation_boxplots_by_dc(
    #     csv_path="./results/loocv_summary_results_sensor_ablation_final_final.csv",
    #     metric="f1",
    # )
    plot_nn_sensor_ablation_dc4_dc6_dc7(csv_path="./results/loocv_summary_results_sensor_ablation_final_final.csv")
    #plot_cross_domain_generalization_boxplot(csv_path=r"C:\Users\Bruker\MasterV26\results\mixed_dataset_experiment.csv")
    #write_cross_domain_generalization_latex_table(csv_path=r"C:\Users\Bruker\MasterV26\results\mixed_dataset_experiment.csv", output_path=r"./results/mixed_training_aksowork_experiment.tex")