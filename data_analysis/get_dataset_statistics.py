# Prints LaTeX tables with:
# 1) total feature-window counts per DC / segmentation / taxonomy
# 2) per-class counts per taxonomy

import os
from pathlib import Path

import pandas as pd

from utils import (
    map_taxonomy_candidate_3,
    map_taxonomy_candidate_4,
)
from feature_extraction.get_paths import get_test_folder_paths


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def build_feature_files(prefixes, expanded_fsr, seg_strategy):
    test_folder_dict = get_test_folder_paths()
    feature_files = {}

    seg_mode = "Repetition" if "Repetition" in seg_strategy else "Window"

    for test_id, folder_path in test_folder_dict.items():

        if test_id.split("_")[0] not in prefixes:
            continue

        current_seg_mode = seg_mode
        current_seg_strategy = seg_strategy

        if "aksowork" in test_id and "Repetition" in current_seg_strategy:
            current_seg_mode = "Window"
            current_seg_strategy = "Window3.5"

        if "test" in test_id:
            full_sensor_config = [
                "right_arm",
                "lower_back",
                "left_fsr",
                "right_fsr",
            ]
        else:
            full_sensor_config = [
                "right_arm",
                "left_arm",
                "lower_back",
                "upper_back",
                "left_fsr",
                "right_fsr",
            ]

        full_sensor_combo_scenario = "_".join(full_sensor_config)

        filename = (
            f"Features_{current_seg_mode}_{test_id}_expanded{expanded_fsr}"
            f"_SEG{current_seg_strategy}_{full_sensor_combo_scenario}.csv"
        )

        feature_files[test_id] = Path(folder_path) / filename

    return feature_files


def apply_taxonomy(df, taxonomy_fn):
    df = df.copy()

    if taxonomy_fn is None:
        #df["label_used"] = df["label"]
        df["label_used"] = df["label"].replace({
        "neutral_load_left": "neutral_load",
        "neutral_load_right": "neutral_load",
        })
    else:
        df["label_used"] = df.apply(
            lambda row: taxonomy_fn(row["label"], row.get("static_label", None)),
            axis=1,
        )

    df = df[df["label_used"].notna()].copy()
    df = df[
        df["label_used"].astype(str).str.lower().str.strip() != "other"
    ].copy()

    return df


# --------------------------------------------------
# Config
# --------------------------------------------------

dataset_scenarios = {
    "DC1": ["test"],
    "DC2": ["aksoprotocol", "prelim"],
    "DC3": ["aksowork"],
    "DC4": ["aksoprotocol", "aksowork", "prelim"],
    "DC5": ["prelim", "aksoprotocol", "test"],
    "DC6": ["test", "aksowork"],
    "DC7": ["prelim", "aksoprotocol", "aksowork", "test"],
}

taxonomy_scenarios = {
    "T1": map_taxonomy_candidate_4,
    "T2": map_taxonomy_candidate_3,
    "T3": None,
}

segmentation_scenarios = [
    "Window2.5",
    "Window3.5",
    "Window5",
    "Repetition3.5",
]

dc_order = ["DC1", "DC2", "DC3", "DC4", "DC5", "DC6", "DC7"]
seg_order = ["Window2.5","Window3.5", "Window5","Repetition3.5"]

expanded_fsr = True

os.makedirs("./results", exist_ok=True)


# --------------------------------------------------
# Collect per-label sample counts
# --------------------------------------------------

rows = []

for DC_id, prefixes in dataset_scenarios.items():

    for seg_strategy in segmentation_scenarios:

        feature_files = build_feature_files(
            prefixes=prefixes,
            expanded_fsr=expanded_fsr,
            seg_strategy=seg_strategy,
        )

        raw_dfs = []

        for test_id, path in feature_files.items():

            if not path.exists():
                print(f"Missing file, skipping: {path}")
                continue

            df = pd.read_csv(path)
            df["test_id"] = test_id
            raw_dfs.append(df)

        if not raw_dfs:
            print(f"No files found for {DC_id}, {seg_strategy}")
            continue

        raw_combined_df = pd.concat(raw_dfs, ignore_index=True)

        for T_id, taxonomy_fn in taxonomy_scenarios.items():

            if T_id == "T3" and DC_id not in ["DC1", "DC2", "DC5"]:
                continue

            labeled_df = apply_taxonomy(raw_combined_df, taxonomy_fn)

            label_counts = labeled_df["label_used"].value_counts().sort_index()
            total_samples = int(label_counts.sum())

            print(f"{DC_id} | {seg_strategy} | {T_id}: {total_samples} samples")

            for label, count in label_counts.items():
                rows.append({
                    "dataset_scenario": DC_id,
                    "segmentation": seg_strategy,
                    "taxonomy": T_id,
                    "label": label,
                    "count": int(count),
                    "total_samples": total_samples,
                })

stats_df = pd.DataFrame(rows)
stats_df.to_csv("./results/class_distribution_stats.csv", index=False)


# --------------------------------------------------
# Table 1: Total samples per DC / segmentation / taxonomy
# --------------------------------------------------

summary_df = (
    stats_df[
        ["dataset_scenario", "segmentation", "taxonomy", "total_samples"]
    ]
    .drop_duplicates()
)

pivot_df = summary_df.pivot_table(
    index=["dataset_scenario", "segmentation"],
    columns="taxonomy",
    values="total_samples",
    aggfunc="first",
).reset_index()

pivot_df.columns.name = None

pivot_df = pivot_df.rename(columns={
    "dataset_scenario": "DC",
    "segmentation": "Segmentation",
    "T1": "T1 samples",
    "T2": "T2 samples",
    "T3": "T3 samples",
})

pivot_df["DC"] = pd.Categorical(pivot_df["DC"], categories=dc_order, ordered=True)
pivot_df["Segmentation"] = pd.Categorical(
    pivot_df["Segmentation"],
    categories=seg_order,
    ordered=True,
)

pivot_df = pivot_df.sort_values(["DC", "Segmentation"])

for col in ["T1 samples", "T2 samples", "T3 samples"]:
    if col in pivot_df.columns:
        pivot_df[col] = pivot_df[col].apply(
            lambda x: "--" if pd.isna(x) else f"{int(x):,}"
        )

pivot_df.to_csv(
    "./results/feature_window_counts_by_dc_seg_taxonomy.csv",
    index=False,
)

latex_total = pivot_df.to_latex(
    index=False,
    escape=True,
    caption=(
        "Number of resulting feature windows across dataset combinations, "
        "segmentation strategies, and taxonomies."
    ),
    label="tab:feature_window_counts",
)

print("\n" + "=" * 80)
print("TOTAL SAMPLE COUNT TABLE")
print("=" * 80)
print(latex_total)

with open(
    "./results/feature_window_counts_by_dc_seg_taxonomy.tex",
    "w",
    encoding="utf-8",
) as f:
    f.write(latex_total)


# --------------------------------------------------
# Tables 2-4: Per-class counts per taxonomy
# --------------------------------------------------

for T_id in ["T1", "T2", "T3"]:

    t_df = stats_df[stats_df["taxonomy"] == T_id].copy()

    if t_df.empty:
        continue

    pivot = t_df.pivot_table(
        index=["dataset_scenario", "segmentation"],
        columns="label",
        values="count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    pivot.columns.name = None

    pivot = pivot.rename(columns={
        "dataset_scenario": "DC",
        "segmentation": "Segmentation",
    })

    pivot["DC"] = pd.Categorical(pivot["DC"], categories=dc_order, ordered=True)
    pivot["Segmentation"] = pd.Categorical(
        pivot["Segmentation"],
        categories=seg_order,
        ordered=True,
    )

    pivot = pivot.sort_values(["DC", "Segmentation"])

    # convert numeric class columns to int
    for col in pivot.columns:
        if col not in ["DC", "Segmentation"]:
            pivot[col] = pivot[col].astype(int)

    pivot.to_csv(f"./results/per_class_counts_{T_id.lower()}.csv", index=False)

    latex = pivot.to_latex(
        index=False,
        escape=True,
        caption=f"Per-class feature window counts for taxonomy {T_id}.",
        label=f"tab:per_class_counts_{T_id.lower()}",
    )

    print("\n" + "=" * 80)
    print(f"PER-CLASS TABLE FOR {T_id}")
    print("=" * 80)
    print(latex)

    with open(
        f"./results/per_class_counts_{T_id.lower()}.tex",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(latex)

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

os.makedirs("./results/dataset_stats_plots", exist_ok=True)

selected_cases = [
    ("DC1", "T3"),
    ("DC2", "T3"),
    ("DC3", "T1"),
    ("DC3", "T2"),
    ("DC7", "T1"),
    ("DC7", "T2"),
]

seg_order = ["Window2.5", "Window3.5", "Window5", "Repetition3.5"]

for dc, tax in selected_cases:

    plot_df = stats_df[
        (stats_df["dataset_scenario"] == dc)
        & (stats_df["taxonomy"] == tax)
    ].copy()

    if plot_df.empty:
        print(f"Skipping {dc}-{tax}: no data")
        continue


    pivot_counts = plot_df.pivot_table(
        index="segmentation",
        columns="label",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )

    pivot_counts = pivot_counts.reindex(seg_order)
    pivot_counts = pivot_counts.dropna(how="all")


    pivot_pct = (
        pivot_counts.div(pivot_counts.sum(axis=1), axis=0)
        * 100
    )

    import matplotlib.colors as mcolors
    n_classes = len(pivot_pct.columns)
    
    base_cmap = plt.cm.Blues
    colors = base_cmap(np.linspace(0.0, 0.90, 256))
    truncated_bupu = mcolors.LinearSegmentedColormap.from_list(
        "truncated_BuPu",
        colors
    )

    cmap = truncated_bupu

    # avoid extremely light colors
    colors = [
        cmap(x)
        for x in np.linspace(0.35, 0.9, n_classes)
    ]
    
    seg_name_map = {
    "Window2.5": "SEG1",
    "Window3.5": "SEG2",
    "Window5": "SEG3",
    "Repetition3.5": "SEG4",
    }
    pivot_counts.index = [
    seg_name_map.get(x, x)
    for x in pivot_counts.index
    ]

    pivot_pct.index = [
        seg_name_map.get(x, x)
        for x in pivot_pct.index
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    pivot_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        width=0.8,
        alpha=0.5
    )

    #ax.set_title(f"Class distribution for {dc} using {tax}")
    ax.set_xlabel("Segmentation strategy")
    ax.set_ylabel("Class percentage (%)")

    #vertical legend next to plot
    ax.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=14
    )
    
    # horizontal legend below plot (use for the protocol plots at least)
    # ax.legend(
    # title="Class",
    # loc="upper center",
    # bbox_to_anchor=(0.5, -0.15),
    # ncol=3,   # adjust depending on number of classes
    # frameon=False
    # )

    plt.xticks(rotation=30, ha="right")

    # annotate bars with raw sample counts
    for bar_group_idx, seg_name in enumerate(pivot_counts.index):

        cumulative_height = 0

        for class_idx, label in enumerate(pivot_counts.columns):

            count = pivot_counts.loc[seg_name, label]
            pct = pivot_pct.loc[seg_name, label]

            if count == 0:
                continue

            y_center = cumulative_height + pct / 2

            ax.text(
                x=bar_group_idx,
                y=y_center,
                s=f"class samples: {int(count)}",
                ha="center",
                va="center",
                fontsize=11,
                color="black",
            )

            cumulative_height += pct
            
    
    totals = pivot_counts.sum(axis=1)

    for bar_idx, (seg_name, total_count) in enumerate(totals.items()):

        ax.text(
            x=bar_idx,
            y=100,  # top of bar
            s=f"total samples={int(total_count)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()

    save_path = (
        f"./results/dataset_stats_plots/"
        f"{dc}_{tax}_class_distribution.pdf"
    )

    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved {save_path}")