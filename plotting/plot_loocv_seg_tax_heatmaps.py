import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
def select_optimal_taxonomy_cell(heatmap_values, dataset_scenario):
    """
    Selects optimal segmentation + taxonomy using detail-aware taxonomy rules.

    Rules:
    - DC3:
        Use T2 if best T2 F1 >= 0.60.
        Otherwise use best T1.
    - DC5, DC7:
        Use T2 if best T2 F1 >= 0.80.
        Otherwise use best T1.
    - Other scenarios:
        Use T3 if best T3 F1 >= 0.80.
        Else use T2 if best T2 F1 >= 0.80.
        Else use best T1.
    """

    def best_for_taxonomy(tax):
        if tax not in heatmap_values.columns:
            return None

        col = heatmap_values[tax].dropna()

        if col.empty:
            return None

        best_seg = col.idxmax()
        best_f1 = col.loc[best_seg]

        return best_seg, tax, best_f1

    if dataset_scenario == "DC3":
        candidate = best_for_taxonomy("T2")
        if candidate is not None and candidate[2] >= 0.60:
            return candidate

        return best_for_taxonomy("T1")

    if dataset_scenario in ["DC5", "DC7"]:
        candidate = best_for_taxonomy("T2")
        if candidate is not None and candidate[2] >= 0.80:
            return candidate

        return best_for_taxonomy("T1")

    candidate = best_for_taxonomy("T3")
    if candidate is not None and candidate[2] >= 0.80:
        return candidate

    candidate = best_for_taxonomy("T2")
    if candidate is not None and candidate[2] >= 0.80:
        return candidate

    return best_for_taxonomy("T1")


def plot_f1_heatmaps_by_classifier(
    csv_path="./results/loocv_summary_full_needs_redo.csv",
    output_dir="./results/heatmaps",
    metric_col="mean_f1",
    std_col="std_f1",
    classifiers=None,  # e.g. ["SVC"], ["NN"], or ["NN", "SVC"]
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    if classifiers is not None:
        df = df[df["classifier"].isin(classifiers)].copy()

    seg_order = ["Window2.5", "Window3.5", "Window5", "Repetition3.5"]
    tax_order = ["T1", "T2", "T3"]

    for classifier in sorted(df["classifier"].unique()):
        clf_df = df[df["classifier"] == classifier].copy()

        dataset_scenarios = sorted(clf_df["dataset_scenario"].unique())
        n_dc = len(dataset_scenarios)

        ncols = 3
        nrows = int(np.ceil(n_dc / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7 * ncols, 5 * nrows),
            squeeze=False,
        )
        seg_display_names = {
            "Window2.5": "SEG1",
            "Window3.5": "SEG2",
            "Window5": "SEG3",
            "Repetition3.5": "SEG4",
        }

        axes_flat = axes.flatten()

        for i, dataset_scenario in enumerate(dataset_scenarios):
            ax = axes_flat[i]

            plot_df = clf_df[
                clf_df["dataset_scenario"] == dataset_scenario
            ].copy()

            heatmap_values = plot_df.pivot(
                index="segmentation",
                columns="taxonomy",
                values=metric_col,
            )

            heatmap_std = plot_df.pivot(
                index="segmentation",
                columns="taxonomy",
                values=std_col,
            )

            heatmap_values = heatmap_values.reindex(
                [s for s in seg_order if s in heatmap_values.index]
            )
            heatmap_values = heatmap_values.reindex(
                columns=[t for t in tax_order if t in heatmap_values.columns]
            )

            heatmap_std = heatmap_std.reindex(index=heatmap_values.index)
            heatmap_std = heatmap_std.reindex(columns=heatmap_values.columns)


            heatmap_values.index = [
                seg_display_names.get(idx, idx)
                for idx in heatmap_values.index
            ]

            heatmap_std.index = [
                seg_display_names.get(idx, idx)
                for idx in heatmap_std.index
            ]

            annot = heatmap_values.copy().astype(str)
            for row in heatmap_values.index:
                for col in heatmap_values.columns:
                    mean_val = heatmap_values.loc[row, col]
                    std_val = heatmap_std.loc[row, col]

                    if pd.isna(mean_val):
                        annot.loc[row, col] = ""
                    else:
                        annot.loc[row, col] = f"{mean_val:.3f}\n± {std_val:.3f}"

            if not heatmap_values.isna().all().all():
                selected = select_optimal_taxonomy_cell(
                    heatmap_values,
                    dataset_scenario
                )

            if selected is not None:
                best_seg, best_tax, best_f1 = selected
                annot.loc[best_seg, best_tax] += "\n★"
                
                
                    
            base_cmap = plt.cm.BuPu
            colors = base_cmap(np.linspace(0.0, 0.56, 256))
            truncated_bupu = mcolors.LinearSegmentedColormap.from_list(
                "truncated_BuPu",
                colors
            )

            sns.heatmap(
                heatmap_values,
                annot=annot,
                fmt="",
                cmap=truncated_bupu,
                linewidths=0.5,
                # cbar=i == n_dc - 1,
                # cbar_kws={"label": "Mean F1"},
                cbar=False,
                vmin=0,
                vmax=1,
                ax=ax,
                annot_kws={"size": 14,  "color": "black",}
            )

            ax.set_title(dataset_scenario)
            ax.set_xlabel("Taxonomy")
            ax.set_ylabel("Segmentation")
            
            ax.set_title(dataset_scenario, fontsize=18)
            ax.set_xlabel("Taxonomy", fontsize=14)
            ax.set_ylabel("Segmentation", fontsize=14)

            ax.tick_params(axis="x", labelsize=12)
            ax.tick_params(axis="y", labelsize=12)
            
            if selected is not None:
                for text in ax.texts:
                    x, y = text.get_position()

                    col_idx = int(round(x - 0.5))
                    row_idx = int(round(y - 0.5))

                    row_name = heatmap_values.index[row_idx]
                    col_name = heatmap_values.columns[col_idx]

                    if row_name == best_seg and col_name == best_tax:
                        text.set_fontweight("bold")
                        text.set_fontsize(14)

        for j in range(n_dc, len(axes_flat)):
            axes_flat[j].axis("off")
        
        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=truncated_bupu, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=axes,
            fraction=0.02,
            pad=0.02,
        )

        cbar.set_label("Mean F1", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        fig.suptitle(f"{classifier} – F1 heatmaps by dataset scenario", fontsize=25)
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])

        save_path = os.path.join(
            output_dir,
            f"{classifier}_f1_heatmaps_by_dataset_scenario.pdf",
        )

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")