import pandas as pd

def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"

def best_rows_to_latex(
    csv_path,
    classifiers=("NN", "SVC", "RFC"),
    metric_for_best="mean_f1",
    scenario_col="dataset_scenario",
    output_path=None,
):
    df = pd.read_csv(csv_path)

    # map DC1 -> KFOLD1, DC2 -> KFOLD2, etc. assumes that only the finest taxonomy has been run
    df["Scenario"] = df[scenario_col].str.replace("DC", "KFOLD", regex=False)

    # Optional: map segmentation names to SEG labels
    seg_map = {
        "Window2.5": "Window 2.5 / SEG1",
        "Window3.5": "Window 3.5 / SEG2",
        "Window5": "Window 5 / SEG3",
        "Repetition3.5": "Repetition 3.5 / SEG4",
    }
    df["Optimal segmentation"] = df["segmentation"].replace(seg_map)

    # Keep only selected classifiers
    df = df[df["classifier"].isin(classifiers)].copy()

    # Pick best segmentation per scenario + classifier
    idx = df.groupby(["Scenario", "classifier"])[metric_for_best].idxmax()
    best = df.loc[idx].sort_values(["Scenario", "classifier"])

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\linewidth}{!}{")
    lines.append(r"\begin{tabular}{l l l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Scenario & Classifier & Optimal segmentation & Accuracy & Precision & Recall & F1-score \\")
    lines.append(r"\midrule")

    for scenario, group in best.groupby("Scenario", sort=True):
        first = True

        for _, row in group.iterrows():
            scenario_text = scenario if first else ""

            line = (
                f"{scenario_text} & "
                f"{row['classifier']} & "
                f"{row['Optimal segmentation']} & "
                f"{fmt(row['mean_accuracy'], row['std_accuracy'])} & "
                f"{fmt(row['mean_precision'], row['std_precision'])} & "
                f"{fmt(row['mean_recall'], row['std_recall'])} & "
                f"{fmt(row['mean_f1'], row['std_f1'])} \\\\"
            )
            lines.append(line)
            first = False

        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex)

    return latex


if __name__ == "__main__":
    best_rows_to_latex(csv_path=r"C:\Users\Bruker\MasterV26\results\kfold_summary_results_NN_SVC_skip_prelim1.csv", output_path=r"C:\Users\Bruker\MasterV26\results\kfold_latex_table.tex")