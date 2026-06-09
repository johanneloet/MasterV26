import pandas as pd
import numpy as np

# this script was created with the aid of chatgpt

df = pd.read_csv("./results/loocv_summary_results_sensor_ablation_final_final.csv")

dc_order = ["DC1", "DC2", "DC3", "DC4", "DC5", "DC6", "DC7"]
sc_order = ["SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8", "SC9"]

def fmt_cell(mean, std):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"

for clf in sorted(df["classifier"].unique()):

    clf_df = df[df["classifier"] == clf].copy()

    table = pd.DataFrame(index=dc_order, columns=sc_order)

    for _, row in clf_df.iterrows():
        dc = row["dataset_scenario"]
        sc = row["sensor_scenario"]

        table.loc[dc, sc] = fmt_cell(
            row["mean_f1"],
            row["std_f1"]
        )

    table = table.fillna("--")

    latex = table.to_latex(
        escape=False,
        column_format="l" + "c" * len(sc_order),
        caption=f"Mean F1-score $\\pm$ standard deviation for {clf} across dataset and sensor configurations.",
        label=f"tab:f1_sensor_ablation_{clf.lower()}",
    )

    print("\n" + "=" * 80)
    print(f"LATEX TABLE FOR {clf}")
    print("=" * 80)
    print(latex)