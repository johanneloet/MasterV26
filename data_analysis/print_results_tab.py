import pandas as pd

# ---- LOAD DATA ----
df = pd.read_csv("./results/kfold_summary_results_NN_SVC_RFC.csv")

# ---- FIND BEST SEGMENTATION PER (dataset, classifier) ----
best_rows = (
    df.sort_values("mean_f1", ascending=False)
      .groupby(["dataset_scenario", "classifier"], as_index=False)
      .first()
)

# ---- FORMAT FUNCTION ----
def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"

# ---- BUILD TABLE ----
rows = []

for dc in sorted(best_rows["dataset_scenario"].unique()):
    subset = best_rows[best_rows["dataset_scenario"] == dc]

    for _, row in subset.iterrows():
        rows.append({
            "Dataset": dc,
            "Classifier": row["classifier"],
            "Segmentation": row["segmentation"],
            "Accuracy": fmt(row["mean_accuracy"], row["std_accuracy"]),
            "Precision": fmt(row["mean_precision"], row["std_precision"]),
            "Recall": fmt(row["mean_recall"], row["std_recall"]),
            "F1": fmt(row["mean_f1"], row["std_f1"]),
        })

table_df = pd.DataFrame(rows)

# ---- PRINT NICELY ----
print("\n=== BEST RESULTS TABLE ===\n")
print(table_df.to_string(index=False))