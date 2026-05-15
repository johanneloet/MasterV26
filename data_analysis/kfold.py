import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
from matplotlib.patches import Rectangle
from collections import defaultdict

from feature_extraction.get_paths import (
    get_feture_paths,
    get_feature_paths_for_multiple_spaces,
    get_test_folder_paths,
)
from data_analysis.run_SVC import run_SVC, run_SVC_with_feature_tuning
from data_analysis.run_NN import run_NN, run_NN_with_feature_tuning
import json
from data_analysis.cf_matrix import make_confusion_matrix
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from utils import map_taxonomy_candidate_3, map_taxonomy_candidate_4, drop_label


def run_stratified_kfold_with_pca(
    clf_name: str = "SVC",
    class_version: int = 1,
    prefixes: list[str] = ["test", "prelim", "aksowork", "aksoprotocol"],
    right_arm: bool = True,
    left_arm: bool = False,
    lower_back: bool = True,
    upper_back: bool = False,
    left_fsr: bool = True,
    right_fsr: bool = True,
    expanded_fsr: bool = False,
    taxonomy_fn=None,
    seg_strategy="Window3.5",
    n_splits: int = 5,
    random_state: int = 42,
):
    test_folder_dict = get_test_folder_paths()

    sensor_flags = {
        "right_arm": right_arm,
        "left_arm": left_arm,
        "lower_back": lower_back,
        "upper_back": upper_back,
        "left_fsr": left_fsr,
        "right_fsr": right_fsr,
    }

    sensor_order = [
        "right_arm",
        "left_arm",
        "lower_back",
        "upper_back",
        "left_fsr",
        "right_fsr",
    ]

    sensor_config = [s for s in sensor_order if sensor_flags[s]]
    sensor_combo_scenario = "_".join(sensor_config)

    feature_files = {}
    seg_mode = "Window"
    if "Repetition" in seg_strategy:
        seg_mode = "Repetition"

    for test_id, folder_path in test_folder_dict.items():
        if "aksowork" in test_id:
            seg_mode = "Window"
            seg_strategy = "Window3.5" #quickfix
        if test_id.split("_")[0] in prefixes:
            filename = (
                f"Features_{seg_mode}_{test_id}_expanded{expanded_fsr}_SEG{seg_strategy}_{sensor_combo_scenario}.csv"
            )
            feature_files[test_id] = Path(folder_path) / filename
    

    dfs = []

    for test_id, path in feature_files.items():
        df = pd.read_csv(path)

        df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")

        print(len(df.columns))
        df["source_test_id"] = test_id
        
        print(df["label"])
        print("TEST ID", test_id)
        if taxonomy_fn is not None:
             df['label'] = df.apply(
            lambda row: taxonomy_fn(row["label"], row.get("static_label", None)),
            axis=1
            )
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    full_df = full_df[full_df["label"].notna()].copy()
    full_df = full_df[full_df["label"] != "other"].copy()

    Y = full_df["label"]
    X = full_df.drop(
        columns=["label", "static_label", "source_test_id"],
        errors="ignore",
    )

    labels = sorted(Y.unique())

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    all_accuracies = []
    all_f1 = []
    all_precision = []
    all_recall = []
    all_Y_true = []
    all_Y_pred = []

    start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y), start=1):
        print(f"\nFold {fold_idx}/{n_splits}...")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        Y_train = Y.iloc[train_idx]
        Y_test = Y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        print(f"PCA components: {pca.n_components_}")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        CV_suffix = (
            f"fold{fold_idx}_{sensor_combo_scenario}"
            f"_expanded{expanded_fsr}_class{class_version}"
        )

        fold_labels = sorted(Y_train.unique())

        if clf_name == "SVC":
            test_results, train_results = run_SVC(
                X_train_pca,
                Y_train,
                X_test_pca,
                Y_test,
                class_names=fold_labels,
                CV_suffix=CV_suffix,
                opt=True,
            )

        elif clf_name == "NN":
            test_results, train_results, *_ = run_NN(
                X_train_pca,
                Y_train,
                X_test_pca,
                Y_test,
                class_names=fold_labels,
                CV_suffix=CV_suffix,
                opt=True,
            )

        else:
            raise ValueError("Invalid classifier name.")

        Y_pred = test_results[0]
        accuracy = test_results[1]
        f1 = test_results[2]
        precision = test_results[3]
        recall = test_results[4]

        all_accuracies.append(accuracy)
        all_f1.append(f1)
        all_precision.append(precision)
        all_recall.append(recall)
        all_Y_true.extend(Y_test)
        all_Y_pred.extend(Y_pred)

        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")

    end = time.time()
    print(f"\nDone! Total time: {end - start:.2f} sec")

    print(f"\nMean Accuracy:  {np.mean(all_accuracies):.3f}")
    print(f"Mean F1:        {np.mean(all_f1):.3f}")
    print(f"Mean Precision: {np.mean(all_precision):.3f}")
    print(f"Mean Recall:    {np.mean(all_recall):.3f}")

    save_path = "./plots_use"
    os.makedirs(save_path, exist_ok=True)

    cm = confusion_matrix(all_Y_true, all_Y_pred, labels=labels)

    run_name = (
        f"{clf_name}_stratifiedK{n_splits}_{sensor_combo_scenario}"
        f"_expanded{expanded_fsr}_class{class_version}"
    )

    make_confusion_matrix(
        cf=cm,
        categories=labels,
        title=f"{clf_name} Stratified {n_splits}-Fold Confusion Matrix",
        savepath=f"./plots_use/{run_name}.pdf",
        color_code={},
    )

    summary = {
    "mean_accuracy": np.mean(all_accuracies),
    "std_accuracy": np.std(all_accuracies, ddof=1),
    "mean_f1": np.mean(all_f1),
    "std_f1": np.std(all_f1, ddof=1),
    "mean_precision": np.mean(all_precision),
    "std_precision": np.std(all_precision, ddof=1),
    "mean_recall": np.mean(all_recall),
    "std_recall": np.std(all_recall, ddof=1),
    }

    return summary

if __name__ == "__main__":
    dataset_scenarios = {
    "DC1": ["test"],                         # Legacy
    "DC2": ["aksoprotocol", "prelim"],                   # Protocol
    "DC3": ["aksowork"],                       # Real-world
    "DC4": ["aksoprotocol", "aksowork", "prelim"],
    "DC5": ["prelim", "aksoprotocol", "test"],
    "DC6": ["test", "aksowork"],
    "DC7": ["prelim", "aksoprotocol", "aksowork", "test"],
}
    
    results = []

    for DC_id, DC in dataset_scenarios.items():
        if DC_id == "DC3":
            seg_scenarios = ["Window2.5", 
                            "Window3.5", "Window5"
                             ]
        else:
            seg_scenarios = [
                "Window2.5", "Window3.5", "Window5", 
                             "Repetition3.5"]
        
        # Evaluate protocol only datasets with their original labels
        if DC_id in ["DC1", "DC2", "DC5"]:
            tax_fn = None
            taxonomy_id = "T3"
        else:
            # otherwsie use T2
            tax_fn = map_taxonomy_candidate_3
            taxonomy_id = "T2"
        
        # configure such that the fullest available sensor combination scenario is used
        if "test" in DC:
            left_arm = False
            upper_back = False
            SC = "SC2"
        else:
            left_arm=True
            upper_back=True
            SC = "SC1"
        
        for seg in seg_scenarios:
            for clf in [
                "NN", 
                "SVC"
                        ]:
                summary = run_stratified_kfold_with_pca(
                    prefixes=DC,
                    left_arm=left_arm,
                    upper_back=upper_back,
                    clf_name=clf,
                    expanded_fsr=True,
                    taxonomy_fn=tax_fn,
                    seg_strategy=seg,
                    n_splits=5,
                )

                summary.update({
                    "dataset_scenario": DC_id,
                    "prefixes": "+".join(DC),
                    "taxonomy": taxonomy_id,
                    "segmentation": seg,
                    "classifier": clf,
                    "expanded_fsr": True,
                    "sensor_scenario": SC,
                })

                results.append(summary)

    results_df = pd.DataFrame(results)
    os.makedirs("./results", exist_ok=True)
    results_df.to_csv("./results/kfold_summary_results_NN_SVC_skip_prelim1.csv", index=False)
    print(results_df)
            
            