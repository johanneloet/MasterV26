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
from data_analysis.run_RFC  import run_RFC
import json
from data_analysis.cf_matrix import make_confusion_matrix
from pathlib import Path

from data_analysis.loocv import select_sensor_columns
from utils import map_taxonomy_candidate_1, map_taxonomy_candidate_2, map_taxonomy_candidate_3, map_taxonomy_candidate_4

# Run  this for DC4, DC5 and DC7 with the fullest sensor combination per dataset

def loocv_pca_train_proto_test_real(
    clf_name: str = "SVC",
    label_mapping: dict | None = None,
    class_version: int = 1,
    prefixes: list[str] = ["test","prelim", "aksowork", "aksoprotocol"],
    right_arm: bool = True,
    left_arm: bool = False,
    lower_back: bool = True,
    upper_back: bool = False,
    left_fsr: bool = True,
    right_fsr: bool = True,
    expanded_fsr: bool = False,
    taxonomy_fn=None,
    seg_strategy="Window3.5",
    save_per_participant_metrics: bool = False,
):
    
    # modified version of run_loocv_with_pca that aims to answer RQ5 !
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
        current_seg_mode = seg_mode
        current_seg_strategy = seg_strategy
        
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

        if "aksowork" in test_id and "Repetition" in current_seg_strategy:
            current_seg_mode = "Window"
            current_seg_strategy = "Window3.5"  # quickfix
        if test_id.split("_")[0] in prefixes:
            filename = (
                f"Features_{current_seg_mode}_{test_id}_expanded{expanded_fsr}_SEG{current_seg_strategy}_{full_sensor_combo_scenario}.csv"
            )
            feature_files[test_id] = Path(folder_path) / filename

    test_ids = list(feature_files.keys())

    all_accuracies = []
    all_f1 = []
    all_precision = []
    all_recall = []
    all_Y_true = []
    all_Y_pred = []
    
    all_participant_metrics = []

    start = time.time()

    for leave_out in test_ids:
        if leave_out.split('_')[0] in ["test", "prelim", "aksoprotocol"]:
            continue #  do not use protocol files for test files
        print(f"\nTesting on {leave_out}...")
        
        # leave out all aksowork files
        leave_out_tests = []
        num_aksowork_files = 5
        for i in range(num_aksowork_files):
            leave_out_tests.append(f'aksowork_{i+1}')
        leave_out_tests.append(f'aksoprotocol_{leave_out.split("_")[1]}') # finally, leave out protocol data corresponding to current test participant
        print("Leaving out", leave_out_tests)
        # Split train/test
        train_dfs = [
            pd.read_csv(path)
            for test_id, path in feature_files.items()
            if test_id not in leave_out_tests
        ]
        test_df = pd.read_csv(feature_files[leave_out])
        
        # SELECT RELEVANT FEATURE COMBOS HERE!!
        train_dfs = [
            select_sensor_columns(pd.read_csv(path), sensor_config)
            for test_id, path in feature_files.items()
            if test_id not in leave_out_tests
        ]

        test_df = select_sensor_columns(
            pd.read_csv(feature_files[leave_out]),
            sensor_config
        )

        train_df = pd.concat(train_dfs, ignore_index=True)

        # Drop unnamed columns columns 
        cols_to_drop_train = [
            c for c in train_df.columns
            if "Unnamed" in c
        ]
        cols_to_drop_test = [
            c for c in test_df.columns

            if "Unnamed" in c
        ]

        train_df = train_df.drop(columns=cols_to_drop_train)
        test_df = test_df.drop(columns=cols_to_drop_test)

        train_df["label_used"] = train_df["label"]
        test_df["label_used"] = test_df["label"]

        if taxonomy_fn is not None:
            train_df["label_used"] = train_df.apply(
                lambda row: taxonomy_fn(row["label"], row.get("static_label", None)),
                axis=1
            )

            test_df["label_used"] = test_df.apply(
                lambda row: taxonomy_fn(row["label"], row.get("static_label", None)),
                axis=1
            )

        train_df = train_df[train_df["label_used"].notna()].copy()
        test_df = test_df[test_df["label_used"].notna()].copy()

        # drop unusual labels
        train_df = train_df[train_df["label_used"] != "other"].copy()
        test_df = test_df[test_df["label_used"] != "other"].copy()
    

        
        Y_train = train_df["label_used"]
        X_train = train_df.drop(
            columns=["label", "label_used", "static_label"],
            errors="ignore"
        )
        
        #print("X_train columns:", X_train.columns.tolist())

        # Use only test labels seen in training
        test_df = test_df[test_df["label_used"].isin(Y_train.unique())].copy()

        Y_test = test_df["label_used"]
        X_test = test_df.drop(
            columns=["label", "label_used", "static_label"],
            errors="ignore"
        )
        
        
        
        print(Y_test.value_counts())
        
        # -------------------------
        # Sanity checks
        # -------------------------
        assert len(X_train) == len(Y_train), f"Train mismatch: {len(X_train)} vs {len(Y_train)}"
        assert len(X_test) == len(Y_test), f"Test mismatch: {len(X_test)} vs {len(Y_test)}"
    
        # OLD LABEL MAPPING CODE
        # if label_mapping is not None:
        #     Y_train = Y_train.map(label_mapping)
        #     Y_test = Y_test.map(label_mapping)
        #     labels = Y_train.unique()
        # else:
        #     labels = Y_train.unique()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        print(f"PCA components: {pca.n_components_}")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        CV_suffix = f"{leave_out}_{sensor_combo_scenario}_expanded{expanded_fsr}_class{class_version}"

        labels = sorted(Y_train.unique())

        if clf_name == "SVC":
            test_results, train_results = run_SVC(
                X_train_pca,
                Y_train,
                X_test_pca,
                Y_test,
                class_names=labels,
                CV_suffix=CV_suffix,
                opt=True,
            )
        elif clf_name == "NN":
            test_results, train_results, *_ = run_NN(
                X_train_pca,
                Y_train,
                X_test_pca,
                Y_test,
                class_names=labels,
                CV_suffix=CV_suffix,
                opt=True,
            )
        elif clf_name == "RFC":
            test_results, train_results, *_ = run_RFC(
                X_train_pca,
                Y_train,
                X_test_pca,
                Y_test,
                class_names=labels,
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
        
        if save_per_participant_metrics:
            all_participant_metrics.append({
                "participant": leave_out,
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            })

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

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    prefix_string = "_".join(prefixes)

    run_name = f"{clf_name}_window3.5SEG_TRAINPROTOTESTREAL_{prefix_string}"

    make_confusion_matrix(
        cf=cm,
        categories=labels,
        #title=f"{clf_name} Confusion Matrix",
        savepath=f"./plots_use/{run_name}.pdf",
        color_code={},
    )

    #return all_accuracies
    
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

    if save_per_participant_metrics:
        summary["participant_metrics"] = all_participant_metrics

    return summary

if __name__ == "__main__":
    dataset_scenarios = {
    "DC4": ["aksoprotocol", "aksowork", "prelim"],
    "DC6": ["test", "aksowork"],
    "DC7": ["prelim", "aksoprotocol", "aksowork", "test"],
    }
    # optimal_tax_and_seg_dict = {
    #     "DC4" : {
    #         "T_opt" : map_taxonomy_candidate_4,
    #         "seg_opt" : "Repetition3.5"
    #     },
    #     "DC6" : {
    #         "T_opt" : map_taxonomy_candidate_4,
    #         "seg_opt" : "Repetition3.5"
    #     },
    #     "DC7" : {
    #         "T_opt" : map_taxonomy_candidate_4,
    #         "seg_opt" : "Repetition3.5"
    #     },
    #     }
    
    optimal_tax_and_seg_dict = {
        "DC4" : {
            "T_opt" : map_taxonomy_candidate_4,
            "seg_opt" : "Window3.5"
        },
        "DC6" : {
            "T_opt" : map_taxonomy_candidate_4,
            "seg_opt" : "Window3.5"
        },
        "DC7" : {
            "T_opt" : map_taxonomy_candidate_4,
            "seg_opt" : "Window3.5"
        },
        }
    
    sensor_combination_scenarios = {
            "SC1" : {
                "right_arm" : True,
                "left_arm" : True,
                "lower_back" : True,
                "upper_back" : True,
                "left_fsr" : True,
                "right_fsr" : True
            },
            "SC2" : {
                "right_arm" : True,
                "left_arm" : False,
                "lower_back" : True,
                "upper_back" : False,
                "left_fsr" : True,
                "right_fsr" : True
            },
        }
    
    classifiers = [
        "NN",
        "SVC",
        "RFC"
    ]
    results  = []
    for DC_id, DC in dataset_scenarios.items():
        if DC_id == "DC4":
            SC_id = "SC1"
        else:
            SC_id = "SC2"
        
        SC = sensor_combination_scenarios[SC_id]
        
        T_seg_opt = optimal_tax_and_seg_dict[DC_id]
        
        for clf in classifiers:
            summary = loocv_pca_train_proto_test_real(
                clf_name=clf,
                prefixes=DC,
                right_arm=SC["right_arm"],
                left_arm=SC["left_arm"],
                upper_back=SC["upper_back"],
                lower_back=SC["lower_back"],
                left_fsr=SC["left_fsr"],
                right_fsr=SC["right_fsr"],
                expanded_fsr=True,
                taxonomy_fn=map_taxonomy_candidate_3,
                seg_strategy=T_seg_opt["seg_opt"],
                save_per_participant_metrics=True # true so we can perhaps make a boxplot for this too
            )
            summary["DC_id"] = DC_id
            summary["classifier"] = clf
            results.append(summary)
    results_df = pd.DataFrame(results)
    os.makedirs("./results", exist_ok=True)
    results_df.to_csv("./results/loocv_summary_train_proto_test_real_window3.5.csv", index=False)
    print(results_df)
        
            
        
        
        
        