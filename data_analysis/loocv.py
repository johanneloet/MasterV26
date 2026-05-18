# Script to run Leave one subject out cross validation with a selected classifier. Based on code from Maria!

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

# def run_loocv_with_pca(label_mapping=None, clf_name = "", window_size=8, norm_IMU=True, mean_fsr=False, hdr=False, class_version=1):
#     feature_files = get_feture_paths(window_length_sec=window_size, norm_IMU=norm_IMU, mean_fsr=mean_fsr, hdr=hdr)
#     test_ids = list(feature_files.keys())


#     start = time.time()
#     for leave_out in test_ids:
#         print(f"\n Testing on {leave_out}...")

#         # Split training and test sets
#         train_dfs = [pd.read_csv(path) for test_id, path in feature_files.items() if test_id != leave_out]
#         test_df = pd.read_csv(feature_files[leave_out])

#         # Combine training sets
#         train_df = pd.concat(train_dfs, ignore_index=True)

#         # Separate features and labels
#         X_train = train_df.drop(columns=["label"])
#         Y_train = train_df["label"]

#         X_test = test_df.drop(columns=["label"])
#         Y_test = test_df["label"]

#         if label_mapping is not None:
#             Y_train = train_df["label"].map(label_mapping)
#             Y_test = test_df["label"].map(label_mapping)
#             if label_mapping == label_mapping_v2:
#                 labels = ["hands_up", "push_pull", "squatting", "lifting", "sit_stand", "walking"]
#             elif label_mapping == label_mapping_v3:
#                 labels = ["hands_up", "push_pull_lift", "squat", "sit", "stand_walk"]
#         else:
#             labels = ["hand_up_back", "hands_forward", "hands_up", "push", "pull", "squatting", "lifting", "sitting", "standing", "walking"]

#         # Scale data (SD of 1 and mean of 0)
#         scaler = StandardScaler().set_output(transform="pandas")
#         # Fit scaler on training data
#         scaler.fit(X_train)
#         # Transfor both train and test set with the scaler
#         X_train_scaled = scaler.transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         # Apply pca
#         pca = PCA(n_components=0.95)
#         pca_fit = pca.fit(X_train_scaled)
#         pca_components = pca.n_components_
#         print (f"pca components: {pca_components}")
#         print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()}")

#         X_train_pca = pca_fit.transform(X_train_scaled)
#         X_test_pca = pca_fit.transform(X_test_scaled)

#         if clf_name == "SVC":
#             test_results, train_results = run_SVC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
#         elif clf_name == "RFC":
#             test_results, train_results = run_RFC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
#         elif clf_name == "NN":
#             test_results, train_results, best_params, Y_test, Y_test_fit = run_NN(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
#         else:
#             print("Model name not possible, model set automatic to svc")
#             test_results, train_results = run_SVC(X_train_pca, Y_train, X_test_pca, Y_test, class_names=labels, CV_suffix=f"{leave_out}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}", opt=True, time_window=f"{window_size}_sec")
#         Y_test_fit = test_results[0]
#         accuracy_test = test_results[1]
#         f1_test = test_results[2]
#         precision_test  = test_results[3]
#         recall_test = test_results[4]

#         accuracy_train = train_results[1]
#         f1_test = test_results[2]
#         precision_test  = test_results[3]
#         recall_test = test_results[4]


#         # Evaluate
#         print(f"Accuracy for {leave_out}: {accuracy_test:.3f}")
#         print(f"Precision for {leave_out}: {precision_test:.3f}")
#         print(f"Recall for {leave_out}: {recall_test:.3f}")
#         print(f"F1 for {leave_out}: {f1_test:.3f}")
#         #print(f"Best hyperparameters {leave_out} : {best_params}")
#         #print(f"Accuracy for train {leave_out}: {accuracy_train:.3f}")
#         # all_accuracies.append(accuracy_test)
#         # all_f1.append(f1_test)
#         # all_precision.append(precision_test)
#         # all_recall.append(recall_test)

#         # all_Y_true.extend(Y_test)
#         # all_Y_pred.extend(Y_test_fit)
#         # all_hyperparameters.extend(best_params)

#         cm = confusion_matrix(Y_test, Y_test_fit, labels=labels)

#         print(f"\nConfusion matrix for {leave_out}:")
#         print(pd.DataFrame(cm, index=labels, columns=labels))

#         # Optional: save or plot the heatmap
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#         disp.plot(cmap='BuPu', xticks_rotation='vertical')
#         plt.title(f"Confusion Matrix – {leave_out}")
#         plt.tight_layout()
#         plt.savefig(f"./plots_use/confmat_{leave_out}.png", dpi=300)
#         plt.close()

#     end = time.time()
#     elapsed = end - start
#     print(f"\n🕒 Done! Total time uesd: {elapsed:.2f} seconds")

#     print(f"\n✅ Mean LOOCV accuracy: {np.mean(all_accuracies):.3f}")
#     print(f"\n✅ Mean LOOCV f1: {np.mean(all_f1):.3f}")
#     print(f"\n✅ Mean LOOCV precision: {np.mean(all_precision):.3f}")
#     print(f"\n✅ Mean LOOCV recall: {np.mean(all_recall):.3f}")

#     # Confusion matrix
#     save_path="./plots_use"
#     os.makedirs(save_path, exist_ok=True)

#     cm = confusion_matrix(all_Y_true, all_Y_pred, labels=labels, normalize='true')
#     print("\n🧮 Confusion Matrix:")
#     print(cm)

#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#     disp.plot(xticks_rotation='vertical',cmap=plt.cm.BuPu, values_format=".2f")
#     for text in disp.ax_.texts:
#         text.set_fontsize(8)
#     plt.tight_layout()

#     save_file1 = os.path.join(save_path, f"{clf_name}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}_{window_size}_sec_CM1.png")
#     plt.savefig(save_file1, dpi=300)
#     plt.close()

#     cm2 = confusion_matrix(all_Y_true, all_Y_pred, labels=labels)
#     df_cm2 = pd.DataFrame(cm2, index=labels, columns=labels)

#     df_cm2['Total'] = df_cm2.sum(axis=1)
#     totals_row = df_cm2.sum(axis=0)
#     totals_row.name = 'Total'

#     df_cm2 = pd.concat([df_cm2, totals_row.to_frame().T])

#     n_rows, n_cols = df_cm2.shape

#     mask = np.zeros_like(df_cm2, dtype=bool)
#     mask[-1, :] = True   # last row (Total Pred)
#     mask[:, -1] = True   # last column (Total True)

#     if label_mapping is None:
#         plt.figure(figsize=(7, 6))
#     else:
#         plt.figure(figsize=(5.5, 5))
#     ax = sns.heatmap(df_cm2, annot=True, fmt='.0f', cmap='BuPu', mask=mask, cbar=True)

#     total_bg_color = '#e9ecff'  # matching 'BuPu'
#     for i in range(n_rows - 1):
#         ax.add_patch(Rectangle((n_cols - 1, i), 1, 1, fill=True, color=total_bg_color, lw=0))
#     for j in range(n_cols - 1):
#         ax.add_patch(Rectangle((j, n_rows - 1), 1, 1, fill=True, color=total_bg_color, lw=0))
#     ax.add_patch(Rectangle((n_cols - 1, n_rows - 1), 1, 1, fill=True, color=total_bg_color, lw=0))

#     # Manually annotate the total row and column
#     for i in range(n_rows - 1):  # all rows except last
#         val = df_cm2.iat[i, -1]
#         ax.text(n_cols - 0.5, i + 0.5, f'{val:.0f}', ha='center', va='center', color='black', fontsize=9)

#     for j in range(n_cols - 1):  # all columns except last
#         val = df_cm2.iat[-1, j]
#         ax.text(j + 0.5, n_rows - 0.5, f'{val:.0f}', ha='center', va='center', color='black', fontsize=9)

#     corner_val = df_cm2.iat[-1, -1]
#     ax.text(n_cols - 0.5, n_rows - 0.5, f'{corner_val:.0f}', ha='center', va='center', color='black', fontsize=9)

#     plt.title('Confusion Matrix with True and Predicted Totals')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')

#     save_file2 = os.path.join(save_path, f"{clf_name}_norm_{norm_IMU}_fsr{mean_fsr}_hdr{hdr}_class_{class_version}_{window_size}_sec_CM2.png")
#     plt.tight_layout()
#     plt.savefig(save_file2, dpi=300)
#     plt.close()

#     return all_accuracies
from utils import map_taxonomy_candidate_1, map_taxonomy_candidate_2, map_taxonomy_candidate_3, map_taxonomy_candidate_4



def select_sensor_columns(df, sensor_config):
    metadata_cols = [
        c for c in ["label", "label_used", "static_label"]
        if c in df.columns
    ]

    sensor_prefixes = {
        "right_arm": "R_Arm",
        "left_arm": "L_Arm",
        "lower_back": "Lower_Back",
        "upper_back": "Upper_Back",
        "left_fsr": "Left",
        "right_fsr": "Right",
    }

    selected_feature_cols = []

    for sensor in sensor_config:
        prefix = sensor_prefixes[sensor]
        selected_feature_cols.extend(
            [c for c in df.columns if c.endswith(prefix)]
        )

    keep_cols = metadata_cols + selected_feature_cols

    return df[keep_cols].copy()

def run_loocv_with_pca(
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
        # if "prelim" in leave_out or "aksoprotocol" in leave_out or "test" in leave_out:
        #     continue # dont test on protocol files.
        print(f"\nTesting on {leave_out}...")
        leave_out_tests = [leave_out]
        if leave_out.split('_')[0] == 'aksoprotocol':
            leave_out_tests.append(f'aksowork_{leave_out.split("_")[1]}')
            print("Leaving out", leave_out_tests)
        elif leave_out.split('_')[0] == 'aksowork':
            leave_out_tests.append(f'aksoprotocol_{leave_out.split("_")[1]}')
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
        else: 
            train_df["label"] = train_df["label"].replace({
            "neutral_load_left": "neutral_load",
            "neutral_load_right": "neutral_load",
            })
            test_df["label"] = test_df["label"].replace({
            "neutral_load_left": "neutral_load",
            "neutral_load_right": "neutral_load",
            })

        # if label_override_fn is not None:
        #     train_df["label_used"] = train_df.apply(
        #         lambda row: label_override_fn(row["label_used"], row["static_label"]),
        #         axis=1
        #     )
        #     test_df["label_used"] = test_df.apply(
        #         lambda row: label_override_fn(row["label_used"], row["static_label"]),
        #         axis=1
        #     )

        train_df = train_df[train_df["label_used"].notna()].copy()
        test_df = test_df[test_df["label_used"].notna()].copy()

        # drop unusual labels
        train_df = train_df[train_df["label_used"] != "other"].copy()
        test_df = test_df[test_df["label_used"] != "other"].copy()
        
        

        # # drop break
        # train_df = train_df[train_df["label_used"] != "break"].copy()
        # test_df = test_df[test_df["label_used"] != "break"].copy()

        # # drop lying_arms_up
        # train_df = train_df[train_df["label_used"] != "lying_arms_up"].copy()
        # test_df = test_df[test_df["label_used"] != "lying_arms_up"].copy()

        
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

    run_name = f"{clf_name}_{sensor_combo_scenario}_expanded{expanded_fsr}_class{class_version}"

    make_confusion_matrix(
        cf=cm,
        categories=labels,
        title=f"{clf_name} Confusion Matrix",
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


# if __name__ == "__main__":
#     run_loocv_with_pca(
#         prefixes=["test", "aksowork", "prelim", "aksoprotocol"],
#         left_arm=False,
#         upper_back=False,
#         clf_name="NN",
#         expanded_fsr=True,
#         taxonomy_fn=map_taxonomy_candidate_4,
#         label_override_fn=None
#     )


# NEW MAIN FUNCTION FOR SCENARIO ANALYSIS
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
                            "Window3.5", 
                            "Window5"
                             ]
        else:
            seg_scenarios = [
                "Window2.5", "Window3.5", "Window5", 
                             "Repetition3.5"]
        
        # Evaluate protocol only datasets with their original labels in addition to the 
        if DC_id in ["DC1", "DC2", "DC5"]:
            taxonomys = {
                         "T1":map_taxonomy_candidate_4,
                         "T2":map_taxonomy_candidate_3,
                         "T3":None}
        else:
            # otherwsie use T2 as max granularity
            taxonomys = {"T1":map_taxonomy_candidate_4,
                         "T2":map_taxonomy_candidate_3,
                         }
        
        # configure such that the fullest available sensor combination scenario is used
        if "test" in DC:
            left_arm = False
            upper_back = False
            SC = "SC"
        else:
            left_arm=True
            upper_back=True
            SC = "SC1"
        
        for seg in seg_scenarios:
            for clf in [
                "NN", 
                "SVC",
                "RFC"
                        ]:
                for taxonomy_id, tax_fn in taxonomys.items():
                    summary = run_loocv_with_pca(
                        prefixes=DC,
                        left_arm=left_arm,
                        upper_back=upper_back,
                        clf_name=clf,
                        expanded_fsr=True,
                        taxonomy_fn=tax_fn,
                        seg_strategy=seg
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
                results_df.to_csv(f"./results/loocv_summary_results_FINAL.csv", index=False)
                print(results_df)
            