from data_analysis.loocv import run_loocv_with_pca
import os
import pandas as pd
from utils import map_taxonomy_candidate_1, map_taxonomy_candidate_2, map_taxonomy_candidate_3, map_taxonomy_candidate_4

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
        "SC3" : {
            "right_arm" : True,
            "left_arm" : True,
            "lower_back" : False,
            "upper_back" : False,
            "left_fsr" : True,
            "right_fsr" : True
        },
        "SC4" : {
            "right_arm" : True,
            "left_arm" : False,
            "lower_back" : False,
            "upper_back" : False,
            "left_fsr" : True,
            "right_fsr" : True
        },
        "SC5" : {
            "right_arm" : False,
            "left_arm" : True,
            "lower_back" : False,
            "upper_back" : False,
            "left_fsr" : True,
            "right_fsr" : True
        },
        "SC6" : {
            "right_arm" : True,
            "left_arm" : True,
            "lower_back" : True,
            "upper_back" : True,
            "left_fsr" : False,
            "right_fsr" : False
        },
        "SC7" : {
            "right_arm" : True,
            "left_arm" : False,
            "lower_back" : True,
            "upper_back" : False,
            "left_fsr" : False,
            "right_fsr" : False
        },
        "SC8" : {
            "right_arm" : True,
            "left_arm" : True,
            "lower_back" : True,
            "upper_back" : False,
            "left_fsr" : True,
            "right_fsr" : True
        },
        "SC9" : {
            "right_arm" : False,
            "left_arm" : False,
            "lower_back" : False,
            "upper_back" : False,
            "left_fsr" : True,
            "right_fsr" : True
        },
    }
    
    classifiers = [
        "NN", 
        "SVC"
    ]
    
    # Define optimal taxonomy and segmentation and classifier strategy based on stage 1 analyses
    optimal_tax_and_seg_dict = {
        "DC1" : {
            "T_opt" : None,
            "seg_opt" : "Repetition3.5",
        },
        "DC2" : {
            "T_opt" : None,
            "seg_opt" : "Repetition3.5"
        },
        "DC3" : {
            "T_opt" : map_taxonomy_candidate_4,
            "seg_opt" : "Window3.5"
        },
        "DC4" : {
            "T_opt" : map_taxonomy_candidate_3,
            "seg_opt" : "Repetition3.5"
        },
        "DC5" : {
            "T_opt" : map_taxonomy_candidate_3,
            "seg_opt" : "Repetition3.5"
        },
        "DC6" : {
            "T_opt" : map_taxonomy_candidate_3,
            "seg_opt" : "Repetition3.5"
        },
        "DC7" : {
            "T_opt" : map_taxonomy_candidate_3,
            "seg_opt" : "Repetition3.5"
        },
        
    }
    
    results = []
    for DC_id, DC in dataset_scenarios.items():
        T_opt = optimal_tax_and_seg_dict[DC_id]["T_opt"]
        seg_opt = optimal_tax_and_seg_dict[DC_id]["seg_opt"]
        for SC_id, SC in sensor_combination_scenarios.items():
            if DC_id in ["DC1", "DC5", "DC6", "DC7"] and SC_id in ["SC1", "SC3", "SC5", "SC6", "SC8"]:
                continue # for dataset scenarios that include legacy data, do not run sensor combinations that include non-legacy sensors
            else:
                for clf in classifiers:
                    summary = run_loocv_with_pca(
                        clf_name=clf,
                        prefixes=DC,
                        right_arm= SC["right_arm"],
                        left_arm= SC["left_arm"],
                        lower_back= SC["lower_back"],
                        upper_back= SC["upper_back"],
                        left_fsr= SC["left_fsr"],
                        right_fsr= SC["right_fsr"],
                        expanded_fsr=True,
                        taxonomy_fn=T_opt,
                        seg_strategy=seg_opt,
                        save_per_participant_metrics=True
                    )
                    
                    summary.update({
                        "dataset_scenario": DC_id,
                        "prefixes": "+".join(DC),
                        "taxonomy": T_opt,
                        "segmentation": seg_opt,
                        "classifier": clf,
                        "expanded_fsr": True,
                        "sensor_scenario": SC_id,
                    })
                    
                    results.append(summary)
    results_df = pd.DataFrame(results)
    os.makedirs("./results", exist_ok=True)
    results_df.to_csv("./results/loocv_summary_results_sensor_ablation.csv", index=False)
    print(results_df)
        
