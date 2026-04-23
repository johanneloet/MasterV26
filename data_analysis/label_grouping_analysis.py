# Workflow:
# For each scenario we wish to evaluate, see thesis, run three clustering algorithms: k-means, hdbscan and agglomerative.
# Study plots for each scenario and conclude on two levels of taxonomy that will work across datasets. One very coarse, and one finer (which will perform worse)

from data_analysis.dbscan_clustering import run_dbscan_on_dataset
from data_analysis.kmeans_clustering import run_kmeans_model_selection_on_dataset, run_kmeans_on_dataset
from data_analysis.agglomerative_clustering import run_agglomerative_on_dataset
from feature_extraction.get_paths import get_test_folder_paths
from plotting.cluster_plots import plot_pca_clusters, save_cluster_label_heatmap
from utils import map_label_hierarchical

import pandas as pd
import numpy as np
import re

def run_clustering_analysis(analysis_scenarios):
    for scenario in analysis_scenarios:
        prefixes = scenario["datasets"]
        right_arm = scenario["sensors"]["right_arm"]
        right_arm  = scenario["sensors"]["right_arm"]
        left_arm   = scenario["sensors"]["left_arm"]
        lower_back = scenario["sensors"]["lower_back"]
        upper_back = scenario["sensors"]["upper_back"]
        left_fsr   = scenario["sensors"]["left_fsr"]
        right_fsr  = scenario["sensors"]["right_fsr"]
        segmentation_strats = scenario["segmentation_strategies"]

        for strat in segmentation_strats:
            match = re.match(r"(Window|Repetition)(\d+(?:\.\d+)?)", strat)
            feature_mode = match.group(1)
            window_length = match.group(2)
            print("------------------------")
            print("Scenario:", scenario["id"], "Segmentation:", strat)
            print("Beginning HDBSCAN clustering...")
            dbscan_df, _, _, _ = run_dbscan_on_dataset(
                left_arm=left_arm, 
                right_arm=right_arm,
                lower_back=lower_back,
                upper_back=upper_back,
                left_fsr=left_fsr,
                right_fsr=right_fsr,
                expanded_fsr=True,
                prefixes=prefixes,
                feature_mode=feature_mode,
                feature_window_sec=window_length
            )
            plot_pca_clusters(dbscan_df,
                              save_path= f"cluster_plots/dbscan_{strat}_{scenario["id"]}.pdf",
                              style_by="prefix")
            save_cluster_label_heatmap(
            dbscan_df,
            filename=f"dbscan_cluster_heatmap_{scenario["id"]}_{strat}.pdf",
            cluster_col="cluster",
            label_col="label",
            title=f"DBSCAN: Cluster vs Label (%) for scenario {scenario["id"]} SEG{strat}",
            map_label_fn=map_label_hierarchical,
            drop_noise=False,   # keep -1 in plot
            min_total_label_count=0,
            sort_labels=False,
            sort_clusters=False,
            )
            print("Beginning k-means analysis...")
            kmeans_df, _, _, _ = run_kmeans_on_dataset(
                left_arm=left_arm, 
                right_arm=right_arm,
                lower_back=lower_back,
                upper_back=upper_back,
                left_fsr=left_fsr,
                right_fsr=right_fsr,
                expanded_fsr=True,
                prefixes=prefixes,
                feature_mode=feature_mode,
                feature_window_length=window_length,
                n_clusters = 4
            )
            plot_pca_clusters(kmeans_df,
                              save_path= f"cluster_plots/kmeans_{strat}_{scenario["id"]}.pdf",
                              style_by="prefix")
            save_cluster_label_heatmap(
            kmeans_df,
            filename=f"kmeans_cluster_heatmap_{scenario["id"]}_{strat}.pdf",
            cluster_col="cluster",
            label_col="label",
            title=f"KMEANS: Cluster vs Label (%) for scenario {scenario["id"]} SEG{strat}",
            map_label_fn=map_label_hierarchical,
            drop_noise=False,   # keep -1 in plot
            min_total_label_count=0,
            sort_labels=False,
            sort_clusters=False,
            )

            print("Beginning agglomerative clustering analysis...")
            agglo_df, _, _, _ ,_ = run_agglomerative_on_dataset(
                left_arm=left_arm, 
                right_arm=right_arm,
                lower_back=lower_back,
                upper_back=upper_back,
                left_fsr=left_fsr,
                right_fsr=right_fsr,
                expanded_fsr=True,
                prefixes=prefixes,
                feature_mode=feature_mode,
                feature_window_sec=window_length,
                n_clusters = 4
            )
            plot_pca_clusters(agglo_df,
                              save_path= f"cluster_plots/agglomerative_{strat}_{scenario["id"]}.pdf",
                              style_by="prefix")
            save_cluster_label_heatmap(
            agglo_df,
            filename=f"agglomerative_cluster_heatmap_{scenario["id"]}_{strat}.pdf",
            cluster_col="cluster",
            label_col="label",
            title=f"AGGLOMERATIVE: Cluster vs Label (%) for scenario {scenario["id"]} SEG{strat}",
            map_label_fn=map_label_hierarchical,
            drop_noise=False,   # keep -1 in plot
            min_total_label_count=0,
            sort_labels=False,
            sort_clusters=False,
            )
            print("Finished scenario:)")


if __name__ == '__main__':
    full_4_sensors = {
        "right_arm":True,
        "left_arm":False,
        "lower_back":True,
        "upper_back":False,
        "left_fsr":True,
        "right_fsr":True,}
    full_6_sensors = {
        "right_arm":True,
        "left_arm":True,
        "lower_back":True,
        "upper_back":True,
        "left_fsr":True,
        "right_fsr":True,}
    dataset_segment_scenarios = [
        #C1
        # {"datasets" : ["test"],
        #  "sensors" : full_4_sensors,
        #  "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
        #  "id" : "C1"
        #  }, 
         #C2
        {"datasets" : ["prelim", "aksoprotocol"],
         "sensors" : full_6_sensors,
         "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
         "id" : "C2"
         }, 
         #C3
        {"datasets" : ["aksowork"],
         "sensors" : full_6_sensors,
         "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
         "id" : "C3"
         }, 
         #C4
         {"datasets" : ["aksowork", "aksoprotocol", "prelim"],
         "sensors" : full_6_sensors,
         "segmentation_strategies" : [
             "Window2.5", "Window3.5", "Window5", 
                                      "Repetition3.5"],
         "id" : "C4" 
         }, 
         #C5
        #  {"datasets" : ["test", "prelim", "aksoprotocol"],
        #  "sensors" : full_4_sensors,
        #  "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
        #  "id" : "C5" 
        #  }, 
        #C6
        #  {"datasets" : ["test", "aksowork"],
        #  "sensors" : full_4_sensors,
        #  "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
        #  "id" : "C6" 
        #  }, 
        #  #C7
        #  {"datasets" : ["test", "aksowork", "prelim", "aksoprotocol"],
        #  "sensors" : full_4_sensors,
        #  "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
        #  "id" : "C7"  
        #  }, 
        ]
    
    run_clustering_analysis(analysis_scenarios=dataset_segment_scenarios)