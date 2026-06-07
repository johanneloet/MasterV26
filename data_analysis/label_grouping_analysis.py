# Workflow:
# For each scenario we wish to evaluate, see thesis, run three clustering algorithms: k-means, hdbscan and agglomerative.
# Study plots for each scenario and conclude on two levels of taxonomy that will work across datasets. One very coarse, and one finer (which will perform worse)

from data_analysis.dbscan_clustering import run_dbscan_on_dataset
from data_analysis.kmeans_clustering import run_kmeans_model_selection_on_dataset, run_kmeans_on_dataset, plot_kmeans_model_selection
from data_analysis.agglomerative_clustering import run_agglomerative_on_dataset, plot_agglomerative_dendrogram
from feature_extraction.get_paths import get_test_folder_paths
from plotting.cluster_plots import plot_pca_clusters, save_cluster_label_heatmap, save_cluster_label_stacked_bar
from utils import map_taxonomy_candidate_1, map_taxonomy_candidate_2, map_taxonomy_candidate_3, map_taxonomy_candidate_4

import pandas as pd
import numpy as np
import re

def add_taxonomy_columns(df, map_label_fns):
    df = df.copy()

    # remove rows without valid labels
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(str)

    for taxonomy_name, map_fn in map_label_fns.items():
        df[taxonomy_name] = df.apply(
            lambda row: map_fn(
                row["label"],
                row.get("static_label", None)
            ),
            axis=1
        )

    return df


def valid_taxonomy_df(df, taxonomy_col):
    return df[
        df[taxonomy_col].notna()
        & (df[taxonomy_col].astype(str).str.lower().str.strip() != "other")
    ].copy()


def run_clustering_analysis(analysis_scenarios):

    map_label_fns = {
        "taxonomy1": map_taxonomy_candidate_1,
        "taxonomy2": map_taxonomy_candidate_2,
        "taxonomy3": map_taxonomy_candidate_3,
        "taxonomy4": map_taxonomy_candidate_4,
    }

    for scenario in analysis_scenarios:

        prefixes = scenario["datasets"]

        right_arm = scenario["sensors"]["right_arm"]
        left_arm = scenario["sensors"]["left_arm"]
        lower_back = scenario["sensors"]["lower_back"]
        upper_back = scenario["sensors"]["upper_back"]
        left_fsr = scenario["sensors"]["left_fsr"]
        right_fsr = scenario["sensors"]["right_fsr"]

        segmentation_strats = scenario["segmentation_strategies"]

        for strat in segmentation_strats:

            match = re.match(r"(Window|Repetition)(\d+(?:\.\d+)?)", strat)

            if match is None:
                raise ValueError(f"Could not parse segmentation strategy: {strat}")

            feature_mode = match.group(1)
            window_length = match.group(2)

            print("------------------------")
            print("Scenario:", scenario["id"], "Segmentation:", strat)

            # HDBSCAN
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
                feature_window_sec=window_length,
                map_label_fn=None,   # IMPORTANT: fixed clustering dataset
            )

            dbscan_df = add_taxonomy_columns(dbscan_df, map_label_fns)

            plot_pca_clusters(
                dbscan_df,
                save_path=f"cluster_plots/dbscan_{strat}_{scenario['id']}_clusters.pdf",
                #style_by="prefix",
                cmap_name="pink_yellow_turquoise",
                algorithm="HDBSCAN"
            )

            for taxonomy_name in map_label_fns:
                tax_df = valid_taxonomy_df(dbscan_df, taxonomy_name)

                save_cluster_label_stacked_bar(
                    tax_df,
                    filename=f"dbscan_cluster_barchart_{scenario['id']}_{strat}_{taxonomy_name}.pdf",
                    cluster_col="cluster",
                    label_col=taxonomy_name,
                    title=f"HDBSCAN: Cluster vs Label (%) for scenario {scenario['id']} {strat}_{taxonomy_name}",
                    map_label_fn=None,
                    drop_noise=False,
                    min_total_label_count=0,
                    sort_labels=False,
                    sort_clusters=False,
                )

            # K-means
            print("Beginning k-means analysis...")
            results = run_kmeans_model_selection_on_dataset(
            right_arm=right_arm,
            left_arm=left_arm,
            lower_back=lower_back,
            upper_back=upper_back,
            left_fsr=left_fsr,
            right_fsr=right_fsr,
            expanded_fsr=True,
            prefixes=prefixes,
            feature_mode=feature_mode,
            feature_window_length=window_length,
            k_values=range(2, 13),
            use_pca=True,
            n_pca=0.95,
            random_state=343,)

            plot_kmeans_model_selection(results, filename=f"{scenario['id']}_kmeans_silhouette.pdf")
            

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
                n_clusters=4,
                map_label_fn=None, # IMPORTANT
                random_state=343
            )

            kmeans_df = add_taxonomy_columns(kmeans_df, map_label_fns)

            plot_pca_clusters(
                kmeans_df,
                save_path=f"cluster_plots/kmeans_{strat}_{scenario['id']}_clusters.pdf",
                #style_by="prefix",
                cmap_name="pink_yellow_turquoise",
                algorithm="Kmeans"
            )

            for taxonomy_name in map_label_fns:
                tax_df = valid_taxonomy_df(kmeans_df, taxonomy_name)

                save_cluster_label_stacked_bar(
                    tax_df,
                    filename=f"kmeans_cluster_barchart_{scenario['id']}_{strat}_{taxonomy_name}.pdf",
                    cluster_col="cluster",
                    label_col=taxonomy_name,
                    title=f"K-means: Cluster vs Label (%) for scenario {scenario['id']} {strat}_{taxonomy_name}",
                    map_label_fn=None,
                    drop_noise=False,
                    min_total_label_count=0,
                    sort_labels=False,
                    sort_clusters=False,
                )

            # Agglomerative
            print("Beginning agglomerative clustering analysis...")

            agglo_df, _, _, _, model = run_agglomerative_on_dataset(
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
                n_clusters=4,
                map_label_fn=None,   # IMPORTANT
            )

            agglo_df = add_taxonomy_columns(agglo_df, map_label_fns)
            plot_agglomerative_dendrogram(model,
                                          filename=f"{scenario['id']}_agglo_tree.pdf" ,                  
            )
            plot_pca_clusters(
                agglo_df,
                save_path=f"cluster_plots/agglomerative_{strat}_{scenario['id']}_clusters.pdf",
                #style_by="prefix",
                cmap_name="pink_yellow_turquoise",
                algorithm="Agglomerative"
            )

            for taxonomy_name in map_label_fns:
                tax_df = valid_taxonomy_df(agglo_df, taxonomy_name)

                save_cluster_label_stacked_bar(
                    tax_df,
                    filename=f"agglo_cluster_barchart_{scenario['id']}_{strat}_{taxonomy_name}.pdf",
                    cluster_col="cluster",
                    label_col=taxonomy_name,
                    title=f"Agglomerative: Cluster vs Label (%) for scenario {scenario['id']} {strat}_{taxonomy_name}",
                    map_label_fn=None,
                    drop_noise=False,
                    min_total_label_count=0,
                    sort_labels=False,
                    sort_clusters=False,
                )

            print("Finished scenario :)")


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
        # {"datasets" : ["prelim", "aksoprotocol"],
        #  "sensors" : full_6_sensors,
        #  "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
        #  "id" : "C2"
        #  }, 
         #C3
        # {"datasets" : ["aksowork"],
        #  "sensors" : full_6_sensors,
        #  "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
        #  "id" : "C3"
        #  }, 
         #C4
         {"datasets" : ["aksowork", "aksoprotocol", "prelim"],
         "sensors" : full_6_sensors,
         "segmentation_strategies" : [
             #"Window2.5", 
             "Window3.5", 
            #     "Window5", 
            # "Repetition3.5"
            ],
         "id" : "C4" 
         }, 
        #  #C5
        #  {"datasets" : ["test", "prelim", "aksoprotocol"],
        #  "sensors" : full_4_sensors,
        #  "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
        #  "id" : "C5" 
        #  }, 
        # #C6
        #  {"datasets" : ["test", "aksowork"],
        #  "sensors" : full_4_sensors,
        #  "segmentation_strategies" : ["Window2.5", "Window3.5", "Window5", "Repetition3.5"],
        #  "id" : "C6" 
        #  }, 
        #  #C7
         {"datasets" : ["test", "aksowork", "prelim", "aksoprotocol"],
         "sensors" : full_4_sensors,
         "segmentation_strategies" : [
             #"Window2.5", 
             "Window3.5", 
            # "Window5", "Repetition3.5"
             ],
         "id" : "C7"  
         }, 
        ]
    
    run_clustering_analysis(analysis_scenarios=dataset_segment_scenarios)