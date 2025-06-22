"""
Complete example of loading CAE-extracted features and survival data
for SubtypeCtAE clustering - matched to your subtypectae.py
"""

import numpy as np
import pandas as pd
import os
import sys
import logging

from subtype_ctae import SubtypeCtAE

# Create a logger
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# Option 1: Load from saved .npy files (if you saved features after CAE training)
def load_cae_features_from_files(features_dir="models"):
    """
    Load CAE-extracted features from saved numpy files
    """
    logger.info("Loading CAE-extracted features...")

    features_dict = {}

    # Load each omics type's features
    omics_types = ["mrna", "cnv", "mirna", "dnameth"]

    for omics in omics_types:
        feature_file = os.path.join(features_dir, f"{omics}.out_ef.npy")
        if os.path.exists(feature_file):
            features = np.load(feature_file)
            features_dict[omics] = features
            logger.info(f"  {omics}: loaded {features.shape} features")
        else:
            logger.info(f"  Warning: {feature_file} not found!")

    return features_dict


# Load survival data
def load_survival_data(survival_path):
    """
    Load survival data from CSV file
    Expected format: columns should include 'time' and 'event' (status)
    """
    logger.info(f"\nLoading survival data from {survival_path}...")

    # Read CSV file
    survival_df = pd.read_csv(survival_path, sep="\t")

    # Ensure numeric types
    survival_df["time"] = pd.to_numeric(survival_df["OS.time"], errors="coerce")
    survival_df["event"] = pd.to_numeric(survival_df["OS"], errors="coerce")

    # Remove any NaN values
    initial_count = len(survival_df)
    survival_df = survival_df.dropna(subset=["time", "event"])
    if len(survival_df) < initial_count:
        logger.info(
            f"  Removed {initial_count - len(survival_df)} samples with missing survival data"
        )

    logger.info(f"  Loaded survival data for {len(survival_df)} samples")
    logger.info(
        f"  Survival time range: {survival_df['time'].min():.1f} - {survival_df['time'].max():.1f}"
    )
    logger.info(f"  Event rate: {survival_df['event'].mean():.1%}")

    return survival_df


# Main pipeline exactly matching subtypectae.py requirements
def run_subtypectae_pipeline():
    """
    Complete example of running SubtypeCtAE with your data
    """

    # === 1. LOAD CAE FEATURES ===
    # Option A: If you saved features as .npy files
    features_dict = load_cae_features_from_files()

    # === 2. LOAD SURVIVAL DATA ===
    survival_data = load_survival_data("data/survival.filtered.tsv")

    # === 3. ENSURE SAMPLE ALIGNMENT ===
    # Make sure the samples in features and survival data match
    # This assumes samples are in the same order
    n_samples_features = next(iter(features_dict.values())).shape[0]
    n_samples_survival = len(survival_data)

    if n_samples_features != n_samples_survival:
        logger.info(f"\nWarning: Sample count mismatch!")
        logger.info(f"  Features: {n_samples_features} samples")
        logger.info(f"  Survival: {n_samples_survival} samples")

        # If you have sample IDs, use them to align
        # Otherwise, take the minimum
        min_samples = min(n_samples_features, n_samples_survival)
        logger.info(f"  Using first {min_samples} samples")

        for omics in features_dict:
            features_dict[omics] = features_dict[omics][:min_samples]
        survival_data = survival_data.iloc[:min_samples]

    # === 4. RUN SUBTYPECTAE ===
    logger.info("\n" + "=" * 50)
    logger.info("Running SubtypeCtAE Clustering")
    logger.info("=" * 50)

    summaries = []

    # Initialize SubtypeCtAE
    thresholds = (0.01, 0.05, 0.1, 0.15, 0.2)
    # thresholds = 0.05 # Best result so far
    for threshold in thresholds:
        logger.info(f"\nRunning with p-value threshold: {threshold}")

        model = SubtypeCtAE(p_value_threshold=threshold)

        # Run the complete pipeline
        # n_clusters = 5 is best result
        # for i in range(5, 6):
        for i in range(2, 6):
            results = model.fit_predict(
                features_dict,
                survival_data,
                n_clusters=i,
                plot_survival=True,
            )
            if results is None:
                summary = {
                    "p_value_threshold": threshold,
                    "status": "Failed",
                }
                summaries.append(summary)
                break
            else:
                summary = {
                    "p_value_threshold": threshold,
                    "n_clusters": results["n_clusters"],
                    "c_index": results["c_index"],
                    "p_value": results["p_value"],
                    "silhouette_score": results["silhouette_score"],
                    "n_selected_features": results["n_selected_features"],
                    "n_total_features": results["n_total_features"],
                }
                summaries.append(summary)

                logger.info(f"\nClustering complete for threshold {threshold}:")
                logger.info(f"Found {results['n_clusters']} cancer subtypes")
                logger.info(f"C-index: {results['c_index']:.3f}")
                logger.info(f"Log-rank p-value: {results['p_value']:.2e}")

    # === 5. SAVE RESULTS ===
    logger.info("\n" + "=" * 50)
    logger.info("Saving Results")
    logger.info("=" * 50)

    count = 0
    with open("clustering_summary.txt", "w") as f:
        f.write("SubtypeCtAE Clustering Summary\n")
        for summary in summaries:
            count += 1
            f.write("=" * 30 + "\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    # # Save cluster labels
    # np.save("cluster_labels.npy", results["labels"])

    # Create detailed results DataFrame
    # results_df = pd.DataFrame(
    #     {
    #         "sample_id": [f"Sample_{i}" for i in range(len(results["labels"]))],
    #         "cluster": results["labels"] + 1,  # 1-indexed
    #         "survival_time": survival_data["time"].values,
    #         "event": survival_data["event"].values,
    #     }
    # )
    # results_df.to_csv("clustering_results.csv", index=False)

    # Save summary statistics
    # summary = {
    #     "n_clusters": results["n_clusters"],
    #     "c_index": results["c_index"],
    #     "p_value": results["p_value"],
    #     "silhouette_score": results["silhouette_score"],
    #     "n_selected_features": results["n_selected_features"],
    #     "n_total_features": results["n_total_features"],
    # }

    return results


if __name__ == "__main__":
    # Example 1: Run with real data files
    # Make sure you have:
    # - cae_features/features_mRNA.npy
    # - cae_features/features_CNV.npy
    # - cae_features/features_miRNA.npy
    # - cae_features/features_Methylation.npy
    # - data/survival_data.csv (with 'time' and 'event' columns)

    results = run_subtypectae_pipeline()

    # Example 2: If you have features in memory
    # features_dict = {
    #     'mRNA': your_mRNA_features,  # numpy array from CAE
    #     'CNV': your_CNV_features,
    #     'miRNA': your_miRNA_features,
    #     'Methylation': your_Meth_features
    # }
    # survival_data = pd.DataFrame({
    #     'time': your_survival_times,
    #     'event': your_event_indicators
    # })
    #
    # model = SubtypeCtAE(p_value_threshold=0.01)
    # results = model.fit_predict(features_dict, survival_data)
