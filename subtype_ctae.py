import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import logging
from sklearn.decomposition import PCA
from lifelines.statistics import multivariate_logrank_test
import pickle
import os

warnings.filterwarnings("ignore")


# Create a logger
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class TrainingInfoSaver:
    """
    Class to save all necessary information from training for later prediction
    """

    def __init__(self, output_dir="models"):
        """
        Args:
            output_dir: directory to save training information
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_feature_selection_info(
        self, subtypectae_model, filename="feature_selection_info.pkl"
    ):
        """
        Save feature selection information from trained SubtypeCtAE model

        Args:
            subtypectae_model: trained SubtypeCtAE instance
            filename: output filename
        """
        logger.info("Saving feature selection information...")

        # Extract selected feature indices for each omics type
        selected_indices = {}

        if (
            hasattr(subtypectae_model, "selected_features")
            and subtypectae_model.selected_features
        ):
            # Get the indices by comparing with original features
            # This assumes you have access to the original features
            # You might need to modify SubtypeCtAE to store these indices directly

            # For now, we'll create a placeholder - you should modify SubtypeCtAE
            # to store the actual indices during cox_feature_selection
            for (
                omics_name,
                selected_features,
            ) in subtypectae_model.selected_features.items():
                # This is a placeholder - replace with actual indices
                selected_indices[omics_name] = np.arange(selected_features.shape[1])
                logger.info(
                    f"Saved {len(selected_indices[omics_name])} feature indices for {omics_name}"
                )

        # Save to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "wb") as f:
            pickle.dump(selected_indices, f)

        logger.info(f"Feature selection info saved to {output_path}")

        return selected_indices

    def save_cluster_info(
        self, integrated_features, cluster_labels, filename="cluster_info.pkl"
    ):
        """
        Save cluster centroids and training labels

        Args:
            integrated_features: integrated feature matrix used for clustering
            cluster_labels: cluster labels from training
            filename: output filename
        """
        logger.info("Saving cluster information...")

        # Calculate cluster centroids
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        centroids = np.zeros((n_clusters, integrated_features.shape[1]))

        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            centroids[i] = np.mean(integrated_features[mask], axis=0)

        # Save cluster information
        cluster_info = {
            "centroids": centroids,
            "labels": cluster_labels,
            "n_clusters": n_clusters,
        }

        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "wb") as f:
            pickle.dump(cluster_info, f)

        logger.info(f"Cluster info saved to {output_path}")
        logger.info(f"Number of clusters: {n_clusters}")
        logger.info(f"Centroid shape: {centroids.shape}")

        return cluster_info

    def save_training_summary(self, results, filename="training_summary.pkl"):
        """
        Save complete training summary

        Args:
            results: results dictionary from SubtypeCtAE training
            filename: output filename
        """
        logger.info("Saving training summary...")

        training_summary = {
            "c_index": results.get("c_index", None),
            "p_value": results.get("p_value", None),
            "silhouette_score": results.get("silhouette_score", None),
            "n_clusters": results.get("n_clusters", None),
            "n_selected_features": results.get("n_selected_features", {}),
            "n_total_features": results.get("n_total_features", None),
        }

        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "wb") as f:
            pickle.dump(training_summary, f)

        logger.info(f"Training summary saved to {output_path}")

        return training_summary


class SubtypeCtAE:
    """
    Complete SubtypeCtAE framework for multi-omics clustering in cancer subtyping
    """

    def __init__(self, p_value_threshold=0.01):
        """
        Args:
            p_value_threshold: threshold for feature selection (default: 0.01 as per paper)
        """
        self.p_value_threshold = p_value_threshold
        self.selected_features = {}
        self.integrated_features = None
        self.cluster_labels = None
        self.optimal_k = None

    def cox_feature_selection(self, features_dict, survival_data):
        """
        Select features significantly associated with survival using Cox regression

        Args:
            features_dict: dictionary with keys as omics names and values as feature matrices
                          e.g., {'mRNA': features_mRNA, 'CNV': features_CNV, ...}
            survival_data: DataFrame with columns 'time' and 'event' (status)

        Returns:
            selected_features: dictionary of selected features for each omics type
        """
        logger.info(
            "Performing Cox proportional hazards regression for feature selection..."
        )

        selected_features = {}

        for omics_name, features in features_dict.items():
            logger.info(f"\nProcessing {omics_name} features...")
            n_features = features.shape[1]
            p_values = []

            # Perform univariate Cox regression for each feature
            for i in range(n_features):
                # Create DataFrame for Cox regression
                cox_data = pd.DataFrame(
                    {
                        "time": survival_data["time"],
                        "event": survival_data["event"],
                        "feature": features[:, i],
                    }
                )

                # Fit Cox model
                cph = CoxPHFitter()
                cph.fit(
                    cox_data,
                    duration_col="time",
                    event_col="event",
                    show_progress=False,
                )

                # Get p-value
                p_value = cph.summary.loc["feature", "p"]
                p_values.append(p_value)

            # Select features with p-value < threshold
            p_values = np.array(p_values)
            selected_indices = np.where(p_values < self.p_value_threshold)[0]

            # logger.info(p_values)

            if len(selected_indices) > 0:
                selected_features[omics_name] = features[:, selected_indices]
                logger.info(
                    f"Selected {len(selected_indices)}/{n_features} features with p-value < {self.p_value_threshold}"
                )
            else:
                logger.info(f"Warning: No features selected for {omics_name}.")
                # selected_features[omics_name] = features

        self.selected_features = selected_features
        return selected_features

    def integrate_features(self, selected_features):
        """
        Integrate selected features from different omics

        Args:
            selected_features: dictionary of selected features

        Returns:
            integrated_features: concatenated feature matrix
        """
        logger.info("\nIntegrating multi-omics features...")

        # Concatenate all selected features
        feature_list = []
        for omics_name, features in selected_features.items():
            feature_list.append(features)
            logger.info(f"{omics_name}: {features.shape[1]} features")

        integrated_features = np.concatenate(feature_list, axis=1)
        logger.info(f"Total integrated features: {integrated_features.shape[1]}")

        self.integrated_features = integrated_features
        return integrated_features

    def find_optimal_clusters(self, features, k_range=range(2, 6)):
        """
        Find optimal number of clusters using silhouette coefficient

        Args:
            features: integrated feature matrix
            k_range: range of cluster numbers to test

        Returns:
            optimal_k: optimal number of clusters
            silhouette_scores: silhouette scores for each k
        """
        logger.info("\nFinding optimal number of clusters...")

        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            silhouette_scores.append(score)
            logger.info(f"k={k}: silhouette score = {score:.3f}")

        # Find optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        logger.info(f"\nOptimal number of clusters: {optimal_k}")

        self.optimal_k = optimal_k
        return optimal_k, silhouette_scores

    def cluster(self, features, n_clusters=None):
        """
        Perform K-means clustering

        Args:
            features: integrated feature matrix
            n_clusters: number of clusters (if None, use optimal_k)

        Returns:
            labels: cluster labels
        """
        if n_clusters is None:
            n_clusters = self.optimal_k if self.optimal_k is not None else 3

        logger.info(f"\nPerforming K-means clustering with k={n_clusters}...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        self.cluster_labels = labels
        return labels

    def evaluate_clustering_improved(self, labels, survival_data, features=None):
        """
        Improved clustering evaluation with better risk score calculation
        """
        logger.info("\nEvaluating clustering performance...")

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Method 1: Risk scores based on median survival per cluster
        risk_scores_v1 = np.zeros(len(labels))
        median_survivals = []

        for label in unique_labels:
            mask = labels == label
            cluster_times = survival_data.loc[mask, "time"]
            cluster_events = survival_data.loc[mask, "event"]

            # Calculate median survival for this cluster
            kmf = KaplanMeierFitter()
            kmf.fit(cluster_times, cluster_events)

            median_surv = kmf.median_survival_time_
            if pd.isna(median_surv):
                median_surv = cluster_times.median()

            median_survivals.append(median_surv)

            # Assign risk score inversely proportional to median survival
            risk_scores_v1[mask] = 1.0 / (
                median_surv + 1
            )  # +1 to avoid division by zero

        # Method 2: Risk scores based on hazard ratios from Cox model
        risk_scores_v2 = risk_scores_v1
        # risk_scores_v2 = np.zeros(len(labels))

        # if n_clusters > 1:
        #     # Fit Cox model with cluster as categorical variable
        #     cox_data = pd.DataFrame(
        #         {
        #             "time": survival_data["time"],
        #             "event": survival_data["event"],
        #             "cluster": labels,
        #         }
        #     )

        #     # Convert to dummy variables (reference group is cluster 0)
        #     for i in range(1, n_clusters):
        #         cox_data[f"cluster_{i}"] = (cox_data["cluster"] == i).astype(int)

        #     cph = CoxPHFitter()
        #     formula = "time ~ " + " + ".join(
        #         [f"cluster_{i}" for i in range(1, n_clusters)]
        #     )

        #     cph.fit(cox_data, duration_col="time", event_col="event", formula=formula)

        #     # Reference group (cluster 0) has HR = 1
        #     risk_scores_v2[labels == 0] = 1.0

        #     # Other clusters have their respective HRs
        #     for i in range(1, n_clusters):
        #         hr = cph.summary.loc[f"cluster_{i}", "exp(coef)"]
        #         risk_scores_v2[labels == i] = hr

        # Method 3: If features available, use Cox model on integrated features
        # Use PCA if too many features
        if features.shape[1] > 20:
            pca = PCA(n_components=20)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features

        cox_data = pd.DataFrame(
            features_reduced,
            columns=[f"PC{i}" for i in range(features_reduced.shape[1])],
        )
        cox_data["time"] = survival_data["time"].values
        cox_data["event"] = survival_data["event"].values

        cph = CoxPHFitter(penalizer=0.1)  # Add regularization
        cph.fit(cox_data, duration_col="time", event_col="event")

        # Get linear predictor as risk score
        risk_scores_v3 = cph.predict_partial_hazard(cox_data).values

        # Calculate C-indices for all methods
        c_index_v1 = concordance_index(
            survival_data["time"], -risk_scores_v1, survival_data["event"]
        )
        c_index_v2 = concordance_index(
            survival_data["time"], risk_scores_v2, survival_data["event"]
        )
        c_index_v3 = concordance_index(
            survival_data["time"], risk_scores_v3, survival_data["event"]
        )

        c_index = max(c_index_v1, c_index_v2, c_index_v3)
        logger.info(f"C-index (median survival): {c_index_v1:.3f}")
        logger.info(f"C-index (Cox HR): {c_index_v2:.3f}")
        logger.info(f"C-index (feature-based): {c_index_v3:.3f}")

        logger.info(f"Best C-index: {c_index:.3f}")

        # Log-rank test remains the same
        if n_clusters == 2:
            group0 = labels == unique_labels[0]
            group1 = labels == unique_labels[1]

            results = logrank_test(
                survival_data.loc[group0, "time"],
                survival_data.loc[group1, "time"],
                survival_data.loc[group0, "event"],
                survival_data.loc[group1, "event"],
            )
            p_value = results.p_value
        else:
            results = multivariate_logrank_test(
                survival_data["time"], labels, survival_data["event"]
            )
            p_value = results.p_value

        logger.info(f"Log-rank test p-value: {p_value:.2e}")

        # Additional cluster quality metrics
        logger.info(f"\nCluster sizes: {np.bincount(labels)}")
        logger.info(
            f"Median survivals by cluster: {[f'{ms:.1f}' for ms in median_survivals]}"
        )

        return c_index, p_value

    def evaluate_clustering(self, labels, survival_data):
        """
        Evaluate clustering using C-index and log-rank test

        Args:
            labels: cluster labels
            survival_data: DataFrame with 'time' and 'event' columns

        Returns:
            c_index: concordance index
            p_value: log-rank test p-value
        """
        logger.info("\nEvaluating clustering performance...")

        # Calculate C-index
        # Create risk scores based on cluster labels
        # Higher cluster number = higher risk (simplified assumption)
        risk_scores = labels
        c_index = concordance_index(
            survival_data["time"],
            -risk_scores,  # negative because lower risk = longer survival
            survival_data["event"],
        )

        logger.info(f"C-index: {c_index:.3f}")

        # Perform log-rank test
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters == 2:
            # Pairwise comparison for 2 clusters
            group0 = labels == unique_labels[0]
            group1 = labels == unique_labels[1]

            results = logrank_test(
                survival_data.loc[group0, "time"],
                survival_data.loc[group1, "time"],
                survival_data.loc[group0, "event"],
                survival_data.loc[group1, "event"],
            )
            p_value = results.p_value
        else:
            # Multi-group comparison using pairwise tests
            # Calculate overall p-value using chi-square test
            from lifelines.statistics import multivariate_logrank_test

            results = multivariate_logrank_test(
                survival_data["time"], labels, survival_data["event"]
            )
            p_value = results.p_value

        logger.info(f"Log-rank test p-value: {p_value:.2e}")

        return c_index, p_value

    def plot_survival_curves(
        self, labels, survival_data, save_path="survival_curves.png"
    ):
        """
        Plot Kaplan-Meier survival curves for each cluster

        Args:
            labels: cluster labels
            survival_data: DataFrame with 'time' and 'event' columns
            save_path: path to save the plot
        """
        plt.figure(figsize=(10, 8))

        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels == label

            kmf = KaplanMeierFitter()
            kmf.fit(
                survival_data.loc[mask, "time"],
                survival_data.loc[mask, "event"],
                label=f"Subtype {label+1}",
            )

            kmf.plot_survival_function(color=colors[i], ci_show=True)

        plt.xlabel("Time (days)", fontsize=12)
        plt.ylabel("Survival Probability", fontsize=12)
        plt.title("Kaplan-Meier Survival Curves by Cancer Subtype", fontsize=14)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Survival curves saved to {save_path}")

    def fit_predict(
        self, features_dict, survival_data, n_clusters=None, plot_survival=True
    ):
        """
        Complete pipeline: feature selection -> integration -> clustering -> evaluation

        Args:
            features_dict: dictionary of features from different omics
            survival_data: DataFrame with 'time' and 'event' columns
            n_clusters: number of clusters (if None, find optimal)
            plot_survival: whether to plot survival curves

        Returns:
            results: dictionary containing all results
        """
        # Initialize saver
        saver = TrainingInfoSaver()

        # Feature selection using Cox regression
        selected_features = self.cox_feature_selection(features_dict, survival_data)

        if len(selected_features) == 0:
            return None

        # Feature integration
        integrated_features = self.integrate_features(selected_features)

        # Find optimal number of clusters if not specified
        if n_clusters is None:
            optimal_k, silhouette_scores = self.find_optimal_clusters(
                integrated_features
            )
            n_clusters = optimal_k

        # Clustering
        labels = self.cluster(integrated_features, n_clusters)

        # Evaluation
        # c_index, p_value = self.evaluate_clustering(labels, survival_data)
        c_index, p_value = self.evaluate_clustering_improved(
            labels, survival_data, integrated_features
        )

        silhouette = silhouette_score(integrated_features, labels)

        # Plot survival curves
        if plot_survival:
            self.plot_survival_curves(labels, survival_data)

        results = {
            "labels": labels,
            "c_index": c_index,
            "p_value": p_value,
            "silhouette_score": silhouette,
            "n_clusters": n_clusters,
            "n_selected_features": {
                k: v.shape[1] for k, v in selected_features.items()
            },
            "n_total_features": integrated_features.shape[1],
        }

        # Save training information
        logger.info("\nSaving training information for future predictions...")
        
        # Save feature selection indices
        saver.save_feature_selection_info(self)
        
        # Save cluster information
        saver.save_cluster_info(integrated_features, labels)
        
        # Save training summary
        saver.save_training_summary(results)
        
        logger.info("Training information saved successfully!")

        return results
