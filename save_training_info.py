"""
Script to save training information needed for predicting new patients
This should be called after training the SubtypeCtAE model
"""

import numpy as np
import pickle
import os
import logging
from sklearn.cluster import KMeans

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
    
    def save_feature_selection_info(self, subtypectae_model, filename="feature_selection_info.pkl"):
        """
        Save feature selection information from trained SubtypeCtAE model
        
        Args:
            subtypectae_model: trained SubtypeCtAE instance
            filename: output filename
        """
        logger.info("Saving feature selection information...")
        
        # Extract selected feature indices for each omics type
        selected_indices = {}
        
        if hasattr(subtypectae_model, 'selected_features') and subtypectae_model.selected_features:
            # Get the indices by comparing with original features
            # This assumes you have access to the original features
            # You might need to modify SubtypeCtAE to store these indices directly
            
            # For now, we'll create a placeholder - you should modify SubtypeCtAE 
            # to store the actual indices during cox_feature_selection
            for omics_name, selected_features in subtypectae_model.selected_features.items():
                # This is a placeholder - replace with actual indices
                selected_indices[omics_name] = np.arange(selected_features.shape[1])
                logger.info(f"Saved {len(selected_indices[omics_name])} feature indices for {omics_name}")
        
        # Save to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'wb') as f:
            pickle.dump(selected_indices, f)
        
        logger.info(f"Feature selection info saved to {output_path}")
        
        return selected_indices
    
    def save_cluster_info(self, integrated_features, cluster_labels, filename="cluster_info.pkl"):
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
            'centroids': centroids,
            'labels': cluster_labels,
            'n_clusters': n_clusters
        }
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'wb') as f:
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
            'c_index': results.get('c_index', None),
            'p_value': results.get('p_value', None),
            'silhouette_score': results.get('silhouette_score', None),
            'n_clusters': results.get('n_clusters', None),
            'n_selected_features': results.get('n_selected_features', {}),
            'n_total_features': results.get('n_total_features', None)
        }
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'wb') as f:
            pickle.dump(training_summary, f)
        
        logger.info(f"Training summary saved to {output_path}")
        
        return training_summary


# Modified SubtypeCtAE class to save feature indices
class SubtypeCtAEWithSaving:
    """
    Extended SubtypeCtAE that saves feature selection indices during training
    """
    
    def __init__(self, p_value_threshold=0.01):
        from subtype_ctae import SubtypeCtAE
        self.base_model = SubtypeCtAE(p_value_threshold)
        self.selected_feature_indices = {}
        self.original_features = {}
    
    def cox_feature_selection_with_saving(self, features_dict, survival_data):
        """
        Enhanced Cox feature selection that saves feature indices
        """
        from lifelines import CoxPHFitter
        
        logger.info("Performing Cox regression with feature index saving...")
        
        selected_features = {}
        
        for omics_name, features in features_dict.items():
            logger.info(f"\nProcessing {omics_name} features...")
            
            # Store original features
            self.original_features[omics_name] = features
            
            n_features = features.shape[1]
            p_values = []
            
            # Perform univariate Cox regression for each feature
            for i in range(n_features):
                # Create DataFrame for Cox regression
                cox_data = pd.DataFrame({
                    'time': survival_data['time'],
                    'event': survival_data['event'],
                    'feature': features[:, i]
                })
                
                # Fit Cox model
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col='time', event_col='event', show_progress=False)
                
                # Get p-value
                p_value = cph.summary.loc['feature', 'p']
                p_values.append(p_value)
            
            # Select features with p-value < threshold
            p_values = np.array(p_values)
            selected_indices = np.where(p_values < self.base_model.p_value_threshold)[0]
            
            if len(selected_indices) > 0:
                selected_features[omics_name] = features[:, selected_indices]
                self.selected_feature_indices[omics_name] = selected_indices
                logger.info(f"Selected {len(selected_indices)}/{n_features} features")
            else:
                logger.info(f"Warning: No features selected for {omics_name}")
                # If no features selected, use all features
                selected_features[omics_name] = features
                self.selected_feature_indices[omics_name] = np.arange(n_features)
        
        self.base_model.selected_features = selected_features
        return selected_features
    
    def fit_predict_with_saving(self, features_dict, survival_data, n_clusters=None, 
                               plot_survival=True, save_dir="models"):
        """
        Complete pipeline with automatic saving of training information
        """
        # Initialize saver
        saver = TrainingInfoSaver(save_dir)
        
        # Feature selection with saving
        selected_features = self.cox_feature_selection_with_saving(features_dict, survival_data)
        
        if len(selected_features) == 0:
            return None
        
        # Feature integration
        integrated_features = self.base_model.integrate_features(selected_features)
        
        # Find optimal clusters if needed
        if n_clusters is None:
            optimal_k, silhouette_scores = self.base_model.find_optimal_clusters(integrated_features)
            n_clusters = optimal_k
        
        # Clustering
        labels = self.base_model.cluster(integrated_features, n_clusters)
        
        # Evaluation
        c_index, p_value = self.base_model.evaluate_clustering_improved(
            labels, survival_data, integrated_features
        )
        
        silhouette = self.base_model.silhouette_score(integrated_features, labels)
        
        # Plot survival curves
        if plot_survival:
            self.base_model.plot_survival_curves(labels, survival_data)
        
        # Prepare results
        results = {
            'labels': labels,
            'c_index': c_index,
            'p_value': p_value,
            'silhouette_score': silhouette,
            'n_clusters': n_clusters,
            'n_selected_features': {k: v.shape[1] for k, v in selected_features.items()},
            'n_total_features': integrated_features.shape[1]
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


# Example usage function
def save_training_info_from_existing_results(
    features_dict, 
    survival_data, 
    subtypectae_results, 
    integrated_features,
    output_dir="models"
):
    """
    Save training information from existing SubtypeCtAE results
    Use this if you already have training results and want to save info for prediction
    
    Args:
        features_dict: original features dictionary
        survival_data: survival data used for training
        subtypectae_results: results from SubtypeCtAE.fit_predict()
        integrated_features: integrated features used for clustering
        output_dir: output directory
    """
    
    saver = TrainingInfoSaver(output_dir)
    
    # Save cluster information
    cluster_info = saver.save_cluster_info(
        integrated_features, 
        subtypectae_results['labels']
    )
    
    # Save training summary
    training_summary = saver.save_training_summary(subtypectae_results)
    
    # Note: For feature selection info, you'll need to re-run the Cox regression
    # or modify your training code to save the indices during the original training
    logger.info("Training info saved. Note: Feature selection indices need to be saved during training.")
    
    return cluster_info, training_summary


if __name__ == "__main__":
    # Example of how to use this with your existing training pipeline
    import pandas as pd
    
    # This would be called in your main training script instead of the regular SubtypeCtAE
    
    # Load your data (example)
    # features_dict = load_cae_features_from_files()
    # survival_data = load_survival_data("data/survival.filtered.tsv")
    
    # Use the enhanced model
    # model = SubtypeCtAEWithSaving(p_value_threshold=0.01)
    # results = model.fit_predict_with_saving(
    #     features_dict, 
    #     survival_data, 
    #     n_clusters=3,
    #     save_dir="models"
    # )
    
    logger.info("Training info saver ready. Use SubtypeCtAEWithSaving in your training pipeline.")