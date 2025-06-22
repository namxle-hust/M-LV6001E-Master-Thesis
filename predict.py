"""
Script to classify new patients using trained SubtypeCtAE models
"""

import numpy as np
import pandas as pd
import torch
import pickle
import os
import logging
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing classes
from cae_model_v2 import ContractiveAutoEncoder, OmicsDataset
from subtype_ctae import SubtypeCtAE

# Create a logger
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SubtypeCtAEPredictor:
    """
    Classifier for new patients using trained SubtypeCtAE models
    """
    
    def __init__(self, model_dir="models"):
        """
        Args:
            model_dir: directory containing trained models and metadata
        """
        self.model_dir = model_dir
        self.cae_models = {}
        self.scalers = {}
        self.selected_feature_indices = {}
        self.cluster_centroids = None
        self.cluster_labels_training = None
        self.metadata = {}
        
    def load_trained_models(self):
        """
        Load all trained CAE models, scalers, and metadata
        """
        logger.info("Loading trained models...")
        
        omics_types = ["mrna", "cnv", "mirna", "dnameth"]
        
        for omics in omics_types:
            # Load CAE model
            model_path = os.path.join(self.model_dir, f"{omics}_cae_model.pth")
            metadata_path = os.path.join(self.model_dir, f"{omics}_metadata.pth")
            scaler_path = os.path.join(self.model_dir, f"{omics}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                # Load metadata
                metadata = torch.load(metadata_path)
                self.metadata[omics] = metadata
                
                # Initialize and load CAE model
                model = ContractiveAutoEncoder(
                    input_dim=metadata['input_dim'],
                    hidden_dims=metadata['hidden_dims'],
                    activation=metadata.get('activation', 'relu'),
                    use_batch_norm=metadata.get('batch_norm', False)
                )
                model.load_state_dict(torch.load(model_path))
                model.eval()
                self.cae_models[omics] = model
                
                logger.info(f"Loaded CAE model for {omics}")
            else:
                logger.warning(f"CAE model files not found for {omics}")
            
            # Load scaler
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[omics] = pickle.load(f)
                logger.info(f"Loaded scaler for {omics}")
            else:
                logger.warning(f"Scaler not found for {omics}")
    
    def load_feature_selection_info(self, selection_file="feature_selection_info.pkl"):
        """
        Load information about which features were selected during training
        """
        selection_path = os.path.join(self.model_dir, selection_file)
        
        if os.path.exists(selection_path):
            with open(selection_path, 'rb') as f:
                self.selected_feature_indices = pickle.load(f)
            logger.info("Loaded feature selection information")
        else:
            logger.warning("Feature selection info not found - will use all features")
    
    def load_cluster_info(self, cluster_file="cluster_info.pkl"):
        """
        Load cluster centroids and training labels
        """
        cluster_path = os.path.join(self.model_dir, cluster_file)
        
        if os.path.exists(cluster_path):
            with open(cluster_path, 'rb') as f:
                cluster_info = pickle.load(f)
                self.cluster_centroids = cluster_info['centroids']
                self.cluster_labels_training = cluster_info['labels']
            logger.info("Loaded cluster information")
        else:
            logger.warning("Cluster info not found - will need to retrain")
    
    def extract_features_new_patient(self, omics_data_paths):
        """
        Extract features from new patient omics data using trained CAE models
        
        Args:
            omics_data_paths: dict with omics names as keys and file paths as values
                             e.g., {'mrna': 'new_patient_mrna.tsv', 'cnv': 'new_patient_cnv.tsv'}
        
        Returns:
            features_dict: dictionary of extracted features
        """
        logger.info("Extracting features for new patient(s)...")
        
        features_dict = {}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for omics, data_path in omics_data_paths.items():
            if omics not in self.cae_models:
                logger.warning(f"No trained model available for {omics}")
                continue
                
            # Load and preprocess data
            data = pd.read_csv(data_path, sep='\t', index_col=0).T
            data_array = data.values.astype(np.float32)
            
            # Apply the same scaler used during training
            if omics in self.scalers:
                data_array = self.scalers[omics].transform(data_array)
            
            # Convert to tensor and extract features
            data_tensor = torch.from_numpy(data_array).to(device)
            model = self.cae_models[omics].to(device)
            
            with torch.no_grad():
                features = model.get_encoder_output(data_tensor)
                features_numpy = features.cpu().numpy()
            
            features_dict[omics] = features_numpy
            logger.info(f"Extracted {features_numpy.shape} features for {omics}")
        
        return features_dict
    
    def apply_feature_selection(self, features_dict):
        """
        Apply the same feature selection that was used during training
        
        Args:
            features_dict: dictionary of extracted features
        
        Returns:
            selected_features_dict: dictionary of selected features
        """
        logger.info("Applying feature selection...")
        
        selected_features_dict = {}
        
        for omics, features in features_dict.items():
            if omics in self.selected_feature_indices:
                selected_indices = self.selected_feature_indices[omics]
                selected_features = features[:, selected_indices]
                selected_features_dict[omics] = selected_features
                logger.info(f"Selected {selected_features.shape[1]}/{features.shape[1]} features for {omics}")
            else:
                # If no selection info, use all features
                selected_features_dict[omics] = features
                logger.info(f"Using all {features.shape[1]} features for {omics} (no selection info)")
        
        return selected_features_dict
    
    def integrate_features(self, selected_features_dict):
        """
        Integrate features from different omics types
        
        Args:
            selected_features_dict: dictionary of selected features
        
        Returns:
            integrated_features: concatenated feature matrix
        """
        logger.info("Integrating multi-omics features...")
        
        feature_list = []
        for omics, features in selected_features_dict.items():
            feature_list.append(features)
            logger.info(f"{omics}: {features.shape[1]} features")
        
        integrated_features = np.concatenate(feature_list, axis=1)
        logger.info(f"Total integrated features: {integrated_features.shape[1]}")
        
        return integrated_features
    
    def predict_subtypes(self, integrated_features):
        """
        Predict cancer subtypes for new patients using cluster centroids
        
        Args:
            integrated_features: integrated feature matrix
        
        Returns:
            predicted_labels: predicted cluster labels
            distances: distances to each cluster centroid
        """
        logger.info("Predicting cancer subtypes...")
        
        if self.cluster_centroids is None:
            raise ValueError("Cluster centroids not loaded. Please load cluster information first.")
        
        # Calculate distances to each cluster centroid
        distances = pairwise_distances(integrated_features, self.cluster_centroids)
        
        # Assign to closest cluster
        predicted_labels = np.argmin(distances, axis=1)
        
        logger.info(f"Predicted subtypes for {len(predicted_labels)} patients")
        
        return predicted_labels, distances
    
    def predict_survival_risk(self, predicted_labels, distances):
        """
        Estimate survival risk based on cluster assignment and distance
        
        Args:
            predicted_labels: predicted cluster labels
            distances: distances to cluster centroids
        
        Returns:
            risk_scores: estimated survival risk scores
        """
        logger.info("Estimating survival risk...")
        
        # Simple risk scoring based on cluster assignment and distance
        # This is a simplified approach - you might want to use the actual
        # Cox model from training for more accurate risk assessment
        
        risk_scores = np.zeros(len(predicted_labels))
        
        for i, (label, dist_row) in enumerate(zip(predicted_labels, distances)):
            # Base risk from cluster assignment (assuming higher cluster = higher risk)
            base_risk = label + 1
            
            # Adjust based on distance to centroid (further = more uncertain)
            distance_to_assigned = dist_row[label]
            uncertainty_factor = 1 + (distance_to_assigned / np.mean(dist_row))
            
            risk_scores[i] = base_risk * uncertainty_factor
        
        return risk_scores
    
    def generate_prediction_report(self, patient_ids, predicted_labels, risk_scores, 
                                 distances, output_file="prediction_report.html"):
        """
        Generate a comprehensive prediction report
        
        Args:
            patient_ids: list of patient identifiers
            predicted_labels: predicted cluster labels
            risk_scores: survival risk scores
            distances: distances to cluster centroids
            output_file: output HTML file path
        """
        logger.info("Generating prediction report...")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Patient_ID': patient_ids,
            'Predicted_Subtype': predicted_labels + 1,  # 1-indexed
            'Risk_Score': risk_scores,
            'Confidence': 1 / (1 + np.min(distances, axis=1))  # Higher confidence = closer to centroid
        })
        
        # Add distance to each cluster
        for i in range(distances.shape[1]):
            results_df[f'Distance_to_Cluster_{i+1}'] = distances[:, i]
        
        # Save detailed results
        results_df.to_csv(output_file.replace('.html', '.csv'), index=False)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>SubtypeCtAE Prediction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; margin: 20px 0; }}
                .high-risk {{ background-color: #ffebee; }}
                .medium-risk {{ background-color: #fff3e0; }}
                .low-risk {{ background-color: #e8f5e8; }}
            </style>
        </head>
        <body>
            <h1>SubtypeCtAE Cancer Subtype Prediction Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Number of patients analyzed:</strong> {len(patient_ids)}</p>
                <p><strong>Subtype distribution:</strong></p>
                <ul>
        """
        
        # Add subtype distribution
        for subtype in np.unique(predicted_labels):
            count = np.sum(predicted_labels == subtype)
            html_content += f"<li>Subtype {subtype + 1}: {count} patients</li>"
        
        html_content += """
                </ul>
            </div>
            
            <h2>Individual Patient Results</h2>
            <table>
                <tr>
                    <th>Patient ID</th>
                    <th>Predicted Subtype</th>
                    <th>Risk Score</th>
                    <th>Confidence</th>
                    <th>Risk Level</th>
                </tr>
        """
        
        # Add individual results
        for _, row in results_df.iterrows():
            risk_level = "High" if row['Risk_Score'] > np.percentile(risk_scores, 66) else \
                        "Medium" if row['Risk_Score'] > np.percentile(risk_scores, 33) else "Low"
            
            risk_class = f"{risk_level.lower()}-risk"
            
            html_content += f"""
                <tr class="{risk_class}">
                    <td>{row['Patient_ID']}</td>
                    <td>{row['Predicted_Subtype']}</td>
                    <td>{row['Risk_Score']:.3f}</td>
                    <td>{row['Confidence']:.3f}</td>
                    <td>{risk_level}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="summary">
                <h2>Interpretation Notes</h2>
                <ul>
                    <li><strong>Predicted Subtype:</strong> Cancer subtype classification (1-indexed)</li>
                    <li><strong>Risk Score:</strong> Estimated survival risk (higher = worse prognosis)</li>
                    <li><strong>Confidence:</strong> Prediction confidence (higher = more reliable)</li>
                    <li><strong>Risk Level:</strong> Categorized risk level based on score distribution</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Prediction report saved to {output_file}")
        
        return results_df
    
    def plot_prediction_visualization(self, predicted_labels, risk_scores, distances, 
                                    save_path="prediction_visualization.png"):
        """
        Create visualization of predictions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subtype distribution
        axes[0, 0].hist(predicted_labels + 1, bins=range(1, max(predicted_labels) + 3), 
                       alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Predicted Subtype')
        axes[0, 0].set_ylabel('Number of Patients')
        axes[0, 0].set_title('Subtype Distribution')
        
        # Risk score distribution
        axes[0, 1].hist(risk_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Risk Score')
        axes[0, 1].set_ylabel('Number of Patients')
        axes[0, 1].set_title('Risk Score Distribution')
        
        # Risk vs Subtype
        axes[1, 0].boxplot([risk_scores[predicted_labels == i] 
                           for i in np.unique(predicted_labels)])
        axes[1, 0].set_xlabel('Predicted Subtype')
        axes[1, 0].set_ylabel('Risk Score')
        axes[1, 0].set_title('Risk Scores by Subtype')
        
        # Distance heatmap
        im = axes[1, 1].imshow(distances.T, aspect='auto', cmap='viridis')
        axes[1, 1].set_xlabel('Patient Index')
        axes[1, 1].set_ylabel('Cluster')
        axes[1, 1].set_title('Distance to Cluster Centroids')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prediction visualization saved to {save_path}")
    
    def predict_new_patients(self, omics_data_paths, patient_ids=None, 
                           generate_report=True, create_visualization=True):
        """
        Complete pipeline to predict cancer subtypes for new patients
        
        Args:
            omics_data_paths: dict with omics data file paths
            patient_ids: list of patient IDs (if None, will generate)
            generate_report: whether to generate HTML report
            create_visualization: whether to create visualizations
        
        Returns:
            results: dictionary with all prediction results
        """
        logger.info("=" * 50)
        logger.info("PREDICTING CANCER SUBTYPES FOR NEW PATIENTS")
        logger.info("=" * 50)
        
        # Extract features
        features_dict = self.extract_features_new_patient(omics_data_paths)
        
        # Apply feature selection
        selected_features_dict = self.apply_feature_selection(features_dict)
        
        # Integrate features
        integrated_features = self.integrate_features(selected_features_dict)
        
        # Predict subtypes
        predicted_labels, distances = self.predict_subtypes(integrated_features)
        
        # Estimate survival risk
        risk_scores = self.predict_survival_risk(predicted_labels, distances)
        
        # Generate patient IDs if not provided
        if patient_ids is None:
            patient_ids = [f"Patient_{i+1}" for i in range(len(predicted_labels))]
        
        # Generate report
        results_df = None
        if generate_report:
            results_df = self.generate_prediction_report(
                patient_ids, predicted_labels, risk_scores, distances
            )
        
        # Create visualization
        if create_visualization:
            self.plot_prediction_visualization(
                predicted_labels, risk_scores, distances
            )
        
        # Return results
        results = {
            'patient_ids': patient_ids,
            'predicted_labels': predicted_labels,
            'risk_scores': risk_scores,
            'distances': distances,
            'integrated_features': integrated_features,
            'results_df': results_df
        }
        
        logger.info("Prediction complete!")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Predict cancer subtypes for new patients')
    parser.add_argument('--model-dir', type=str, default='models', 
                       help='Directory containing trained models')
    parser.add_argument('--mrna', type=str, help='mRNA data file for new patients')
    parser.add_argument('--cnv', type=str, help='CNV data file for new patients')
    parser.add_argument('--mirna', type=str, help='miRNA data file for new patients')
    parser.add_argument('--dnameth', type=str, help='DNA methylation data file for new patients')
    parser.add_argument('--patient-ids', type=str, help='File with patient IDs (one per line)')
    parser.add_argument('--output', type=str, default='predictions', 
                       help='Output prefix for results')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = SubtypeCtAEPredictor(model_dir=args.model_dir)
    
    # Load trained models
    predictor.load_trained_models()
    predictor.load_feature_selection_info()
    predictor.load_cluster_info()
    
    # Prepare omics data paths
    omics_data_paths = {}
    if args.mrna:
        omics_data_paths['mrna'] = args.mrna
    if args.cnv:
        omics_data_paths['cnv'] = args.cnv
    if args.mirna:
        omics_data_paths['mirna'] = args.mirna
    if args.dnameth:
        omics_data_paths['dnameth'] = args.dnameth
    
    if not omics_data_paths:
        logger.error("No omics data files provided!")
        return
    
    # Load patient IDs if provided
    patient_ids = None
    if args.patient_ids:
        with open(args.patient_ids, 'r') as f:
            patient_ids = [line.strip() for line in f]
    
    # Run prediction
    results = predictor.predict_new_patients(
        omics_data_paths=omics_data_paths,
        patient_ids=patient_ids
    )
    
    logger.info(f"Predictions completed for {len(results['patient_ids'])} patients")


if __name__ == "__main__":
    main()