import argparse
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.impute import KNNImputer

# Create a logger
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def remove_features(df, threshold=20):
    """
    Remove features (rows) with more than 20% missing values.
    """
    # If "NA" is stored as a string and not actual NaN, convert it
    df = df.replace("NA", pd.NA)

    # Calculate % of missing values per row
    missing_percent = df.isna().mean(axis=1) * 100

    # logger.info(f"Missing percentage per row: {missing_percent}")

    # Keep rows with missing percentage <= threshold
    cleaned_df = df[missing_percent <= threshold].reset_index(drop=True)

    return cleaned_df


def knnimpute(df, k=10, verbose=True):
    """
    Apply KNN imputation to a dataframe where:
    - First column contains feature names
    - Each column represents a sample/patient
    - Each row represents a feature
    """
    # Get feature names from first column
    feature_names = df.iloc[:, 0].copy()

    # Extract data (all columns except first)
    data = df.iloc[:, 1:].copy()

    # logger.info(df)

    # Check for missing values in data
    if verbose:
        total_missing = data.isna().sum().sum()
        logger.info(
            f"knnimpute - Total missing values before imputation: {total_missing}"
        )

    # Transpose the data (samples as rows, features as columns)
    data.index = feature_names
    data_transposed = data.T

    # logger.info(data_transposed)

    imputer = KNNImputer(n_neighbors=min(k, len(data_transposed) - 1))
    imputed_data = imputer.fit_transform(data_transposed)

    # The shape of imputed_data tells us how many features were kept
    kept_features_count = imputed_data.shape[1]

    if verbose:
        dropped_count = len(feature_names) - kept_features_count
        logger.info(
            f"knnimpute - KNNImputer dropped {dropped_count} features during imputation"
        )

    # Convert back to original format with the unique row IDs
    imputed_df = pd.DataFrame(
        imputed_data,
        columns=imputer.get_feature_names_out(),
        index=data_transposed.index,
    )

    # Transpose back to original orientation and convert to result
    # Simply reset_index and rename to avoid fragmentation warning
    result = imputed_df.T.reset_index().rename(columns={"index": "Features"})

    logger.info(f"knnimpute - Result:\n{result}")

    return result


def meanimpute(data):
    """
    Fill missing values in DataFrame with respective row means, preserving ID column.
    """
    # Check total missing values
    total_missing = data.isna().sum().sum()
    logger.info(f"meanimpute - Total missing values: {total_missing}")
    if total_missing == 0:
        return data

    # Split ID column and data
    id_column = data.iloc[:, 0:1]
    data_values = data.iloc[:, 1:].copy()

    # Calculate row means and fill NaNs
    for idx in data_values.index:
        row = data_values.loc[idx]
        row_mean = row.mean()
        data_values.loc[idx, row.isna()] = row_mean

    # Return combined result
    return pd.concat([id_column, data_values], axis=1)


def select_top_scoring_features(data, threshold=0.5, num_features=1000):
    """
    Select features based on expression difference of each feature: wi = (mi,1 - mi,0) / (sigmai,1 + sigmai,0)
    """
    # Extract numeric values (skip the first column if it's non-numeric like IDs)
    numeric_data = data.iloc[:, 1:].astype(float)

    # Dictionary to hold expression difference values
    row_scores = {}

    for idx in numeric_data.index:
        row = numeric_data.loc[idx]
        high_values = row[row >= threshold]
        low_values = row[row < threshold]

        # Compute mean and standard deviation for both groups
        mean_high = high_values.mean()
        mean_low = low_values.mean()
        std_high = high_values.std()
        std_low = low_values.std()

        # Calculate expression difference score
        denominator = (
            std_high + std_low if std_high + std_low != 0 else 1e-6
        )  # avoid division by zero
        score = (mean_high - mean_low) / denominator
        row_scores[idx] = score

    # Convert to Series and select top rows
    scores_series = pd.Series(row_scores)
    top_indices = scores_series.sort_values(ascending=False).head(num_features).index

    # Return the top rows from the original data
    selected_data = data.loc[top_indices].copy()
    return selected_data


def filter_by_sample_ids(df, samples):
    # Get the column names from samples that also exist in df
    sample_cols = [col for col in samples[0].values if col in df.columns]

    # Select the first column and the matching sample columns
    filtered_df = df[[df.columns[0]] + sample_cols]

    # Rename first column
    filtered_df = filtered_df.rename(columns={filtered_df.columns[0]: "Features"})

    return filtered_df


def run(input, type, num_features, output, fill_missing_method, sample_ids):
    # Load data input file path
    df = pd.read_csv(input, sep="\t")
    logger.info(f"Input shape: {df.shape}")

    # Load sample IDs
    samples = pd.read_csv(sample_ids, sep="\t", header=None)

    # Filter omic data by sample ids
    df = filter_by_sample_ids(df, samples)
    logger.info(f"Filtered shape: {df.shape}")

    # Remove features with more than 20% missing values for DNA methylation only
    if type == "dnameth":
        df = remove_features(df)
        logger.info(f"Feature removed shape: {df.shape}")

    # Fill missing values for DNA Methylation & CNV gene level
    if type == "dnameth" or type == "cnv":
        df = knnimpute(df) if fill_missing_method == "knnimpute" else meanimpute(df)

    # Select top scoring features
    result = select_top_scoring_features(df, num_features=num_features)

    # Export result
    result.to_csv(output, sep="\t", index=False)

    logger.info(f"Result shape: {result.shape}")
    logger.info(result.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Preprocess xena browser TCGA data"""
    )
    parser.add_argument(
        "-s",
        "--sample-ids",
        type=str,
        help="Sample IDs file path",
        required=False,
    )
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument("--output", type=str, help="Input file path", required=True)
    parser.add_argument(
        "--type",
        type=str,
        choices=["cnv", "dnameth", "mirna", "mrna"],
        help="Type of data to preprocess",
        required=True,
    )
    parser.add_argument(
        "--num-features",
        type=int,
        help="Number of features to select",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--fill-missing-method",
        type=str,
        choices=["knnimpute", "mean"],
        default="knnimpute",
        help="Fill missing method: KNNimpute or Mean",
    )
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    run(
        args.input,
        args.type,
        args.num_features,
        args.output,
        args.fill_missing_method,
        args.sample_ids,
    )
