import argparse
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from skimage.filters import threshold_otsu


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

    # Keep rows with missing percentage <= threshold
    cleaned_df = df[missing_percent <= threshold].reset_index(drop=True)

    logger.info(
        f"Removed {len(df) - len(cleaned_df)} features with >{threshold}% missing values"
    )

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

    # Check for missing values in data
    if verbose:
        total_missing = data.isna().sum().sum()
        logger.info(
            f"knnimpute - Total missing values before imputation: {total_missing}"
        )

    # Transpose the data (samples as rows, features as columns)
    data.index = feature_names
    data_transposed = data.T

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
    result = imputed_df.T.reset_index().rename(columns={"index": "Features"})

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


def select_top_scoring_features_genewise(data, num_features=1000):
    """
    Select features based on Gene-wise weights as mentioned in the CtAE paper.
    Gene-wise weight = mean * standard deviation for each feature
    """
    # Extract numeric values (skip the first column if it's non-numeric like IDs)
    numeric_data = data.iloc[:, 1:].astype(float)

    # Calculate gene-wise weights
    row_scores = {}

    for idx in numeric_data.index:
        row = numeric_data.loc[idx]

        # Calculate mean and standard deviation
        mean_val = row.mean()
        std_val = row.std()

        # Gene-wise weight = mean * std (as per paper)
        score = abs(mean_val * std_val)  # Use absolute value to handle negative means
        row_scores[idx] = score

    # Convert to Series and select top rows
    scores_series = pd.Series(row_scores)
    top_indices = scores_series.sort_values(ascending=False).head(num_features).index

    # Return the top rows from the original data
    selected_data = data.loc[top_indices].copy()

    logger.info(f"Selected top {num_features} features using gene-wise weights")

    return selected_data


def select_top_scoring_features(data, num_features=1000, method="genewise"):
    """
    Select features based on specified method.
    """
    if method == "genewise":
        return select_top_scoring_features_genewise(data, num_features)
    else:
        # Original Otsu method
        numeric_data = data.iloc[:, 1:].astype(float)
        threshold = threshold_otsu(numeric_data.values.flatten())
        logger.info(f"Otsu threshold: {threshold}")

        row_scores = {}

        for idx in numeric_data.index:
            row = numeric_data.loc[idx]
            high_values = row[row >= threshold]
            low_values = row[row < threshold]

            mean_high = high_values.mean()
            mean_low = low_values.mean()
            std_high = high_values.std()
            std_low = low_values.std()

            denominator = std_high + std_low if std_high + std_low != 0 else 1e-6
            score = (mean_high - mean_low) / denominator
            row_scores[idx] = score

        scores_series = pd.Series(row_scores)
        top_indices = (
            scores_series.sort_values(ascending=False).head(num_features).index
        )
        selected_data = data.loc[top_indices].copy()

        return selected_data


def apply_minmax_normalization(df):
    """
    Apply min-max normalization to all features as per the paper.
    """
    # Get feature names from first column
    feature_names = df.iloc[:, 0].copy()

    # Extract numeric data
    numeric_data = df.iloc[:, 1:].astype(float)

    # Apply min-max scaling row-wise (each feature)
    scaler = MinMaxScaler()

    # Transpose to scale features (rows), then transpose back
    normalized_data = scaler.fit_transform(numeric_data.T).T

    # Create normalized dataframe
    normalized_df = pd.DataFrame(
        normalized_data, index=numeric_data.index, columns=numeric_data.columns
    )

    # Add feature names back
    result = pd.concat([feature_names, normalized_df], axis=1)

    logger.info("Applied min-max normalization to all features")

    return result


def reorder_features_by_correlation(df):
    """
    Reorder features based on Pearson correlation as described in the paper (Equations 6-8).
    """
    # Extract numeric data
    numeric_data = df.iloc[:, 1:].astype(float)

    # Calculate correlation matrix between features (rows)
    corr_matrix = numeric_data.T.corr()  # Transpose to get feature correlations

    # Calculate cumulative correlation for each feature as per paper
    p_values = []

    for i in range(len(corr_matrix)):
        # Get absolute correlations for this feature
        row_corr = np.abs(corr_matrix.iloc[i].values)

        # Calculate geometric mean of correlations (Equation 6)
        # Avoid zero by adding small epsilon
        row_corr[row_corr == 0] = 1e-10
        p_i = np.prod(row_corr) ** (1.0 / len(row_corr))
        p_values.append(p_i)

    # Create series with correlation scores
    p_series = pd.Series(p_values, index=corr_matrix.index)

    # Sort in descending order (Equation 8)
    sorted_indices = p_series.sort_values(ascending=False).index

    # Reorder the dataframe
    reordered_df = df.loc[sorted_indices].reset_index(drop=True)

    logger.info("Reordered features based on correlation coefficients")

    return reordered_df


def filter_by_sample_ids(df, samples):
    # Get the column names from samples that also exist in df
    sample_cols = [col for col in samples[0].values if col in df.columns]

    # Select the first column and the matching sample columns
    filtered_df = df[[df.columns[0]] + sample_cols]

    # Rename first column
    filtered_df = filtered_df.rename(columns={filtered_df.columns[0]: "Features"})

    return filtered_df


def run(
    input,
    type,
    num_features,
    output,
    fill_missing_method,
    sample_ids,
    feature_selection_method="genewise",
    apply_normalization=True,
    reorder_features=True,
):

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
    result = select_top_scoring_features(
        df, num_features=num_features, method=feature_selection_method
    )

    # Apply min-max normalization as per paper
    if apply_normalization:
        result = apply_minmax_normalization(result)

    # Reorder features based on correlation as per paper
    if reorder_features:
        result = reorder_features_by_correlation(result)

    # Export result
    result.to_csv(output, sep="\t", index=False)

    logger.info(f"Result shape: {result.shape}")
    logger.info(f"Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Preprocess xena browser TCGA data following CtAE paper"""
    )
    parser.add_argument(
        "-s",
        "--sample-ids",
        type=str,
        help="Sample IDs file path",
        required=False,
    )
    parser.add_argument("--input", type=str, help="Input file path", required=True)
    parser.add_argument("--output", type=str, help="Output file path", required=True)
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
    parser.add_argument(
        "--feature-selection",
        type=str,
        choices=["genewise", "otsu"],
        default="genewise",
        help="Feature selection method: genewise (as per paper) or otsu",
    )
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Skip min-max normalization",
    )
    parser.add_argument(
        "--no-reorder",
        action="store_true",
        help="Skip feature reordering by correlation",
    )

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    # Default feature counts from paper
    default_features = {"mrna": 2000, "cnv": 1500, "dnameth": 1000, "mirna": 300}

    # Use paper defaults if not specified
    num_features = args.num_features
    if num_features == -1:  # Use -1 as flag for paper defaults
        num_features = default_features.get(args.type, 1000)
        logger.info(f"Using paper default: {num_features} features for {args.type}")

    run(
        args.input,
        args.type,
        num_features,
        args.output,
        args.fill_missing_method,
        args.sample_ids,
        feature_selection_method=args.feature_selection,
        apply_normalization=not args.no_normalization,
        reorder_features=not args.no_reorder,
    )
