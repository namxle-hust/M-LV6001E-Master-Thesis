import argparse
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

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


def knnimpute(data, k=10):
    """
    Fill missing values in microarray data using the KNNimpute method as described in:
    "Missing value estimation methods for DNA microarrays" by Troyanskaya et al.

    The KNN-based method selects genes with expression profiles similar to the gene of interest
    to impute missing values using a weighted average based on Euclidean distance.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing gene expression data with missing values.
        Rows are genes, columns are experiments/arrays.
        First column is assumed to be a gene ID.

    k : int, default=10
        Number of nearest neighbors to use for imputation.
        Optimal value is typically between 10-20 as per the paper.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing values imputed using KNNimpute method.
    """

    # Check total missing values
    total_missing = data.isna().sum().sum()
    print(f"Total missing values: {total_missing}")

    if total_missing == 0:
        return data.copy()  # No missing values to fill

    # Split ID column and data
    id_column = data.iloc[:, 0:1]
    data_values = data.iloc[:, 1:].copy()

    data_matrix = data_values.values

    # Get the indices of missing values
    has_missing = data_values.isna().any(axis=1)
    missing_rows = np.where(has_missing)[0]

    # If no missing values, return the original data
    if len(missing_rows) == 0:
        return data.copy()

    # Create a copy of data_values for imputation
    imputed_data = data_values.copy()

    # For each id with missing values
    for idx in missing_rows:
        # Find which columns have missing values for this id
        missing_cols = np.where(np.isnan(data_matrix[idx, :]))[0]

        for col in missing_cols:
            # Find ids that have this value present
            valid_ids = ~np.isnan(data_matrix[:, col])

            # We need a reference expression profile for similarity comparison
            # Create a reference profile by excluding the missing columns
            id_profile = data_matrix[idx, :]

            # Create a mask for valid columns in this id (exclude current missing column)
            valid_cols = ~np.isnan(id_profile)
            valid_cols[col] = False  # Exclude the current column with missing value

            # Skip if no valid columns exist for comparison
            if not np.any(valid_cols):
                continue

            # Calculate Euclidean distances between this id and all other ids
            # using only columns that are valid in the reference id
            distances = []
            for i in range(len(data_matrix)):
                if (
                    valid_ids[i] and i != idx
                ):  # Has value in missing column and is not the id itself
                    # Calculate Euclidean distance using only valid columns
                    dist = np.sqrt(
                        np.sum(
                            (id_profile[valid_cols] - data_matrix[i, valid_cols]) ** 2
                        )
                    )
                    distances.append((i, dist))

            # Sort by distance and select k nearest neighbors
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:k]

            # If no neighbors found, continue to next column
            if not neighbors:
                continue

            # Calculate weighted average of the k nearest neighbors
            # Weight is inversely proportional to the distance
            sum_weighted_values = 0
            sum_weights = 0

            for neighbor_idx, dist in neighbors:
                # Avoid division by zero by adding a small value
                weight = 1.0 / (dist + 1e-6)
                sum_weighted_values += weight * data_values.iloc[neighbor_idx, col]
                sum_weights += weight

            # Impute the missing value
            if sum_weights > 0:
                imputed_value = sum_weighted_values / sum_weights
                imputed_data.iloc[idx, col] = imputed_value

    # Return the imputed data with the ID column
    result = pd.concat([id_column, imputed_data], axis=1)
    return result


def fill_missing_data(data):
    """
    Fill missing values in DataFrame with respective row means, preserving ID column.
    """
    # Check total missing values
    total_missing = data.isna().sum().sum()
    print(f"Total missing values: {total_missing}")
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


def fill_missing_cnv_data(data):
    # Print total missing values
    total_missing = data.isna().sum().sum()
    print("Total missing values:", total_missing)

    filled_data = data.fillna(2)

    return filled_data


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


def run(path, type, num_features, output, fill_missing_method, survival_path):
    # Load data file path
    df = pd.read_csv(path, sep="\t")
    logger.info(f"Data shape: {df.shape}")

    # # Load survival data
    # survival_df = pd.read_csv(survival_path, sep="\t", header=None)

    # # Create a dictionary of new columns to add
    # new_columns = {}
    # for sample in survival_df[0]:  # Assuming the first column contains column names
    #     if sample not in df.columns:
    #         new_columns[sample] = np.nan  # Collect columns to add

    # # Add all new columns at once to avoid fragmentation
    # if new_columns:
    #     # Create a DataFrame with the new columns
    #     new_df = pd.DataFrame(new_columns, index=df.index)

    #     # Concatenate with original DataFrame
    #     df = pd.concat([df, new_df], axis=1)

    #     # Optional: Create a copy to defragment the DataFrame
    #     df = df.copy()

    # logger.info(f"df.shape: {df.shape}")
    # logger.info(df.head(100))
    # logger.info(f"Missing samples: {len(new_columns.keys())}")

    # Remove features with more than 20% missing values
    df = remove_features(df)
    logger.info(f"df.shape: {df.shape}")

    # Fill missing values
    df = (
        fill_missing_cnv_data(df)
        if type == "cnv"
        else (
            knnimpute(df)
            if fill_missing_method == "knnimpute"
            else fill_missing_data(df)
        )
    )

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
        "-s", "--survival-path", type=str, help="Survival data file path", required=False
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
        args.survival_path,
    )
