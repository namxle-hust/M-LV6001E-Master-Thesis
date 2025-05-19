import argparse
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Create a logger
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

data_dir = "pp2"

output_file = "sample_ids.txt"


def load_data():
    # Load the survival data
    survival_data = pd.read_csv("data/survival.tsv", sep="\t")

    # Load the omics data
    omics_cnv = pd.read_csv(f"{data_dir}/cnv.clean.tsv", sep="\t", index_col=0)
    omics_dnameth = pd.read_csv(f"{data_dir}/dnameth.clean.tsv", sep="\t", index_col=0)
    omics_mrna = pd.read_csv(f"{data_dir}/mrna.clean.tsv", sep="\t", index_col=0)
    omics_mirna = pd.read_csv(f"{data_dir}/mirna.clean.tsv", sep="\t", index_col=0)

    # Get common columns across all DataFrames
    common_columns = (
        set(omics_cnv.columns)
        .intersection(omics_dnameth.columns)
        .intersection(omics_mrna.columns)
        .intersection(omics_mirna.columns)
    )

    # Get all common columns
    all_common_columns = common_columns.intersection(survival_data["sample"].values)

    logger.info(f"Common columns across all DataFrames: {len(common_columns)}")

    # Get the patient/sample IDs from survival data
    patient_ids = set(survival_data["sample"].values)

    # Get columns in common_columns that do not exist in survival_data["sample"].values
    columns_not_in_survival_data = common_columns - patient_ids

    logger.info(
        f"Columns in common_columns but not in survival_data: {len(columns_not_in_survival_data)}"
    )
    logger.info(columns_not_in_survival_data)

    # Write the IDs to a text file
    with open(output_file, "w") as file:
        for column in all_common_columns:
            file.write(f"{column}\n")  # Write each column ID on a new line


def run():
    load_data()


run()
