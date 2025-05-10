import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt
import logging

# Create a logger
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def load_data():
    # Load the survival data
    survival_data = pd.read_csv("data/survival.tsv", sep="\t")

    # Load the omics data
    omics_cnv = pd.read_csv("cnv.clean.tsv", sep="\t", index_col=0)
    omics_dnameth = pd.read_csv("dnameth.mean.clean.tsv", sep="\t", index_col=0)
    omics_mrna = pd.read_csv("mrna.clean.tsv", sep="\t", index_col=0)
    omics_mirna = pd.read_csv("mirna.clean.tsv", sep="\t", index_col=0)

    # Merge the omics data
    # multi_omics_data = pd.concat(
    #     [omics_cnv, omics_dnameth, omics_mrna, omics_mirna], axis=1
    # )

    return omics_mirna, survival_data


def run():
    # Load data
    multi_omics_data, survival_data = load_data()

    # Merge the omics data with the survival data (matching patient IDs)
    merged_data = pd.merge(
        multi_omics_data.T,
        survival_data[["sample", "OS.time", "OS"]],
        left_index=True,
        right_on="sample",
        how="inner",
    )

    logger.info(f"Merged data shape: {merged_data.shape}")

    logger.info(merged_data["sample"])

    # Drop non-numeric columns
    merged_data = merged_data.drop(columns=["sample"])
    logger.info(merged_data.head())

    cph = CoxPHFitter()
    cph.fit(merged_data, duration_col="OS.time", event_col="OS")

    # Display the summary of the Cox Proportional Hazards model
    cph.print_summary()

    # Draw plots
    final_df = merged_data.copy()
    final_df["risk_score"] = cph.predict_partial_hazard(merged_data)
    median_risk_score = final_df["risk_score"].median()
    final_df["risk_group"] = np.where(
        final_df["risk_score"] >= median_risk_score, "High-risk", "Low-risk"
    )

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(8, 6))

    for group in final_df['risk_group'].unique():
        group_data = final_df[final_df['risk_group'] == group]
        
        # Fit the Kaplan-Meier estimator to the data for this group
        kmf.fit(group_data['OS.time'], event_observed=group_data['OS'], label=group)
        
        # Plot the Kaplan-Meier curve
        kmf.plot()

    # Customize the plot
    plt.title('Kaplan-Meier Curve Based on Risk Score from Cox Model')
    plt.xlabel('Time (OS.time)')
    plt.ylabel('Survival Probability')
    plt.legend()

    # Save the plot to a file (e.g., as a PNG)
    plt.savefig('kaplan_meier_curve.png', format='png')  # You can change the file name and format here
    plt.show()



run()
