import matplotlib.pyplot as plt
import numpy as np

from zia import BASE_PATH
from zia.config import read_config
import pandas as pd

if __name__ == "__main__":
    protein_order = ["CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4", "GS", "HE"]
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#f0f0f0', '#636363']
    config = read_config(BASE_PATH / "configuration.ini")
    df = pd.read_csv(config.reports_path / "lobule_percentiles.csv", sep=",", index_col=False)
    for species, species_df in df.groupby("species"):
        print(species)
        fig, ax = plt.subplots(dpi=300)
        ax: plt.Axes
        protein_gb = species_df.groupby("protein")

        for protein, color in zip(protein_order, colors):
            protein_df = protein_gb.get_group(protein)
            print(protein)
            expression_profiles = []
            percentages = None
            for a, lobule_df in protein_df.groupby(["subject", "roi", "lobule"]):
                if percentages is None:
                    percentages = lobule_df["percentage_max"].values
                expression_profiles.append(lobule_df["data"].values)

            expression_profiles = np.vstack(expression_profiles)

            mean = np.mean(expression_profiles, axis=0)
            std = np.std(expression_profiles, axis=0)
            ste = std / np.sqrt(expression_profiles.shape[0])

            ax.plot(percentages, mean, color=color)
            ax.fill_between(percentages, mean - 1.96 * ste, mean + 1.96 * ste, color=color, alpha=0.5)

            ax.set_title(species)
            ax.set_xlabel("percentile of max expression (%)")
            ax.set_ylabel("percentile of area occupied (%)")
            ax.set_xlim(left=0, right=100)

        plt.show()
