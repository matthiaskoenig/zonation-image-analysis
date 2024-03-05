import matplotlib.pyplot as plt
import numpy as np

from zia import BASE_PATH
from zia.config import read_config
import pandas as pd

if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "expression_profiles"
    report_path.mkdir(exist_ok=True, parents=True)
    protein_order = ["CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4", "GS", "HE"]
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#f0f0f0', '#636363']
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

            expression_profiles = np.vstack(expression_profiles) * 100

            mean = np.median(expression_profiles, axis=0)
            std = np.std(expression_profiles, axis=0)
            ste = std / np.sqrt(expression_profiles.shape[0])
            q1 = np.percentile(expression_profiles, q=25, axis=0)
            q3 = np.percentile(expression_profiles, q=75, axis=0)

            ax.plot(percentages, mean, color=color, label=protein, linewidth=3)
            ax.fill_between(percentages, q1, q3, color=color, alpha=0.3, linewidth=0.0)

            ax.set_title(species)
            ax.set_xlabel("normalized expression (%)")
            ax.set_ylabel("area of expression (%)")
            ax.set_xlim(left=0, right=100)
            ax.set_ylim(bottom=-3, top=103)
            ax.legend(frameon=False)

        fig.savefig(report_path / f"expression_profile_{species}.png")
        plt.show()
