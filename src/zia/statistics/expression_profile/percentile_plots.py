import numpy as np

from zia import BASE_PATH
from zia.config import read_config
import pandas as pd

if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    df = pd.read_csv(config.reports_path / "lobule_percentiles.csv", sep=",", index_col=False)

    for species, species_df in df.groupby("species"):
        print(species)
        for protein, protein_df in species_df.groupby("protein"):
            print(protein)
            expression_profiles = []
            percentages = None
            for a, lobule_df in protein_df.groupby(["subject", "roi", "lobule"]):
                if percentages is None:
                    percentages = lobule_df["percentage_max"].values
                expression_profiles.append(lobule_df["data"].values)

            expression_profiles = np.vstack(expression_profiles)
            print(expression_profiles.shape)
