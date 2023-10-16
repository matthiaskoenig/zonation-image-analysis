import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.stats as st

from zia.statistics.data_provider import SlideStatsProvider


def qq_plot(df: pd.DataFrame, attr: str, log=False):
    gb = df.groupby("species")

    for species, color in zip(SlideStatsProvider.species_order, SlideStatsProvider.colors):
        species_df = gb.get_group(species)
        x = species_df[attr]
        if log:
            np.log10(x)

        # print(min(x))
        # x = x[x > 1.5]
        fig, ax = plt.subplots(dpi=300)

        st.probplot(x, dist="norm", plot=ax)

        ax: plt.Axes
        ax.set_title(f"{species} ({attr})")

        plt.show()


if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()

    qq_plot(df, "area", log=True)
    qq_plot(df, "perimeter", log=True)
    qq_plot(df, "compactness", log=False)
