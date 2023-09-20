from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zia.processing.lobulus_statistics import SlideStats


def visualize_slide_stats(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 1, dpi=600)
    ax: plt.Axes
    ax.hist(np.log(df["area"]), bins="sqrt")
    ax.set_xlabel("log area")
    ax.set_ylabel("count")
    plt.show()

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax: plt.Axes
    ax.hist(df["compactness"], bins="sqrt")
    ax.set_xlabel("compactness")
    ax.set_ylabel("count")
    plt.show()

    fig, ax = plt.subplots(1, 1, dpi=600)
    df_with_vessel = df[df["n_central_vessel"] > 0]
    ax: plt.Axes
    ax.scatter(df_with_vessel["central_vessel_cross_section"], df_with_vessel["area"])
    ax.set_ylabel("lobule area")
    ax.set_xlabel("pericentral vessel area")
    plt.show()



if __name__ == "__main__":
    result_dir = Path(__file__).parent
    slide_stats = SlideStats.load_from_file_system(result_dir, "NOR_021")

    df = slide_stats.to_dataframe()
    visualize_slide_stats(df)
