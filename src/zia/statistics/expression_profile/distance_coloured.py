from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zia import BASE_PATH
from zia.config import read_config
from zia.statistics.utils.data_provider import SlideStatsProvider


def plot_pic(array, title: str = None, cmap: str = "seismic") -> plt.Figure:
    fig: plt.Figure
    ax: plt.Axes
    base_size = 6.4
    hw_ratio = array.shape[0] * 1.1 / array.shape[1]

    if hw_ratio > 0.5:
        h, w = base_size, base_size / hw_ratio
    else:
        h, w = base_size * hw_ratio, base_size

    pixel_width = 0.22724690376093626  # µm level 0
    p_factor = 2 ** 7 * pixel_width

    fig, ax = plt.subplots(1, 1, figsize=(w, h), dpi=300)
    fig.set_facecolor("lightgray")
    ax.set_facecolor("lightgray")
    if title is not None:
        fig.suptitle(title)
    show = ax.imshow(array, cmap=matplotlib.colormaps.get_cmap(cmap), vmin=0, vmax=1)
    cax = ax.inset_axes([0.78, 0.1, 0.2, 0.05])
    fig.colorbar(show, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks([0, 1], ["PF", "CV"])
    ax.axis("off")

    ax.plot([0.02 * array.shape[1], 0.02 * array.shape[1] + (1000 / p_factor)], [1.1 * array.shape[0], 1.1 * array.shape[0]], color="black",
            marker="none", linewidth=2)
    ax.text(x=0.02 * array.shape[1] + (1000 / p_factor) / 2,
            y=1.1 * array.shape[0],
            s="1 mm",
            fontsize=10,
            ha="center",
            va="bottom",
            )
    # fig.tight_layout()
    return fig


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "distance_coloured"
    report_path.mkdir(exist_ok=True, parents=True)
    protein_order = ["CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4", "GS", "HE"]
    species_order = SlideStatsProvider.species_order
    colors = SlideStatsProvider.colors
    df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)
    species_gb = df.groupby("species")

    for (subject, roi, protein), subject_df in df.groupby(["subject", "roi", "protein"]):
        max_h, max_w = np.max(subject_df["height"]), np.max(subject_df["width"])

        min_h, min_w = np.min(subject_df["height"]), np.min(subject_df["width"])

        template = np.zeros(shape=(max_h + 1, max_w + 1))
        template[template == 0] = np.nan

        template[subject_df["height"].values, subject_df["width"].values] = subject_df["pv_dist"]

        template = template[min_h:, min_w:]
        fig = plot_pic(template)

        fig.savefig(report_path / f"distance_{subject}_{roi}_{protein}.png")
        fig: plt.Figure
        plt.close(fig)
