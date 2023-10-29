from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zia import BASE_PATH
from zia.config import read_config
from zia.processing.lobulus_statistics import SlideStats
from zia.statistics.utils.data_provider import SlideStatsProvider


def plot_pic(array, slide_stats: SlideStats, min_h, min_w, title: str = None, cmap: str = "magma", ) -> plt.Figure:
    fig: plt.Figure
    ax: plt.Axes
    base_size = 6.4
    array = array[min_h:, min_w:]
    hw_ratio = array.shape[0] * 1.1 / array.shape[1]

    if hw_ratio > 0.5:
        h, w = base_size, base_size / hw_ratio
    else:
        h, w = base_size * hw_ratio, base_size

    pixel_width = 0.22724690376093626  # µm level 0
    p_factor = 2 ** 7 * pixel_width

    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(w, h), dpi=300, sharex=True, height_ratios=[0.95, 0.05])

    if title is not None:
        ax.set_title(title)
    show = ax.imshow(array, cmap=matplotlib.colormaps.get_cmap(cmap))
    cax = ax1.inset_axes([0.78, 0, 0.2, 1])
    fig.colorbar(show, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks([np.nanmax(array), np.nanmin(array)], ["PP", "PV"])
    ax.axis("off")
    ax1.axis("off")

    ax1.plot([0.02 * array.shape[1], 0.02 * array.shape[1] + (1000 / p_factor)], [0, 0], color="black",
             marker="none", linewidth=2)

    slide_stats.plot_on_axis(ax,
                             lobulus_ec="white",
                             cvessel_ec="none",
                             cvessel_fc="none",
                             pvessel_fc="none",
                             pvessel_ec="none",
                             offset=(min_h, min_w))

    ax1.text(x=0.02 * array.shape[1] + (1000 / p_factor) / 2,
             y=0,
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

    slide_stats_dict = SlideStatsProvider.get_slide_stats()
    for (subject, roi), subject_df in df.groupby(["subject", "roi"]):

        arrays = []
        mins = []

        slide_stats = slide_stats_dict[str(subject)][str(roi)]

        for protein, protein_df in subject_df.groupby("protein"):
            max_h, max_w = np.max(subject_df["height"]), np.max(subject_df["width"])

            min_h, min_w = np.min(subject_df["height"]), np.min(subject_df["width"])

            template = np.zeros(shape=(max_h + 1, max_w + 1))
            template[template == 0] = np.nan

            template[subject_df["height"].values, subject_df["width"].values] = subject_df["pv_dist"]

            arrays.append(template)
            mins.append((min_h, min_w))

        idx = np.argmax([arr[arr != np.nan].size for arr in arrays])

        template = arrays[idx]
        min_h, min_w = mins[idx]

        fig = plot_pic(template, slide_stats, min_h, min_w, title=f"Subject: {subject}, ROI: {roi}")
        fig.savefig(report_path / f"distance_{subject}_{roi}.png", bbox_inches='tight', pad_inches=0)
        fig: plt.Figure
        plt.close(fig)
