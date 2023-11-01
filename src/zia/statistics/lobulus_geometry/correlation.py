from pathlib import Path
from typing import Tuple, List, Optional, Callable

import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.lobulus_geometry.plotting.boxplots import identity
from zia.statistics.lobulus_geometry.plotting.correlation_plots import scatter_plot_correlation
from zia.statistics.utils.data_provider import SlideStatsProvider

if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("boxplots")
    df.to_csv(report_path / "slide_statistics_df.csv", index=False)

    attributes = ["perimeter", "area", "compactness"]
    labels = ["Perimeter", "Area", "Compactness"]
    logs = [True, True, False]

    # df["minimum_bounding_radius"] = df["minimum_bounding_radius"] / 1000
    # df["minimum_bounding_radius_unit"] = "mm"

    test_results_path = SlideStatsProvider.get_report_path() / "statistical_test_results"

    len_attr = len(attributes)

    idxs = [(i, k) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]
    attr_pairs = [(attributes[i], attributes[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]
    logs_pairs = [(logs[i], logs[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]
    label_pairs = [(labels[i], labels[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]

    fig, axes = plt.subplots(nrows=int((len_attr ** 2 - len_attr) / 2), ncols=len(SlideStatsProvider.species_order), dpi=300,
                             figsize=(len(SlideStatsProvider.species_order) * 2.5, int((len_attr ** 2 - len_attr) / 2) * 2.5),
                             layout="constrained")

    species_gb = df.groupby("species")
    for i, (attr_pair, log_pair, label_pair) in enumerate(zip(attr_pairs, logs_pairs, label_pairs)):
        for k, (species, color) in enumerate(zip(SlideStatsProvider.species_order, SlideStatsProvider.get_species_colors_as_rgb())):
            species_df = species_gb.get_group(species)
            scatter_plot_correlation(species_df, color=color, attr=attr_pair, log=log_pair, labels=label_pair, ax=axes[i, k], scatter=False)
    plt.show()

    fig2, axes2 = plt.subplots(nrows=2, ncols=2, figsize=(2 * 2 * 4, 2 * 2 * 4))

    subaxes = []
    n = 2
    pad = 0.05
    for idx in [(0, 0), (1, 0), (1, 1)]:
        for i in range(n):
            for k in range(n):
                ax = axes2[idx[0], idx[1]]
                subaxes.append(ax.inset_axes(((k + pad / 2) / n, (i + pad / 2) / n, (1 - pad) / n, (1 - pad) / n),
                                             transform=ax.transAxes))

    plt.show()
