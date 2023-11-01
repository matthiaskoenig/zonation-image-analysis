from pathlib import Path
from typing import Tuple, List, Optional, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from zia.statistics.lobulus_geometry.plotting.boxplots import identity
from zia.statistics.lobulus_geometry.plotting.correlation_plots import scatter_plot_correlation
from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize

if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("correlation")
    df.to_csv(report_path / "slide_statistics_df.csv", index=False)

    attributes = ["perimeter", "area", "compactness"]
    labels = ["Perimeter", "Area", "Compactness"]
    logs = [True, True, False]

    # df["minimum_bounding_radius"] = df["minimum_bounding_radius"] / 1000
    # df["minimum_bounding_radius_unit"] = "mm"

    test_results_path = SlideStatsProvider.get_report_path() / "statistical_test_results"

    len_attr = len(attributes)

    idxs = [(k - 1, i) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]
    attr_pairs = [(attributes[i], attributes[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]
    logs_pairs = [(logs[i], logs[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]
    label_pairs = [(labels[i], labels[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(2 * 4, 2 * 4), layout="constrained")

    fig_subaxes = []
    n = 2
    pad = 0.1
    for idx in idxs:
        subaxes = []
        for i in range(n):
            for k in range(n):
                ax = axes[idx[0], idx[1]]
                subaxes.append(ax.inset_axes(((k + pad / 2) / n, (i + pad / 2) / n, (1 - pad) / n, (1 - pad) / n),
                                             transform=ax.transAxes))
        fig_subaxes.append(subaxes)

    species_gb = df.groupby("species")

    unit_pairs = []

    legend_handles = []
    for i, (attr_pair, log_pair, label_pair, subaxes) in enumerate(zip(attr_pairs, logs_pairs, label_pairs, fig_subaxes)):
        limits_x = np.min(df[attr_pair[0]]), np.max(df[attr_pair[0]]),
        limits_y = np.min(df[attr_pair[1]]), np.max(df[attr_pair[1]]),

        unit_pairs.append((set(df[f"{attr_pair[0]}_unit"]).pop(), set(df[f"{attr_pair[1]}_unit"]).pop()))
        for k, (species, color, ax) in enumerate(zip(SlideStatsProvider.species_order, SlideStatsProvider.get_species_colors_as_rgb(), subaxes)):
            species_df = species_gb.get_group(species)
            scatter_plot_correlation(species_df, color=color, attr=attr_pair, log=log_pair, labels=label_pair, limits_x=limits_x, limits_y=limits_y,
                                     ax=ax, scatter=False)
            if i == 0:
                legend_handles.append(Patch(color=color, label=capitalize(species)))

    for subaxes in fig_subaxes:
        for ax in subaxes:
            ax.set_ylabel("")
            ax.set_xlabel("")

    axes[0, 1].axis("off")

    for idx, attr_pair in zip(idxs, attr_pairs):
        axes[idx[0], idx[1]].spines[["bottom", "top", "left", "right"]].set_visible(False)
        axes[idx[0], idx[1]].set_xticks([])
        axes[idx[0], idx[1]].set_yticks([])

    for idx, attr_pair, unit_pair, subaxes in zip(idxs, attr_pairs, unit_pairs, fig_subaxes):
        if idx[1] == 0:
            axes[idx[0], idx[1]].set_ylabel(f"{capitalize(attr_pair[1])} ({unit_pair[1]})", fontsize=12, fontweight="bold", labelpad=20)

        if idx[0] == 1:
            axes[idx[0], idx[1]].set_xlabel(f"{capitalize(attr_pair[0])} ({unit_pair[0]})", fontsize=12, fontweight="bold", labelpad=20)

        subax_arr = np.array(subaxes, dtype=object).reshape(2, 2)
        if idx[0] != 1:
            for ax in subax_arr.flatten():
                ax.set_xticklabels([])

        if idx[0] == 1:
            for ax in subax_arr[-1, :].flatten():
                ax.set_xticklabels([])

        if idx[1] != 0:
            for ax in subax_arr.flatten():
                ax.set_yticklabels([])

        if idx[1] == 0:
            for ax in subax_arr[:, 1].flatten():
                ax.set_yticklabels([])

    axes[0, 1].legend(frameon=False, handles=legend_handles, loc="upper left", prop=dict(size=12))
    plt.savefig(report_path / "correlation_matrix.png")
    plt.show()
