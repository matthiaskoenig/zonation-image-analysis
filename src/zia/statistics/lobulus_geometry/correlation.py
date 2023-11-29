from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from zia.statistics.lobulus_geometry.plotting.correlation_plots import scatter_plot_correlation
from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize


def plot_correlation(slide_stats_df: pd.DataFrame,
                     report_path: Path,
                     attributes: List[str],
                     labels: List[str],
                     logs: List[bool],
                     binned: bool):
    species_order = SlideStatsProvider.species_order

    species_colors = SlideStatsProvider.get_species_colors_as_rgb()

    fig, axes = plt.subplots(nrows=len(attributes), ncols=len(species_order), figsize=(len(species_order) * 2.5, len(attributes) * 2.5),
                             layout="constrained")

    species_gb = slide_stats_df.groupby("species")

    attr_pairs = [(attributes[i], attributes[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]
    logs_pairs = [(logs[i], logs[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]
    label_pairs = [(labels[i], labels[k]) for i in range(len(attributes) - 1) for k in range(i + 1, len(attributes))]

    unit_pairs = []

    legend_handles = []
    for i, (attr_pair, log_pair, label_pair) in enumerate(zip(attr_pairs, logs_pairs, label_pairs)):
        limits_x = np.min(slide_stats_df[attr_pair[0]]), np.max(slide_stats_df[attr_pair[0]]),
        limits_y = np.min(slide_stats_df[attr_pair[1]]), np.max(slide_stats_df[attr_pair[1]]),

        unit_pairs.append((set(slide_stats_df[f"{attr_pair[0]}_unit"]).pop(), set(slide_stats_df[f"{attr_pair[1]}_unit"]).pop()))
        for k, (species, color) in enumerate(zip(species_order, species_colors)):
            species_df = species_gb.get_group(species)
            scatter_plot_correlation(species_df, color=color, attr=attr_pair, log=log_pair, labels=label_pair, limits_x=limits_x, limits_y=limits_y,
                                     ax=axes[i, k], scatter=not binned)
            if i == 0:
                legend_handles.append(Patch(color=color, label=capitalize(species)))

    for ax in axes[:, 1:].flatten():
        ax.set_ylabel("")

    for ax, species in zip(axes[0, :].flatten(), species_order):
        ax.set_title(capitalize(species), fontweight="bold")

    plt.savefig(report_path / f"correlation_matrix{'_binned' if binned else ''}.png", dpi=600)
    plt.savefig(report_path / f"correlation_matrix{'_binned' if binned else ''}.svg", dpi=600)

    plt.show()


if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("correlation")
    df.to_csv(report_path / "slide_statistics_df.csv", index=False)

    attributes = ["perimeter", "area", "compactness"]
    labels = ["Perimeter", "Area", "Compactness"]
    logs = [True, True, False]

    plot_correlation(df, report_path, attributes, labels, logs, binned=True)
