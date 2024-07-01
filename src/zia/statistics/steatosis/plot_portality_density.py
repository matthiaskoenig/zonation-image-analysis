from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.steatosis.utils.boxplots import violin_plot
from zia.statistics.steatosis.utils.utils import map_to_group, PIXEL_SIZE
from zia.statistics.utils.data_provider import capitalize


def plot_droplet_portality(report_path: Path,
                           distance_df: pd.DataFrame,
                           group_order: List[str],
                           colors: List[str],
                           level: int = 7):
    y_labels = ["MS surface coverage", "MS droplet coverage", "Mean MS droplet area"]
    attributes = ["droplet_area_fraction", "mean_droplet_area"]
    units = ["%", "%", "µm$^{2}$"]

    distance_df["droplet_area_fraction"] = distance_df["total_droplet_area"] / ((PIXEL_SIZE * 2 ** level) ** 2)

    # colors = ["#77AADD", "#EE8866", "#DDDDDD", "#44BB99"]

    group_gb = distance_df.groupby("group")

    fig, axes = plt.subplots(nrows=len(y_labels), ncols=len(group_order), dpi=300,
                             figsize=(len(group_order) * 2, len(y_labels) * 2),
                             layout="constrained")
    print(group_gb.groups.keys())
    print(group_order)

    # density per portality bin

    for col, group in enumerate(group_order):

        group_df = group_gb.get_group(group)
        # group_df = group_df[group_df["droplet_count"] > 0]

        bins = np.histogram_bin_edges(group_df["pv_dist"], range=(0, 1), bins=12)
        binned, bins = pd.cut(group_df["pv_dist"], bins=bins, retbins=True)

        d = bins[1] - bins[0]

        x = []
        y_frames = []
        steatotis_fraction = []
        for i in range(len(bins) - 1):
            df_bin = group_df[(group_df["pv_dist"] > bins[i]) & (group_df["pv_dist"] <= bins[i + 1])]
            df_bin_non_zero = df_bin[df_bin["droplet_count"] > 0]
            x.append((bins[i] + bins[i + 1]) / 2)
            y_frames.append(df_bin_non_zero)
            steatotis_fraction.append(len(df_bin_non_zero) / len(df_bin))

        axes[0, col].plot(
            x, steatotis_fraction,
            marker="o",
            markerfacecolor=colors[col],
            markeredgecolor="black",
            linewidth=1,
            markersize=4,
            zorder=10,
            color="black"
        )

        for row, attr in enumerate(attributes):
            ax = axes[row + 1, col]

            bp, vs = violin_plot(data=[y[attr] for y in y_frames],
                                 ax=ax,
                                 positions=x,
                                 widths=d,
                                 log=False,
                                 show_violins=False,
                                 whis=(5, 95))

            ax.plot(x, [y[attr].median() for y in y_frames],
                    marker="o",
                    markerfacecolor=colors[col],
                    markeredgecolor="black",
                    linewidth=1,
                    markersize=4,
                    zorder=10,
                    color="black")

            for box in bp["boxes"]:
                box.set(facecolor=colors[col], linewidth=0.5)
            for box in bp["medians"]:
                box.set(color="None", linewidth=0)
            for box in bp["caps"]:
                box.set(linewidth=0.5)
            for box in bp["whiskers"]:
                box.set(linewidth=0.5)

    fig: plt.Figure
    fig.supxlabel("Portality (-)", fontsize=14)

    for ax in axes[-1, :].flatten():
        ax.xaxis.set_ticks([0, 1], labels=["PP", "PV"])

    for i, group in enumerate(group_order):
        for ax in axes[:, i].flatten():
            ax.set_xlim(left=0, right=1)

    for ax in axes[:-1, 1:].flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for ax in axes[:-1, 0].flatten():
        ax.set_xticklabels([])

    for ax in axes[-1, 1:].flatten():
        ax.set_yticklabels([])

    for ylabel, unit, ax in zip(y_labels, units, axes[:, 0].flatten()):
        ax.set_ylabel(f"{ylabel} ({unit})")

    for group, ax in zip(group_order, axes[0, :].flatten()):
        ax.set_title(capitalize(group), fontsize=12)

    for row_ax in axes:
        mins, maxs = [], []

        for ax in row_ax:
            mins.append(ax.get_ylim()[0])
            maxs.append(ax.get_ylim()[1])

        for ax in row_ax:
            ax.set_ylim(bottom=min(mins), top=max(maxs))

    fig.savefig(report_path / f"gradient_portality.png", dpi=600)
    fig.savefig(report_path / f"gradient_portality.svg", dpi=600)

    plt.show()
