from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.steatosis.utils import map_to_group, PIXEL_SIZE
from zia.statistics.utils.data_provider import capitalize


def plot_droplet_portality(report_path: Path,
                           distance_df: pd.DataFrame,
                           group_order: List[str],
                           colors: List[str],
                           level: int = 7):
    plt.style.use("tableau-colorblind10")

    y_labels = ["MS mean droplet area", "MS droplet density", "MS area fraction"]
    units = ["µm$^2$", "mm$^{-2}$", "%"]

    distance_df["group"] = distance_df.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)

    # colors = ["#77AADD", "#EE8866", "#DDDDDD", "#44BB99"]

    group_gb = distance_df.groupby("group")

    fig, axes = plt.subplots(nrows=len(y_labels), ncols=len(group_order), dpi=300,
                             figsize=(len(group_order) * 2, len(y_labels) * 2),
                             layout="constrained")

    print(group_order)

    # density per portality bin

    for col, group in enumerate(group_order):
        group_df = group_gb.get_group(group)

        bins = np.histogram_bin_edges(group_df["pv_dist"], range=(0, 1), bins=12)
        binned, bins = pd.cut(group_df["pv_dist"], bins=bins, retbins=True)

        x = []
        droplet_area = []
        droplet_area_medians = []
        droplet_area_fraction = []
        droplet_density = []

        for i in range(len(bins) - 1):
            df_bin = group_df[(group_df["pv_dist"] > bins[i]) & (group_df["pv_dist"] <= bins[i + 1])]
            x.append((bins[i] + bins[i + 1]) / 2)
            droplet_area.append(df_bin["mean_droplet_area"])
            droplet_area_medians.append(np.median(df_bin["mean_droplet_area"]))

            droplet_area_fraction.append(
                df_bin["mean_droplet_area"].sum() / (len(df_bin) * (PIXEL_SIZE * 2 ** level) ** 2) * 100
            )

            droplet_density.append(
                df_bin["droplet_count"].sum() / (len(df_bin) * (PIXEL_SIZE * 2 ** level) ** 2 / 1e6)  # µm² -> mm²
            )

        d = (bins[1] - bins[0])

        # plot area medians
        ax: plt.Axes = axes[0, col]
        bp = ax.boxplot(x=droplet_area, positions=x, widths=d, patch_artist=True, showfliers=False,
                        whis=(5, 95))

        ax.plot(x, droplet_area_medians,
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

        lobule_count = len(group_df.groupby(["subject", "roi", "lobule"]))

        ax.text(x=0.02, y=0.98, s=f"n={lobule_count}", fontsize=10, ha="left", va="top", transform=ax.transAxes)

        ax.set_xticks([])

        # plot droplet density
        ax: plt.Axes = axes[1, col]

        ax.plot(x, droplet_density,
                marker="o",
                markerfacecolor=colors[col],
                markeredgecolor="black",
                linewidth=1,
                markersize=4,
                zorder=10,
                color="black")

        # plot
        ax: plt.Axes = axes[2, col]

        ax.plot(x, droplet_area_fraction,
                marker="o",
                markerfacecolor=colors[col],
                markeredgecolor="black",
                linewidth=1,
                markersize=4,
                zorder=10,
                color="black")

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
