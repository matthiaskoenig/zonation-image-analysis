from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.steatosis.utils import map_to_group
from zia.statistics.utils.data_provider import capitalize


def plot_droplet_portality(report_path: Path,
                  distance_df: pd.DataFrame,
                  group_order: List[str],
                  colors: List[str],
                  attributes: List[str],
                  y_labels: List[str],
                  units: List[str]):
    plt.style.use("tableau-colorblind10")

    distance_df["group"] = distance_df.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)

    #colors = ["#77AADD", "#EE8866", "#DDDDDD", "#44BB99"]

    group_gb = distance_df.groupby("group")


    fig, axes = plt.subplots(nrows=len(attributes), ncols=len(group_order), dpi=300,
                             figsize=(len(group_order) * 2, len(attributes) * 1.85),
                             layout="constrained")

    medians_array = np.empty(shape=(len(attributes), len(group_order)), dtype=object)
    for col, group in enumerate(group_order):
        group_df = group_gb.get_group(group)

        for row, attribute in enumerate(attributes):
            ax: plt.Axes = axes[row, col]

            bins = np.histogram_bin_edges(group_df["pv_dist"], range=(0, 1), bins=12)
            binned, bins = pd.cut(group_df["pv_dist"], bins=bins, retbins=True)

            x = []
            y = []
            medians = []

            for i in range(len(bins) - 1):
                df_bin = group_df[(group_df["pv_dist"] > bins[i]) & (group_df["pv_dist"] <= bins[i + 1])]
                x.append((bins[i] + bins[i + 1]) / 2)
                y.append(df_bin[attribute])
                medians.append(np.median(df_bin[attribute]))

            medians_array[row, col] = (x, medians)
            d = (bins[1] - bins[0])

            bp = ax.boxplot(x=y, positions=x, widths=d, patch_artist=True, showfliers=False,
                            whis=(5, 95))

            ax.plot(x, medians,
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

    """# plot all species per protein
    for i in range(len(protein_order)):
        for (x, medians), c in zip(medians_array[i, :], colors):
            axes[i, -1].plot(x, medians, marker="o", color=c, markeredgecolor="black",
                             markersize=4)

    # plot all proteins per species
    for i in range(len(group_order)):
        for k, (x, medians) in enumerate(medians_array[:, i]):
            axes[-1, i].plot(x, medians, marker="o", color=f"C{k}", markeredgecolor="black",
                             markersize=4)"""

    fig: plt.Figure
    fig.supxlabel("Portality (-)", fontsize=14)
    #fig.supylabel("Normalized intensity (-)", fontsize=14)

    for ax in axes[-1, :].flatten():
        ax.xaxis.set_ticks([0, 1], labels=["PP", "PV"])

    axes[-1, -1].set_axis_off()

    for i, group in enumerate(group_order):
        for ax in axes[:, i].flatten():
            ax.set_xlim(left=0, right=1)

    #for ax in axes.flatten():
        #ax.set_ylim(top=1.2, bottom=-0.05)

    for ax in axes[:-1, 1:].flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for ax in axes[:-1, 0].flatten():
        ax.set_xticklabels([])

    for ax in axes[-1, 1:].flatten():
        ax.set_yticklabels([])

    for ylabel, unit, ax in zip(y_labels, units, axes[:, 0].flatten()):
        ax.set_ylabel(f"{ylabel}_({unit})", fontsize=12, fontweight="bold")
        #ax.yaxis.set_label_position("right")

    for group, ax in zip(group_order, axes[0, :].flatten()):
        ax.set_title(group.title(), fontsize=12)

    axes[-2, -1].xaxis.set_ticks([0, 1], labels=["PP", "PV"])

    #handles = []

    """ for i, protein in enumerate(protein_order):
        handles.append(Line2D([], [], linestyle="-", color=f"C{i}", marker="o", markeredgecolor="black", label=protein))

    axes[-1, -1].legend(handles=handles, frameon=False, ncols=1, prop=dict(size=10))"""

    fig.savefig(report_path / f"gradient_portality.png", dpi=600)
    fig.savefig(report_path / f"gradient_portality.svg", dpi=600)

    plt.show()
