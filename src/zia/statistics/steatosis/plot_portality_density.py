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
    y_labels = ["MS droplet density", "MS area fraction", "Mean droplet size", "Droplet area skew"]
    units = ["mm$^{-2}$", "%", "µm$^{2}$", "-"]

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

        for subject, subject_df in group_df.groupby("subject"):
            print(subject)
            print("subject", len(subject_df))
            x = []
            droplet_area = []
            droplet_area_perc = []
            droplet_area_std = []
            droplet_area_fraction = []
            droplet_area_fraction_perc = []
            droplet_area_fraction_std = []
            droplet_density = []
            droplet_density_perc = []
            droplet_density_std = []
            droplet_area_skew = []

            for i in range(len(bins) - 1):
                df_bin = subject_df[(subject_df["pv_dist"] > bins[i]) & (subject_df["pv_dist"] <= bins[i + 1])]
                x.append((bins[i] + bins[i + 1]) / 2)

                st_bin = df_bin[df_bin["droplet_count"] > 0]

                if len(st_bin) == 0:
                    droplet_area.append(np.nan)  #
                    droplet_area_perc.append((np.nan, np.nan))
                    droplet_area_std.append(np.nan)
                    droplet_area_fraction.append(np.nan)
                    droplet_area_fraction_perc.append((np.nan, np.nan))
                    droplet_area_fraction_std.append(np.nan)
                    droplet_density.append(np.nan)
                    droplet_density_perc.append((np.nan, np.nan))
                    droplet_density_std.append(np.nan)
                    droplet_area_skew.append(np.nan)

                else:
                    droplet_area.append((st_bin["total_droplet_area"] / st_bin["droplet_count"]).mean())
                    droplet_area_perc.append(np.percentile(st_bin["total_droplet_area"] / st_bin["droplet_count"], [25, 75]))
                    droplet_area_std.append((st_bin["total_droplet_area"] / st_bin["droplet_count"]).std())
                    droplet_area_skew.append((st_bin["total_droplet_area"] / st_bin["droplet_count"]).skew())
                    droplet_area_fraction.append(
                        df_bin["total_droplet_area"].mean() / ((PIXEL_SIZE * 2 ** level) ** 2) * 100
                    )

                    droplet_area_fraction_perc.append(
                        np.percentile(df_bin["total_droplet_area"] / ((PIXEL_SIZE * 2 ** level) ** 2) * 100, [25, 75])
                    )

                    droplet_area_fraction_std.append(df_bin["total_droplet_area"].std() / ((PIXEL_SIZE * 2 ** level) ** 2) * 100)

                    droplet_density.append(
                        df_bin["droplet_count"].mean() / ((PIXEL_SIZE * 2 ** level) ** 2 / 1e6)  # µm² -> mm²
                    )
                    droplet_density_perc.append(np.percentile(df_bin["droplet_count"] / ((PIXEL_SIZE * 2 ** level) ** 2 / 1e6), [25, 75]))

                    droplet_density_std.append(df_bin["droplet_count"].std() / ((PIXEL_SIZE * 2 ** level) ** 2 / 1e6))

            ax: plt.Axes = axes[0, col]

            print(droplet_density_perc)

            min_y = [x[0] for x in droplet_density_perc]
            max_y = [x[1] for x in droplet_density_perc]

            print(min_y, max_y)

            ax.plot(x, droplet_density,
                    marker="o",
                    markerfacecolor=colors[col],
                    markeredgecolor="black",
                    linewidth=1,
                    markersize=4,
                    zorder=10,
                    color="black")

            ax.fill_between(x,np.array(droplet_density) - np.array(droplet_density_std),
                            np.array(droplet_density) + np.array(droplet_density_std),
                            alpha=0.2, color=colors[col])

            # plot
            ax: plt.Axes = axes[1, col]

            min_y = [x[0] for x in droplet_area_fraction_perc]
            max_y = [x[1] for x in droplet_area_fraction_perc]

            ax.plot(x, droplet_area_fraction,
                    marker="o",
                    markerfacecolor=colors[col],
                    markeredgecolor="black",
                    linewidth=1,
                    markersize=4,
                    zorder=10,
                    color="black")
            ax.fill_between(x, np.array(droplet_area_fraction) - np.array(droplet_area_fraction_std),
                            np.array(droplet_area_fraction) + np.array(droplet_area_fraction_std),
                            alpha=0.2, color=colors[col])

            ax: plt.Axes = axes[2, col]

            min_y = [x[0] for x in droplet_area_perc]
            max_y = [x[1] for x in droplet_area_perc]

            ax.plot(x, droplet_area,
                    marker="o",
                    markerfacecolor=colors[col],
                    markeredgecolor="black",
                    linewidth=1,
                    markersize=4,
                    zorder=10,
                    color="black")
            ax.fill_between(x, np.array(droplet_area) - np.array(droplet_area_std),
                            np.array(droplet_area) + np.array(droplet_area_std), alpha=0.2, color=colors[col])

            ax: plt.Axes = axes[3, col]


            ax.plot(x, droplet_area_skew,
                    marker="o",
                    markerfacecolor=colors[col],
                    markeredgecolor="black",
                    linewidth=1,
                    markersize=4,
                    zorder=10,
                    color="black")
            #ax.fill_between(x, min_y, max_y, alpha=0.2, color=colors[col])

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
