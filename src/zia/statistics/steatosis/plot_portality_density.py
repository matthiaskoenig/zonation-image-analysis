from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from zia.statistics.steatosis.utils.boxplots import violin_plot
from zia.statistics.steatosis.utils.get_data import get_example_image
from zia.statistics.steatosis.utils.utils import map_to_group, PIXEL_SIZE
from zia.statistics.utils.data_provider import capitalize
from scipy.interpolate import interp1d


def from_hex(color: str) -> Tuple[float, ...]:
    return (tuple(int(color.strip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)))


def plot_droplet_portality(report_path: Path,
                           distance_df: pd.DataFrame,
                           group_order: List[str],
                           colors: List[str],
                           level: int = 7):
    y_labels = ["Surface coverage", "Mean droplet size"]
    attributes = ["mean_droplet_area"]
    units = ["%", "µm$^{2}$"]

    distance_df["droplet_area_fraction"] = distance_df["total_droplet_area"] / ((PIXEL_SIZE * 2 ** level) ** 2) * 100

    # colors = ["#77AADD", "#EE8866", "#DDDDDD", "#44BB99"]

    group_gb = distance_df.groupby("group")

    fig, axes = plt.subplots(nrows=len(y_labels) + 1, ncols=len(group_order), dpi=300,
                             figsize=(len(group_order) * 2, (len(y_labels) + 1) * 2),
                             layout="constrained")
    print(group_gb.groups.keys())
    print(group_order)

    # density per portality bin

    for col, group in enumerate(group_order):

        group_df = group_gb.get_group(group)
        # group_df = group_df[group_df["droplet_count"] > 0]

        bins = np.histogram_bin_edges(group_df["pv_dist"], range=(0, 1), bins=12)
        binned, bins = pd.cut(group_df["pv_dist"], bins=bins, retbins=True)

        axes[0, col].imshow(get_example_image(group))

        d = bins[1] - bins[0]

        x = []
        y_frames_non_zero = []
        y_frames = []
        steatotis_fraction = []

        percentiles = [45, 40, 30, 20, 10]

        bin_low = []
        bin_high = []
        median = []
        means = []
        for i in range(len(bins) - 1):
            df_bin = group_df[(group_df["pv_dist"] > bins[i]) & (group_df["pv_dist"] <= bins[i + 1])]
            df_bin_non_zero = df_bin[df_bin["droplet_count"] > 0]

            x.append((bins[i] + bins[i + 1]) / 2)
            y_frames_non_zero.append(df_bin_non_zero)
            y_frames.append(df_bin)
            steatotis_fraction.append(df_bin["droplet_area_fraction"].mean())

            lows = []
            highs = []
            for p in percentiles:
                low, high = np.percentile(df_bin["droplet_area_fraction"], (50 - p, 50 + p))
                lows.append(low)
                highs.append(high)
            median.append(np.percentile(df_bin["droplet_area_fraction"], 50))
            means.append(np.mean(df_bin["droplet_area_fraction"]))

            bin_low.append(lows)
            bin_high.append(highs)

        # # get the max value of all bins
        # max_d = max([np.percentile(d, 95) for d in bin_data])
        # min_d = min([np.percentile(d, 5) for d in bin_data])
        #
        # # create a array at which we can evaluate the percentile
        #
        # d_reads = np.linspace(min_d, max_d, 100)
        #
        # #
        # p_reads = np.vstack([np.interp(d_reads, bin_d, bin_p) for bin_d, bin_p in zip(bin_data, bin_percentiles)]).T
        # cmap_colors = [from_hex(colors[col]) + (1,), from_hex(colors[col]) + (0,)]
        # cmap = LinearSegmentedColormap.from_list("whatever", cmap_colors)

        x_ip = np.linspace(0, 1, 100)

        position_low = []
        max_val_low = []
        position_high = []
        max_val_high = []

        for ip, p in enumerate(percentiles):
            low_y = [p_vec[ip] for p_vec in bin_low]
            high_y = [p_vec[ip] for p_vec in bin_high]

            pos_low = np.argmax(low_y)
            pos_high = np.argmax(high_y)

            position_low.append(x[pos_low])
            position_high.append(x[pos_high])

            max_val_low.append(low_y[pos_low])
            max_val_high.append(high_y[pos_high])

            f_low_y = interp1d(x, low_y, fill_value="extrapolate")
            f_high_y = interp1d(x, high_y, fill_value="extrapolate")

            axes[1, col].fill_between(
                x_ip,
                np.maximum(f_low_y(x_ip), 0),
                np.maximum(f_high_y(x_ip), 0),
                color=colors[col],
                alpha=1 / len(percentiles),
                edgecolor="None"
            )

        axes[1, col].plot(
            x, means,
            marker="P",
            markerfacecolor=colors[col],
            markeredgecolor="black",
            linewidth=1,
            markersize=4,
            zorder=20,
            color="black"
        )

        median_ip = interp1d(x, median, fill_value="extrapolate")
        axes[1, col].plot(
            x_ip, median_ip(x_ip),
            marker="none",
            linewidth=2,
            zorder=10,
            color=colors[col]
        )

        for pos_low, max_low, p in zip(position_low, max_val_low, percentiles):
            if max_low < 0.1:
                continue

            pos = max(0.05, pos_low)
            pos = min(0.95, pos)

            axes[1, col].text(pos, max_low, f"{50 - p}%", fontsize=5, va="center", fontweight="bold", ha="center", color="grey")

        for pos_high, max_high, p in zip(position_high, max_val_high, percentiles):
            if max_high < 0.1:
                continue

            pos = max(0.1, pos_high)
            pos = min(0.9, pos)

            axes[1, col].text(pos, max_high, f"{50 + p}%", fontsize=5, va="center", fontweight="bold", ha="center", color="gray")

        for row, attr in enumerate(attributes):
            ax = axes[row + 2, col]

            bp, vs = violin_plot(data=[y[attr] for y in y_frames_non_zero],
                                 ax=ax,
                                 positions=x,
                                 widths=d,
                                 log=False,
                                 show_violins=False,
                                 whis=(5, 95))

            ax.plot(x, [y[attr].median() for y in y_frames_non_zero],
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

    for ax in axes[0, :].flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[1:-1, :].flatten():
        ax.xaxis.set_ticks([0, 1])

    for i, group in enumerate(group_order):
        for ax in axes[1:, i].flatten():
            ax.set_xlim(left=0, right=1)

    for ax in axes[:-1, 1:].flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for ax in axes[:-1, 0].flatten():
        ax.set_xticklabels([])

    for ax in axes[-1, 1:].flatten():
        ax.set_yticklabels([])

    for ylabel, unit, ax in zip(y_labels, units, axes[1:, 0].flatten()):
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

    fig.savefig(report_path / f"droplet-portality.png", dpi=600)
    fig.savefig(report_path / f"droplet-portality.svg", dpi=600)

    plt.show()
