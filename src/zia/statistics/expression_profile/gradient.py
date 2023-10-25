from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from zia import BASE_PATH
from zia.config import read_config
import pandas as pd

from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize

if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "expression_gradient"
    report_path.mkdir(exist_ok=True, parents=True)
    protein_order = ["CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4", "GS", "HE"]
    species_order = SlideStatsProvider.species_order
    colors = SlideStatsProvider.colors
    protein_markers = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']

    df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)
    species_gb = df.groupby("species")

    fig, axes = plt.subplots(nrows=len(protein_order) + 1, ncols=len(species_order) + 1, dpi=300,
                             figsize=(len(species_order) * 2, len(protein_order) * 1.85),
                             layout="constrained")

    medians_array = np.empty(shape=(len(protein_order), len(species_order)), dtype=object)
    for col, species in enumerate(species_order):
        species_df = species_gb.get_group(species)
        protein_gb = species_df.groupby("protein")

        for row, protein in enumerate(protein_order):
            ax: plt.Axes = axes[row, col]
            protein_df = protein_gb.get_group(protein)

            # normalization based on max intensity values of one slide
            norm_dfs = []
            for (subject, roi), subject_df in protein_df.groupby(["subject", "roi"]):
                max_intensity = np.percentile(subject_df["intensity"], 99)

                subject_df["intensity"] = subject_df["intensity"] / max_intensity
                norm_dfs.append(subject_df)

            protein_df = pd.concat(norm_dfs)

            # protein_df = protein_df[protein_df["d_central"] <= x_limits[species]]

            # bins = int(np.ceil(x_limits[species] / bin_size))
            bins = np.histogram_bin_edges(protein_df["pv_dist"], range=(0, 1), bins=12)
            binned, bins = pd.cut(protein_df["pv_dist"], bins=bins, retbins=True)

            x = []
            y = []
            medians = []

            for i in range(len(bins) - 1):
                df_bin = protein_df[(protein_df["pv_dist"] > bins[i]) & (protein_df["pv_dist"] <= bins[i + 1])]
                x.append((bins[i] + bins[i + 1]) / 2)
                y.append(df_bin["intensity"])
                medians.append(np.median(df_bin["intensity"]))

            medians_array[row, col] = (x, medians)
            d = (bins[1] - bins[0])

            bp = ax.boxplot(x=y, positions=x, widths=d, patch_artist=True, showfliers=False,
                            whis=[5, 95])

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

            ax.set_xticks([])

    # plot all species per protein
    for i in range(len(protein_order)):
        for (x, medians), c in zip(medians_array[i, :], colors):
            axes[i, -1].plot(x, medians, marker="o", color=c, markeredgecolor="black",
                             markersize=4)

    # plot all proteins per species
    for i in range(len(species_order)):
        for (x, medians), m in zip(medians_array[:, i], protein_markers):
            axes[-1, i].plot(x, medians, marker="o", color=m, markerfacecolor=m, markeredgecolor="black",
                             markersize=4)

    fig: plt.Figure
    fig.supxlabel("Portality (-)", fontsize=14)
    fig.supylabel("Normalized intensity (-)", fontsize=14)

    for ax in axes[-1, :].flatten():
        ax.xaxis.set_ticks([0, 1], labels=["PF", "CV"])

    axes[-1, -1].set_axis_off()

    for i, species in enumerate(species_order):
        for ax in axes[:, i].flatten():
            ax.set_xlim(left=0, right=1)

    for ax in axes.flatten():
        ax.set_ylim(top=1.2, bottom=0)

    for ax in axes[:-1, 1:].flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for ax in axes[:-1, 0].flatten():
        ax.set_xticklabels([])

    for ax in axes[-1, 1:].flatten():
        ax.set_yticklabels([])

    for protein, ax in zip(protein_order, axes[:-1, -1].flatten()):
        ax.set_ylabel(protein, fontsize=14, fontweight="bold")
        ax.yaxis.set_label_position("right")

    for species, ax in zip(species_order, axes[0, :].flatten()):
        ax.set_title(capitalize(species), fontsize=14, fontweight="bold")

    handles = []

    for protein, c in zip(protein_order, protein_markers):
        handles.append(Line2D([], [], linestyle="-", color=c, marker="o", markeredgecolor="black", label=protein))

    """for species, c in zip(species_order, colors):
        handles.append(Patch(color=c, label=species))"""

    """for i in range(2):
        handles.append(Patch(color="none", label=""))"""

    axes[-1, -1].legend(handles=handles, frameon=False, ncols=1, prop=dict(size=10))

    fig.savefig(report_path / f"gradient.png")
    plt.show()
