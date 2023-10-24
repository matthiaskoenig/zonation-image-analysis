from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from zia import BASE_PATH
from zia.config import read_config
import pandas as pd

from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize


def dist(d_p: float, d_c: float) -> float:
    return 1 - d_c / (d_p + d_c)


def get_xlimits(slide_stat_df: pd.DataFrame, q=50) -> Dict[str, float]:
    limits = {}
    for species, df in slide_stat_df.groupby("species"):
        limits[species] = np.percentile(df["minimum_bounding_radius"], q)
    return limits


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "expression_gradient"
    report_path.mkdir(exist_ok=True, parents=True)
    protein_order = ["CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4", "GS", "HE"]
    species_order = SlideStatsProvider.species_order
    colors = SlideStatsProvider.colors
    df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)
    species_gb = df.groupby("species")

    x_limits = get_xlimits(SlideStatsProvider.get_slide_stats_df(), 50)
    bin_size = 50  # µm
    fig, axes = plt.subplots(nrows=len(protein_order), ncols=len(species_order), dpi=300, figsize=(len(species_order) * 2, len(protein_order) * 1.85),
                             layout="constrained")
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

            protein_df = protein_df[protein_df["d_central"] <= x_limits[species]]

            bins = int(np.ceil(x_limits[species] / bin_size))

            binned, bins = pd.cut(protein_df["d_central"], bins=bins, retbins=True)

            x = []
            y = []
            for i in range(len(bins) - 1):
                df_bin = protein_df[(protein_df["d_central"] > bins[i]) & (protein_df["d_central"] <= bins[i + 1])]

                x.append((bins[i] + bins[i + 1]) / 2)
                y.append(df_bin["intensity"])

            d = (bins[1] - bins[0])
            bp = ax.boxplot(x=y, positions=x, widths=d, patch_artist=True, showfliers=False, medianprops=dict(color="black"),
                            whis=[5, 95])
            for box in bp["boxes"]:
                box.set(facecolor=colors[col], linewidth=0.5)
            for box in bp["medians"]:
                box.set(linewidth=0.5)
            for box in bp["caps"]:
                box.set(linewidth=0.5)
            for box in bp["whiskers"]:
                box.set(linewidth=0.5)

            ax.set_xticks([])
            # ax.fill_between(x, ler, her, alpha=0.5, color=colors[col])

            # ax.set_xlim(left=0, right=1.4)
            # ax.set_ylim(bottom=0, top=1)
            # ax.legend(frameon=False)

    fig: plt.Figure
    fig.supxlabel("Distance to lobule center (µm)", fontsize=14)
    fig.supylabel("Normalized intensity (-)", fontsize=14)

    for species, ax in zip(species_order, axes[-1, :].flatten()):
        x = np.arange(0, 1500, 200)
        this_range = x[x <= x_limits[species]]
        ax.xaxis.set_ticks(this_range, labels=this_range)

    for i, species in enumerate(species_order):
        for ax in axes[:, i].flatten():
            ax.set_xlim(left=0, right=x_limits[species])

    for ax in axes.flatten():
        ax.set_ylim(top=1.1, bottom=0)

    for ax in axes[:-1, 1:].flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    for ax in axes[:-1, 0].flatten():
        ax.set_xticklabels([])

    for ax in axes[-1, 1:].flatten():
        ax.set_yticklabels([])

    for protein, ax in zip(protein_order, axes[:, -1].flatten()):
        ax.set_ylabel(protein, fontsize=14, fontweight="bold")
        ax.yaxis.set_label_position("right")

    for species, ax in zip(species_order, axes[0, :].flatten()):
        ax.set_title(capitalize(species), fontsize=14, fontweight="bold")

    fig.savefig(report_path / f"expression_profile.png")
    plt.show()
