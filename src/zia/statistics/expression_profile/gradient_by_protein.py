from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

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
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#f0f0f0', '#636363']

    df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)
    species_gb = df.groupby("species")

    fig, axes = plt.subplots(nrows=1, ncols=4, dpi=300, figsize=(4 * 2.5, 1.2 * 2.5),
                             layout="constrained")
    for col, species in enumerate(species_order):
        species_df = species_gb.get_group(species)
        protein_gb = species_df.groupby("protein")

        for row, protein in enumerate(protein_order):
            ax: plt.Axes = axes.flatten()[col]
            protein_df = protein_gb.get_group(protein)

            # normalization based on max intensity values of one slide
            norm_dfs = []
            for (subject, roi), subject_df in protein_df.groupby(["subject", "roi"]):
                max_intensity = np.percentile(subject_df["intensity"], 99)

                subject_df["intensity"] = subject_df["intensity"] / max_intensity
                norm_dfs.append(subject_df)

            protein_df = pd.concat(norm_dfs)

            bins = 20

            binned, bins = pd.cut(protein_df["pv_dist"], bins=bins, retbins=True)

            x = []
            y = []
            ler = []
            her = []
            for i in range(len(bins) - 1):
                df_bin = protein_df[(protein_df["pv_dist"] > bins[i]) & (protein_df["pv_dist"] <= bins[i + 1])]
                norm_intensity = df_bin["intensity"]
                x.append((bins[i] + bins[i + 1]) / 2)
                y.append(np.median(norm_intensity))
                ler.append(np.percentile(norm_intensity, 25))
                her.append(np.percentile(norm_intensity, 75))

            ax.plot(x, y, color=colors[row], marker="o", markeredgecolor="black", label=protein)
            ax.fill_between(x, ler, her, alpha=0.2, color=colors[row], edgecolor="none")

    fig: plt.Figure
    fig.supxlabel("Portality (-)", fontsize=12)
    fig.supylabel("Normalized intensity (-)", fontsize=12)

    for species, ax in zip(species_order, axes.flatten()):
        x = np.arange(0, 1500, 100)

        ax.set_xlim(left=0, right=1)
        ax.xaxis.set_ticks([0, 1], ["PF", "CV"])
        ax.set_title(capitalize(species), fontsize=10, fontweight="bold", y=0.85)

    for ax in axes.flatten():
        ax.set_ylim(top=1.0, bottom=0.2)

    for ax in axes[1:].flatten():
        ax.set_yticklabels([])

    h, l = axes[0].get_legend_handles_labels()
    lgd = fig.legend(
        h, l,
        frameon=False,
        loc="outside upper center",
        ncol=6
    )

    # fig.tight_layout()
    fig.savefig(report_path / f"gradient_by_protein.png", bbox_extra_artists=(lgd,))
    plt.show()
