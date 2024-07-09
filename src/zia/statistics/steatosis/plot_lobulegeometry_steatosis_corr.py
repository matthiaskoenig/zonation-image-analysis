from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde, spearmanr

from zia.statistics.steatosis.utils.get_data import DIET
from zia.statistics.steatosis.utils.utils import PIXEL_SIZE, SPECIES_ORDER, GROUP_ORDER, SPECIES_COLORS
from zia.statistics.utils.data_provider import capitalize


def map_to_group(species, diet) -> pd.DataFrame:
    if species in ["mouse", "rat"]:
        return f"{diet}W"
    return "Stea."


def get_from_species(sp, gr):
    if sp in ["mouse", "rat"]:
        return f"{sp} ({gr} HDF)"
    else:
        return sp


def plot_lobulegeo_steatosis_correlation(reportpath: Path,
                                         portality_droplet_df: pd.DataFrame,
                                         slide_stats_df_steatosis: pd.DataFrame,
                                         slide_stats_df_control: pd.DataFrame,
                                         attributes: List[str],
                                         logs: List[bool],
                                         labels: List[str],
                                         units: List[str]):
    marker_dict = {
        "Control": "o",
        "2W": "D",
        "4W": "s",
        "Stea.": "v"
    }

    slide_stats_df_steatosis["diet"] = slide_stats_df_steatosis["subject"].map(DIET)
    slide_stats_df_steatosis["group"] = slide_stats_df_steatosis.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)
    slide_stats_df_control["group"] = "Control"

    lobule_stats_df = pd.concat([slide_stats_df_control, slide_stats_df_steatosis], ignore_index=True)

    portality_droplet_df["droplet_area_fraction"] = portality_droplet_df["total_droplet_area"] / ((PIXEL_SIZE * 2 ** 7) ** 2) * 100

    # colors = ["#77AADD", "#EE8866", "#DDDDDD", "#44BB99"]

    fig, axes = plt.subplots(nrows=len(attributes), ncols=len(GROUP_ORDER), dpi=600,
                             figsize=(len(GROUP_ORDER) * 2, len(attributes) * 2),
                             layout="constrained")
    fig: plt.Figure
    for i, (attribute, y_log) in enumerate(zip(attributes, logs)):

        for k, (sp, group_order) in enumerate(GROUP_ORDER.items()):
            print(sp)
            species_lobule_stats = lobule_stats_df[lobule_stats_df["species"] == sp]
            species_droplet_stats = portality_droplet_df[portality_droplet_df["species"] == sp]

            print(len(species_lobule_stats), len(species_droplet_stats))
            print(species_droplet_stats.columns)
            print(species_droplet_stats.group.unique())

            species_fat = []
            species_geo = []

            for gr in group_order:

                group_lobule_stats = species_lobule_stats[species_lobule_stats["group"] == gr]

                group_droplet_stats = species_droplet_stats[species_droplet_stats["group"] == get_from_species(sp, gr)]

                group_fat = []
                group_geo = []

                for (roi, subject), roi_subject_df in group_lobule_stats.groupby(by=["roi", "subject"]):
                    roi_subject_droplet_df = group_droplet_stats[(group_droplet_stats["roi"] == int(roi)) &
                                                                 (group_droplet_stats["subject"] == subject)]

                    for _, row in roi_subject_df.iterrows():
                        if len(roi_subject_droplet_df) == 0:
                            continue
                        else:
                            fat_area = roi_subject_droplet_df[(roi_subject_droplet_df["lobule"] == row["lobule_id"])]["total_droplet_area"].sum()

                        if fat_area > 0:
                            group_fat.append(fat_area)
                            group_geo.append(row[attribute])

                species_fat.extend(group_fat)
                species_geo.extend(group_geo)

            x = np.log10(species_fat)
            y = species_geo if not y_log else np.log10(species_geo)
            xy = np.vstack([x, y])

            gkde = gaussian_kde(xy)

            z = gkde(xy)
            axes[i, k].scatter(
                x=species_fat,
                y=species_geo,
                c=z,
                s=3
            )
            spearman_corr = spearmanr(species_fat, species_geo).statistic

            # spearman_corr_log = spearmanr(np.log(x) if x_log else x, np.log(y) if y_log else y)
            # pearson_corr = pearsonr(x, y)
            # pearson_corr_log = pearsonr(np.log(x) if x_log else x, np.log(y) if y_log else y)

            if spearman_corr < 0:
                x = 0.02
                ha = "left"
            else:
                x = 0.98
                ha = "right"

            axes[i, k].text(x=x, y=0.02, s=f"r={spearman_corr:.2f}", fontsize=12, ha=ha, transform=axes[i, k].transAxes)

    for ax in axes.flatten():
        ax: plt.Axes
        ax.set_xscale("log")

    for ax_row, log in zip(axes, logs):
        if log:
            for ax in ax_row:
                ax.set_yscale("log")

    for ax in axes[:-1, :].flatten():
        ax.set_xticklabels([])

    for ax in axes[:, 1:].flatten():
        ax.set_yticklabels([])

    for ax_row in axes:

        mins = []
        maxs = []

        for ax in ax_row:
            mins.append(ax.get_ylim()[0])
            maxs.append(ax.get_ylim()[1])

        for ax in ax_row:
            ax.set_ylim(bottom=min(mins), top=max(maxs))

    for i in range(axes.shape[1]):

        ax_col = axes[:, i]

        mins = []
        maxs = []

        for ax in ax_col:
            mins.append(ax.get_xlim()[0])
            maxs.append(ax.get_xlim()[1])

        for ax in ax_col:
            ax.set_xlim(left=min(mins), right=max(maxs))

    for ax, label, unit in zip(axes[:, 0], labels, units):
        ax.set_ylabel(f"{capitalize(label)} ({unit})", fontsize=10)

    for ax, species in zip(axes[0, :], SPECIES_ORDER):
        ax.set_title(capitalize(species), fontsize=11, fontweight='bold')

    for ax in axes[1, :]:
        ax.axline((10, 10), (100, 100), color='grey', linestyle='--')

    fig.supxlabel("Total MS droplet area (µm$^2$)", fontsize=10)

    fig.savefig(reportpath / f"corr_steatosis_lobule_geometry.png", dpi=600)
    fig.savefig(reportpath / f"corr_steatosis_lobule_geometry.svg", dpi=600)

    plt.show()
