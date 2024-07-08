from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.steatosis.utils.get_data import DIET
from zia.statistics.steatosis.utils.utils import PIXEL_SIZE, SPECIES_ORDER, GROUP_ORDER, SPECIES_COLORS


def map_to_group(species, diet) -> pd.DataFrame:
    if species in ["mouse", "rat"]:
        return f"{diet}W"
    return "Stea."


def get_from_species(sp, gr):
    if sp in ["mouse", "rat"]:
        return f"{sp} ({gr} HDF)"
    else:
        return sp


def plot_lobulegeo_steatosis_correlation(report_path_stats_steatosis_test: Path,
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
                             figsize=(len(GROUP_ORDER) * 3, len(attributes) * 2),
                             layout="constrained")

    for i, attribute in enumerate(attributes):
        for k, (sp, group_order) in enumerate(GROUP_ORDER.items()):
            print(sp)
            species_lobule_stats = lobule_stats_df[lobule_stats_df["species"] == sp]
            species_droplet_stats = portality_droplet_df[portality_droplet_df["species"] == sp]

            print(len(species_lobule_stats), len(species_droplet_stats))
            print(species_droplet_stats.columns)
            print(species_droplet_stats.group.unique())
            for gr in group_order:

                print(gr)
                group_lobule_stats = species_lobule_stats[species_lobule_stats["group"] == gr]

                group_droplet_stats = species_droplet_stats[species_droplet_stats["group"] == get_from_species(sp, gr)]

                print(len(group_lobule_stats), len(group_droplet_stats))

                group_fat = []
                group_geo = []

                for (roi, subject), roi_subject_df in group_lobule_stats.groupby(by=["roi", "subject"]):
                    print(roi, subject)
                    print(group_droplet_stats.roi.unique(), group_droplet_stats.subject.unique())
                    roi_subject_droplet_df = group_droplet_stats[(group_droplet_stats["roi"] == int(roi)) &
                                                                 (group_droplet_stats["subject"] == subject)]

                    print(len(roi_subject_df), len(roi_subject_droplet_df))
                    for _, row in roi_subject_df.iterrows():
                        if len(roi_subject_droplet_df) == 0:
                            fat_area = 0
                        else:
                            fat_area = roi_subject_droplet_df[(roi_subject_droplet_df["lobule"] == row["lobule_id"])]["total_droplet_area"].sum()

                        group_fat.append(fat_area)
                        group_geo.append(row[attribute])

                axes[i, k].scatter(x=group_fat,
                                   y=group_geo,
                                   c=SPECIES_COLORS[sp],
                                   alpha=0.3,
                                   marker=marker_dict[gr]
                                   )

    plt.show()
