from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.steatosis.utils.boxplots import box_plot_species_comparison
from zia.statistics.steatosis.utils.descriptive_stats import create_stats_from_data_dict
from zia.statistics.steatosis.utils.utils import SPECIES_COLORS_RGB, create_data_dict


def map_to_group(species, diet) -> pd.DataFrame:
    if species in ["mouse", "rat"]:
        return f"{diet}W"
    return "Stea."


def calc_average_density(slide_stats_df: pd.DataFrame, area: float) -> float:
    return len(slide_stats_df) / (area / 1e6)  # mm^2


def calc_average_area_density(slide_stats_df: pd.DataFrame, area: float) -> float:
    return (slide_stats_df["area"].sum() / area) * 100


def species_comparison_droplet_density(droplet_stats_df: pd.DataFrame,
                                       wsi_df: pd.DataFrame,
                                       report_path: Path,
                                       stats_excel: Path):
    attributes = ["Average droplet density", "Surface coverage"]
    labels = ["droplet density", "surface coverage"]
    units = ["mm$^{-2}$", "%"]
    logs = [False, False]

    subject_roi_gb = droplet_stats_df.groupby(["subject", "roi"])

    wsi_df["Average droplet density"] = wsi_df.apply(
        lambda row: calc_average_density(
            subject_roi_gb.get_group((row["subject"], row["roi"])),
            row["area"]
        ),
        axis=1
    )

    wsi_df["Surface coverage"] = wsi_df.apply(
        lambda row: calc_average_area_density(
            subject_roi_gb.get_group((row["subject"], row["roi"])),
            row["area"]
        ),
        axis=1
    )

    wsi_df["group"] = wsi_df.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)

    data_dict = create_data_dict(attributes, wsi_df)

    stats = create_stats_from_data_dict(data_dict=data_dict)

    with pd.ExcelWriter(stats_excel, mode='a' if stats_excel.exists() else 'w', if_sheet_exists="replace"  if stats_excel.exists() else None) as w:
        stats.to_excel(w, sheet_name='species-comparison-droplet-density', index=False)

    plot_species_comparison_density(data_dict, attributes, logs, labels, units, report_path)


def plot_species_comparison_density(data_dict: dict,
                                    attributes: List[str],
                                    logs: List[bool],
                                    labels: List[str],
                                    units: List[str],
                                    report_path: Path):
    fig, axes = plt.subplots(1, 2, dpi=300,
                             figsize=(2 * 2.5, 2.5),
                             layout="constrained")

    for (attr, ax, log, y_label, unit) in zip(attributes, axes, logs, labels, units):
        box_plot_species_comparison(data_dict,
                                    attr,
                                    y_label=y_label,
                                    colors=SPECIES_COLORS_RGB,
                                    log=log,
                                    ax=ax,
                                    annotate_n=True,
                                    unit=unit,
                                    show_violins=False)

    plt.savefig(report_path / "species_comparison_droplet_density.png", dpi=600)
    plt.savefig(report_path / "species_comparison_droplet_density.svg", dpi=600)

    plt.show()
