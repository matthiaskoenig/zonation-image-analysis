from pathlib import Path
from typing import List, Dict

import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.steatosis.utils.descriptive_stats import create_stats_from_data_dict
from zia.statistics.steatosis.utils.boxplots import box_plot_species_comparison
from zia.statistics.steatosis.utils.utils import SPECIES_ORDER, SPECIES_COLORS_RGB, create_data_dict

def map_to_group(species, diet) -> pd.DataFrame:
    if species in ["mouse", "rat"]:
        return f"{diet}W"
    return "Stea."
def plot_species_droplet_comparison(data_dict: Dict[str, Dict[str, Dict[str, pd.Series]]],
                                    report_path: Path,
                                    attributes: List[str],
                                    labels: List[str],
                                    logs: List[bool],
                                    units: List[str]):
    fig, axes = plt.subplots(1, len(attributes), dpi=300,
                             figsize=(len(attributes) * 2.5, 2.5),
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
                                    show_violins=True)

    plt.savefig(report_path / "species_droplet_comparison.png", dpi=600)
    plt.savefig(report_path / "species_droplet_comparison.svg", dpi=600)

    plt.show()


def droplet_species_comparison(droplet_stats_df: pd.DataFrame,
                               report_path: Path,
                               stats_excel: Path,
                               attributes: List[str],
                               labels: List[str],
                               logs: List[bool],
                               units: List[str]):
    droplet_stats_df["group"] = droplet_stats_df.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)

    data_dict = create_data_dict(attributes, droplet_stats_df)

    stats = create_stats_from_data_dict(data_dict=data_dict)

    print(stats_excel.exists())
    with pd.ExcelWriter(stats_excel, mode='a' if stats_excel.exists() else 'w', if_sheet_exists="replace"  if stats_excel.exists() else None) as w:
        stats.to_excel(w, sheet_name="species-comparison-droplet-geometry", index=False)

    plot_species_droplet_comparison(data_dict,
                                    report_path,
                                    attributes,
                                    labels,
                                    logs,
                                    units)
