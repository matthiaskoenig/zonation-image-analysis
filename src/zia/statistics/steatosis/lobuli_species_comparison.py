from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.steatosis.utils.get_data import DIET
from zia.statistics.steatosis.utils.boxplots import box_plot_species_comparison
from zia.statistics.steatosis.utils.descriptive_stats import create_stats_from_data_dict
from zia.statistics.steatosis.utils.utils import SPECIES_COLORS_RGB, create_data_dict
from zia.statistics.utils.data_provider import capitalize


def map_to_group(species, diet) -> pd.DataFrame:
    if species in ["mouse", "rat"]:
        return f"{diet}W"
    return "Stea."


def plot_species_lobuli_comparison(data_dict: Dict[str, Dict[str, Dict[str, pd.Series]]],
                                   report_path: Path,
                                   attributes: List[str],
                                   labels: List[str],
                                   logs: List[bool],
                                   units: List[str]):
    fig, axes = plt.subplots(len(attributes), 1, dpi=300,
                             figsize=(5, len(attributes) * 2),
                             layout="constrained")
    anno_ax_size = 0.08

    for i, (attr, ax, log, y_label, unit) in enumerate(zip(attributes, axes, logs, labels, units)):
        annotate_group = True if i == 0 else False
        annotate_n = True if i == len(attributes) - 1 else False
        box_plot_species_comparison(data_dict,
                                    attr,
                                    y_label=y_label,
                                    colors=SPECIES_COLORS_RGB,
                                    log=log,
                                    ax=ax,
                                    annotate_n=annotate_n,
                                    annotate_group=annotate_group,
                                    anno_ax_size=anno_ax_size,
                                    unit=unit,
                                    show_violins=True,
                                    plot_fc=True)

    fig.tight_layout()

    plt.savefig(report_path / "species_comparison.png", dpi=600)
    plt.savefig(report_path / "species_comparison.svg", dpi=600)


    plt.show()


def species_lobuli_comparison(
        slide_stats_df_steatosis: pd.DataFrame,
        slide_stats_df_control: pd.DataFrame,
        report_path: Path,
        attributes: List[str],
        labels: List[str],
        logs: List[bool],
        units: List[str]):
    slide_stats_df_control["group"] = "control"

    slide_stats_df_steatosis["diet"] = slide_stats_df_steatosis["subject"].map(DIET)
    slide_stats_df_steatosis["group"] = slide_stats_df_steatosis.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)
    slide_stats_df_control["group"] = "Control"

    df = pd.concat([slide_stats_df_control, slide_stats_df_steatosis], ignore_index=True)

    data_dict = create_data_dict(attributes, df)

    stats = create_stats_from_data_dict(data_dict=data_dict)
    stats.to_excel(report_path / 'lobuli_species_comparison.xlsx')

    plot_species_lobuli_comparison(data_dict,
                                   report_path,
                                   attributes,
                                   labels, logs, units)


def plot_fold_change(data_dict: Dict[str, Dict[str, Dict[str, pd.Series]]],
                     report_path: Path,
                     attributes: List[str],
                     labels: List[str],
                     units: List[str]):
    fig, axes = plt.subplots(len(attributes), 1, dpi=300,
                             figsize=(5, len(attributes) * 2),
                             layout="constrained")

    for i, (attr, ax, log, y_label, unit) in enumerate(zip(attributes, axes, labels, units)):
        len_groups = sum([len(x) for x in data_dict.values()])
        in_axes = []
        x = 0

        for k, (sp, group_dict) in enumerate(data_dict.items()):

            n_sub_groups = len(group_dict)

            width = n_sub_groups / len_groups
            in_ax = ax.inset_axes((x, 0, width, 1), transform=ax.transAxes)
            in_axes.append(in_ax)
            x += width

            if i != 0:
                in_ax.yaxis.set_tick_params(which='both', labelleft=False)
            else:
                in_ax.yaxis.set_tick_params(which='minor', labelleft=False)
            if i == 0:
                in_ax.set_ylabel(f"{capitalize(y_label)} ({unit})")

            in_ax.xaxis.set_tick_params(which='both', labelbottom=False)

            # Customize the appearance of the ticks (optional)
            if i != 0:
                in_ax.tick_params(axis='both', which='both', length=0, width=0)
            else:
                in_ax.tick_params(axis='x', which='both', length=0, width=0)

            control = group_dict["Control"][attr]
            folds = []
            for l, (gr, attr_dict) in enumerate(group_dict.items()):
                med = np.median(attr_dict[attr])

                fold_c = med / np.median(control)
                folds.append(fold_c)

            in_ax.plot(range(len(folds)),
                       folds,
                       marker="o",
                       linestyle="-",
                       color=SPECIES_COLORS_RGB[sp],
                       markerfacecolor=SPECIES_COLORS_RGB[sp],
                       markeredgecolor="black"
                       )

    plt.savefig(report_path / "lobuli_comparison_fold_change.png", dpi=600)
    plt.savefig(report_path / "lobuli_comparison_fold_change.svg", dpi=600)

    plt.show()
