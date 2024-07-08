from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from zia.statistics.lobulus_geometry.testing import test_kruskal, test_levenne, test_dunns
from zia.statistics.steatosis.utils.get_data import DIET
from zia.statistics.steatosis.utils.boxplots import box_plot_species_comparison
from zia.statistics.steatosis.utils.descriptive_stats import create_stats_from_data_dict
from zia.statistics.steatosis.utils.utils import SPECIES_COLORS_RGB, create_data_dict, SPECIES_ORDER, GroupTestResults, TestResult
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
                                   units: List[str],
                                   test_results: GroupTestResults):
    fig, axes = plt.subplots(len(attributes), 2, dpi=300,
                             width_ratios=(3, 1),
                             figsize=(6.5, len(attributes) * 2),
                             layout="constrained")
    anno_ax_size = 0.08

    for i, (attr, ax_row, log, y_label, unit) in enumerate(zip(attributes, axes, logs, labels, units)):
        annotate_group = True if i == 0 else False
        annotate_n = True if i == len(attributes) - 1 else False
        box_plot_species_comparison(data_dict,
                                    attr,
                                    y_label=y_label,
                                    colors=SPECIES_COLORS_RGB,
                                    log=log,
                                    ax=ax_row[0],
                                    annotate_n=annotate_n,
                                    annotate_group=annotate_group,
                                    anno_ax_size=anno_ax_size,
                                    unit=unit,
                                    show_violins=True,
                                    plot_fc=False,
                                    test_results=test_results)

    plot_fold_change(data_dict,
                     attributes,
                     axes[:, 1],
                     test_results)

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
    slide_stats_df_steatosis["diet"] = slide_stats_df_steatosis["subject"].map(DIET)
    slide_stats_df_steatosis["group"] = slide_stats_df_steatosis.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)
    slide_stats_df_control["group"] = "Control"

    df = pd.concat([slide_stats_df_control, slide_stats_df_steatosis], ignore_index=True)

    data_dict = create_data_dict(attributes, df)

    test_results = test_between_groups(df, attributes, logs, report_path)

    stats = create_stats_from_data_dict(data_dict=data_dict)
    stats.to_excel(report_path / 'lobuli_species_comparison.xlsx')

    plot_species_lobuli_comparison(data_dict,
                                   report_path,
                                   attributes,
                                   labels, logs, units,
                                   test_results)


def test_between_groups(slide_stats_df: pd.DataFrame,
                        attributes: List[str],
                        logs: List[bool],
                        report_path: Path) -> GroupTestResults:
    # subject comparison
    with pd.ExcelWriter(report_path / "test-group-comparison.xlsx") as subject_writer:
        species_gb = slide_stats_df.groupby("species")

        results = {}
        for species in SPECIES_ORDER:
            species_df = species_gb.get_group(species)
            kruskal_results_subjects = test_kruskal("group", attributes, species_df, logs)
            kruskal_results_subjects.to_excel(subject_writer, f"kruskal-wallis-{species}", index=False)

            levennes_result = test_levenne("group", attributes, species_df, logs)
            levennes_result.to_excel(subject_writer, f"levenne-{species}", index=False)

            dunns_subject = test_dunns("group", attributes, species_df, logs)
            dunns_subject.to_excel(subject_writer, f"dunns-{species}", index=False)

            result = TestResult(kruskal=kruskal_results_subjects,
                                dunns=dunns_subject)

            results[species] = result

    return GroupTestResults(results)


def plot_fold_change(data_dict: Dict[str, Dict[str, Dict[str, pd.Series]]],
                     attributes: List[str],
                     axes: List[plt.Axes],
                     group_test_results: GroupTestResults
                     ):
    marker_dict = {
        "4W": "s",
        "2W": "D",
        "Stea.": "v"
    }

    for ax, attr in zip(axes, attributes):
        fold_change = []
        pvals = []
        markers = []
        colors = []
        labels = []

        for species in SPECIES_ORDER:

            test_result = group_test_results.test_result[species]

            group_dict = data_dict[species]

            control = group_dict["Control"]

            dunns_result = test_result.dunns
            attr_result = dunns_result[dunns_result["attr"] == attr]

            for gr, attr_dict in group_dict.items():
                if gr == "Control":
                    continue

                pval = attr_result[((attr_result.group1 == "Control") & (attr_result.group2 == gr)) | (
                        (attr_result.group1 == gr) & (attr_result.group2 == "Control"))].iloc[0]["pvalue"]

                median_control = np.median(control[attr])
                median_compare = np.median(attr_dict[attr])

                fold_change.append(((median_compare - median_control) / abs(median_control)) * 100)
                pvals.append(pval)

                markers.append(marker_dict[gr])
                colors.append(SPECIES_COLORS_RGB[species])
                labels.append(f"{species} ({gr})")

        for x, y, m, c, l in zip(fold_change, pvals, markers, colors, labels):
            ax.plot(x, -np.log10(y), marker=m, markerfacecolor=c, markeredgecolor="black", label=l, zorder=10)

    for ax in axes:
        ax: plt.Axes

        ax.axhline(y=-np.log10(0.05), xmin=0, xmax=1,
                   color="red",
                   linestyle="dashed", zorder=0)

        ax.axvline(x=0, ymin=0, ymax=1,
                   color="grey",
                   linestyle="dashed",
                   zorder=0)

    min_x, max_x = [], []

    for ax in axes:
        min_x.append(ax.get_xlim()[0])
        max_x.append(ax.get_xlim()[1])

    min_x = min(min_x)
    max_x = max(max_x)

    for ax in axes:
        ax.set_xlim(left=-max(abs(min_x), abs(max_x)) * 1.1, right=max(abs(min_x), abs(max_x)) * 1.1)

    for ax in axes[:-1]:
        ax.xaxis.set_ticks([], [])

    axes[-1].set_xlabel("Median change (%)")

    for ax in axes:
        ax.yaxis.tick_right()
        ax.set_ylabel("-log10 p-value")
        ax.yaxis.set_label_position("right")

    lines = [
        Line2D([0], [0],
               markeredgecolor="black",
               linestyle="none",
               markerfacecolor="white",
               label="2W",
               marker="D",
               markersize=5),
        Line2D([0], [0],
               markeredgecolor="black",
               linestyle="none",
               markerfacecolor="white",
               label="4W",
               marker="s",
               markersize=5),

        Line2D([0], [0],
               markeredgecolor="black",
               linestyle="none",
               markerfacecolor="white",
               label="Stea.",
               marker="v",
               markersize=5),

    ]

    axes[0].legend(handles=lines, frameon=False, ncol=3, handletextpad=0.05,
                   loc="lower center",
                   columnspacing=0.2, bbox_to_anchor=(0.5, 0.95),
                   fontsize=8)
