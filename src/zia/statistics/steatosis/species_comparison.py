from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from zia.statistics.steatosis.get_data import DIET
from zia.statistics.steatosis.plotting.boxplots import box_plot_species_comparison
from zia.statistics.steatosis.utils import SPECIES_COLORS_RGB, SPECIES_ORDER, map_to_group
from zia.statistics.utils.data_provider import SlideStatsProvider


def plot_species_droplet_comparison(droplet_stats_df: pd.DataFrame,
                                    report_path: Path,
                                    attributes: List[str],
                                    labels: List[str],
                                    logs: List[bool],
                                    # test_results_path: Path,
                                    units: List[str]):
    # kruskal_result = pd.read_excel(test_results_path / f"test-species-comparison.xlsx", sheet_name="kruskal-wallis", index_col=False)
    # dunns_result = pd.read_excel(test_results_path / "test-species-comparison.xlsx", sheet_name="dunns-post-hoc", index_col=False)
    droplet_stats_df["group"] = droplet_stats_df.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)

    fig, axes = plt.subplots(1, len(attributes), dpi=300,
                             figsize=(len(attributes) * 2.5, 2.5),
                             layout="constrained")

    for (attr, ax, log, y_label, unit) in zip(attributes, axes, logs, labels, units):
        # if kruskal_result[kruskal_result["attr"] == attr].iloc[0]["pvalue"] < 0.05:
        #    test_results_attr = dunns_result[dunns_result["attr"] == attr]
        # else:
        #    test_results_attr = None

        box_plot_species_comparison(droplet_stats_df,
                                    attr,
                                    y_label=y_label,
                                    species_order=SPECIES_ORDER,
                                    colors=SPECIES_COLORS_RGB,
                                    log=log,
                                    ax=ax,
                                    annotate_n=True,
                                    unit=unit,
                                    show_violins=True)

    plt.savefig(report_path / "species_comparison.png", dpi=600)
    plt.savefig(report_path / "species_comparison.svg", dpi=600)

    plt.show()


def plot_species_lobuli_comparison(
        slide_stats_df_steatosis: pd.DataFrame,
        slide_stats_df_control: pd.DataFrame,
        report_path: Path,
        attributes: List[str],
        labels: List[str],
        logs: List[bool],
        # test_results_path: Path,
        units: List[str]):
    # kruskal_result = pd.read_excel(test_results_path / f"test-species-comparison.xlsx", sheet_name="kruskal-wallis", index_col=False)
    # dunns_result = pd.read_excel(test_results_path / "test-species-comparison.xlsx", sheet_name="dunns-post-hoc", index_col=False)

    slide_stats_df_control["group"] = "control"

    slide_stats_df_steatosis["diet"] = slide_stats_df_steatosis["subject"].map(DIET)
    slide_stats_df_steatosis["group"] = slide_stats_df_steatosis.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)

    df = pd.concat([slide_stats_df_control, slide_stats_df_steatosis], ignore_index=True)

    print(df.columns)
    fig, axes = plt.subplots(len(attributes), 1, dpi=300,
                             figsize=(5, len(attributes) * 2),
                             layout="constrained")
    anno_ax_size = 0.08

    for i, (attr, ax, log, y_label, unit) in enumerate(zip(attributes, axes, logs, labels, units)):
        # if kruskal_result[kruskal_result["attr"] == attr].iloc[0]["pvalue"] < 0.05:
        #    test_results_attr = dunns_result[dunns_result["attr"] == attr]
        # else:
        #    test_results_attr = None

        annotate_group = True if i == 0 else False
        annotate_n = True if i == len(attributes) - 1 else False
        box_plot_species_comparison(df,
                                    attr,
                                    y_label=y_label,
                                    species_order=SPECIES_ORDER,
                                    colors=SPECIES_COLORS_RGB,
                                    log=log,
                                    ax=ax,
                                    annotate_n=annotate_n,
                                    annotate_group=annotate_group,
                                    anno_ax_size=anno_ax_size,
                                    unit=unit,
                                    show_violins=True)

    plt.savefig(report_path / "species_comparison.png", dpi=600)
    plt.savefig(report_path / "species_comparison.svg", dpi=600)

    plt.show()


def calc_average_density(slide_stats_df: pd.DataFrame, area: float) -> float:
    return len(slide_stats_df) / (area / 1e6)  # mm^2


def calc_average_area_density(slide_stats_df: pd.DataFrame, area: float) -> float:
    return (slide_stats_df["area"].sum() / area) * 100


def plot_species_comparison_density(slide_stats_df: pd.DataFrame,
                                    wsi_df: pd.DataFrame,
                                    report_path: Path):
    # kruskal_result = pd.read_excel(test_results_path / f"test-species-comparison.xlsx", sheet_name="kruskal-wallis", index_col=False)
    # dunns_result = pd.read_excel(test_results_path / "test-species-comparison.xlsx", sheet_name="dunns-post-hoc", index_col=False)

    fig, axes = plt.subplots(1, 2, dpi=300,
                             figsize=(2 * 2.5, 2.5),
                             layout="constrained")

    attributes = ["average density", "average area density"]
    labels = ["droplet density", "droplet area fraction"]
    units = ["mm$^{-1}$", "%"]
    logs = [False, False]

    subject_roi_gb = slide_stats_df.groupby(["subject", "roi"])

    wsi_df["average density"] = wsi_df.apply(
        lambda row: calc_average_density(
            subject_roi_gb.get_group((row["subject"], row["roi"])),
            row["area"]
        ),
        axis=1
    )

    wsi_df["average area density"] = wsi_df.apply(
        lambda row: calc_average_area_density(
            subject_roi_gb.get_group((row["subject"], row["roi"])),
            row["area"]
        ),
        axis=1
    )

    wsi_df["group"] = wsi_df.apply(lambda row: map_to_group(row['species'], row['diet']), axis=1)

    for (attr, ax, log, y_label, unit) in zip(attributes, axes, logs, labels, units):
        # if kruskal_result[kruskal_result["attr"] == attr].iloc[0]["pvalue"] < 0.05:
        #    test_results_attr = dunns_result[dunns_result["attr"] == attr]
        # else:
        #    test_results_attr = None

        box_plot_species_comparison(wsi_df,
                                    attr,
                                    y_label=y_label,
                                    species_order=SPECIES_ORDER,
                                    colors=SPECIES_COLORS_RGB,
                                    log=log,
                                    ax=ax,
                                    annotate_n=True,
                                    unit=unit,
                                    show_violins=False)

    plt.savefig(report_path / "species_comparison_density.png", dpi=600)
    plt.savefig(report_path / "species_comparison_density.svg", dpi=600)

    plt.show()
