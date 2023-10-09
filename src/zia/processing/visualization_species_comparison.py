import re
from collections.abc import Callable
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zia import BASE_PATH
from zia.config import read_config
from zia.processing.lobulus_statistics import SlideStats


def identity(x):
    return x


def create_subplots() -> Tuple[plt.Figure, plt.Axes]:
    return plt.subplots(1, 1, dpi=600)


def visualize_species_comparison(df: pd.DataFrame, species_oder: list[str], colors: [List[Tuple[int]]], report_path: Path = None):
    box_plot_species_comparison(df, "area", "area", species_oder, colors, report_path=report_path, log=True)
    box_plot_species_comparison(df, "compactness", "compactness", species_oder, colors, report_path=report_path, limits=(0, 1))
    box_plot_species_comparison(df, "perimeter", "perimeter", species_oder, colors, report_path=report_path)

    violin_plot_species_comparison(df, "area", "area", report_path=report_path, axis_cut_off_percentile=0.01)
    violin_plot_species_comparison(df, "compactness", "compactness", report_path=report_path, limits=(0, 1))
    violin_plot_species_comparison(df, "perimeter", "perimeter", report_path=report_path)
    pass


def visualize_subject_comparison(df: pd.DataFrame, species_oder: list[str], colors: [List[Tuple[int]]], report_path: Path = None) -> None:
    groupby = df.groupby("species")
    for species, color in zip(species_oder, colors):
        species_df = groupby.get_group(species)
        box_plot_subject_comparison(species_df, species, "area", "area", report_path=report_path, log=True, color=color)
        box_plot_subject_comparison(species_df, species, "compactness", "compactness", report_path=report_path, limits=(0, 1), color=color)
        box_plot_subject_comparison(species_df, species, "perimeter", "perimeter", report_path=report_path, color=color)


def box_plot_subject_comparison(species_df: pd.DataFrame,
                                species: str,
                                attribute: str,
                                y_label: str,
                                report_path: Path = None,
                                log=False,
                                cut_off_percentile=None,
                                limits=None,
                                color=(0, 0, 0)):
    data_dict = {}

    unit = None

    for subject, subject_df in species_df.groupby("subject"):
        data_dict[str(subject)] = subject_df[attribute]
        if unit is None:
            unit = set(species_df[f"{attribute}_unit"]).pop()

    fig, ax = create_subplots()
    fig.suptitle(species)

    if not log:
        bplot = ax.boxplot(data_dict.values(),
                           showfliers=False,
                           showcaps=False,
                           medianprops=dict(color="black"),
                           patch_artist=True)
    else:
        bplot = box_plot_log(data_dict, ax)

    for patch in bplot['boxes']:
        patch.set_facecolor(color + (0.3,))

    for i, (subject, data) in enumerate(data_dict.items()):
        x_scatter = np.random.normal(i + 1, 0.05, size=len(data))
        ax.scatter(x_scatter,
                   data,
                   color=color + (0.5,),
                   s=1)

    if cut_off_percentile is not None:
        p0s = []
        p1s = []
        for data_values in data_dict.values():
            p0 = np.percentile(data_values, q=cut_off_percentile * 100)
            p1 = np.percentile(data_values, q=(1 - cut_off_percentile) * 100)

            p0s.append(p0)
            p1s.append(p1)

        print(p0s)
        print(p1s)

        ax.set_ylim(bottom=min(p0s), top=max(p1s))

    if limits is not None:
        ax.set_ylim(limits)

    ax.set_xticklabels([k.replace("_Human", "") for k in data_dict.keys()])
    ax.set_ylabel(f"{y_label} ({unit})")

    if report_path is not None:
        plt.savefig(report_path / f"subjects_{species}_{attribute}.jpeg")

    plt.show()


def box_plot_species_comparison(df: pd.DataFrame,
                                attribute: str,
                                y_label: str,
                                species_order: List[str],
                                colors: List[Tuple[int]],
                                report_path: Path = None,
                                log=False,
                                axis_cut_off_percentile=None,
                                limits=None):
    data_dict = {}
    species_subject_dict = {}

    unit = None

    groupby = df.groupby(by="species")

    for species in species_order:
        species_df = groupby.get_group(species)
        data_dict[str(species)] = species_df[attribute]
        if unit is None:
            unit = set(species_df[f"{attribute}_unit"]).pop()

        subject_data = {}

        for subject, subject_df in species_df.groupby(by="subject"):
            subject_data[subject] = subject_df[attribute]

        species_subject_dict[species] = subject_data

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax: plt.Axes

    if not log:
        bplot = ax.boxplot([data_dict[s] for s in species_order],
                           showfliers=False,
                           showcaps=False,
                           medianprops=dict(color="black"),
                           patch_artist=True)
    else:
        bplot = box_plot_log(data_dict, ax)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color + (0.3,))

    for i, (species, subject_dict) in enumerate(species_subject_dict.items()):
        for subject, data in subject_dict.items():
            x_scatter = np.random.normal(i + 1, 0.05, size=len(data))
            ax.scatter(x_scatter,
                       data,
                       color=colors[i] + (0.5,),
                       s=1)

    if axis_cut_off_percentile is not None:
        p0s = []
        p1s = []
        for data_values in data_dict.values():
            p0 = np.percentile(data_values, q=axis_cut_off_percentile * 100)
            p1 = np.percentile(data_values, q=(1 - axis_cut_off_percentile) * 100)

            p0s.append(p0)
            p1s.append(p1)

        print(p0s)
        print(p1s)

        ax.set_ylim(bottom=min(p0s), top=max(p1s))

    if limits is not None:
        ax.set_ylim(limits)

    ax.set_xticklabels(species_order)
    ax.set_ylabel(f"{y_label} ({unit})")

    if report_path is not None:
        plt.savefig(report_path / f"species_{attribute}.jpeg")
    plt.show()


def violin_plot_species_comparison(df: pd.DataFrame,
                                   attribute: str,
                                   y_label: str,
                                   fun: Callable[[pd.Series], pd.Series] = None,
                                   report_path: Path = None,
                                   log=False,
                                   axis_cut_off_percentile=0.01,
                                   limits=None):
    species_order = ["mouse", "rat", "pig", "human"]
    a = 0.5
    colors = [(102 / 255, 194 / 255, 165 / 255),
              (252 / 255, 141 / 255, 98 / 255),
              (141 / 255, 160 / 255, 203 / 255),
              (231 / 255, 138 / 255, 195 / 255)]
    data_dict = {}
    species_subject_dict = {}

    if fun is None:
        fun = identity

    unit = None
    for species, species_df in df.groupby(by="species"):
        data_dict[species] = fun(species_df[attribute])
        if unit is None:
            unit = set(species_df[f"{attribute}_unit"]).pop()

        subject_data = {}

        for subject, subject_df in species_df.groupby(by="subject"):
            subject_data[subject] = fun(subject_df[attribute])

        species_subject_dict[species] = subject_data

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax: plt.Axes

    data_plot = [data_dict[s].values for s in species_order]

    quantiles = [np.percentile(d, [25, 50, 75]) for d in data_plot]

    vplots = ax.violinplot(data_plot, showmedians=False, showextrema=False)

    for pc, c in zip(vplots["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.8)
        pc.set_edgecolor("black")

    for i, (q1, median, q3) in enumerate(quantiles):
        whiskers_min, whiskers_max = adjacent_values(sorted(data_plot[i]), q1, q3)

        ax.scatter(i + 1, median, marker='o', color='white', s=15, zorder=3)
        ax.vlines(i + 1, q1, q3, color='k', linestyle='-', lw=5)
        ax.vlines(i + 1, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    if log:
        ax.set_yscale("log")

    if limits is not None:
        ax.set_ylim(limits)

    ax.set_xticks(np.arange(1, len(data_plot) + 1), species_order)
    ax.set_ylabel(f"{y_label} ({unit})")

    if report_path is not None:
        plt.savefig(report_path / f"species_violin_{attribute}.jpeg")
    plt.show()


def get_species_from_name(subject) -> Optional[str]:
    """Metadata for image"""
    rat_pattern = re.compile("NOR-\d+")
    pig_pattern = re.compile("SSES2021 \d+")
    mouse_pattern = re.compile("MNT-\d+")
    human_pattern = re.compile("UKJ-19-\d+_Human")
    if re.search(pig_pattern, subject):
        return "pig"
    if re.search(mouse_pattern, subject):
        return "mouse"
    if re.search(rat_pattern, subject):
        return "rat"
    if re.search(human_pattern, subject):
        return "human"

    return None


def merge_to_one_df(slide_stats: Dict[str, Dict[str, SlideStats]]) -> pd.DataFrame:
    dfs = []
    for subject, rois in slide_stats.items():
        species = get_species_from_name(subject)

        for roi, slide_stat in rois.items():
            slide_stat_df = slide_stat.to_dataframe()
            slide_stat_df["species"] = species
            slide_stat_df["subject"] = subject
            slide_stat_df["roi"] = roi

            dfs.append(slide_stat_df)

    return pd.concat(dfs, ignore_index=True)


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def box_plot_log(data_dict: Dict[str, pd.Series], ax: plt.Axes) -> dict:
    bxpstats = []
    for d in data_dict.values():
        d = np.log(d)
        q1, median, q3 = np.percentile(d, [25, 50, 75])
        whislo, whishi = adjacent_values(sorted(d.values), q1, q3)

        bxpstats.append(
            dict(med=np.exp(median),
                 q1=np.exp(q1),
                 q3=np.exp(q3),
                 whislo=np.exp(whislo),
                 whishi=np.exp(whishi))
        )

    bplot = ax.bxp(bxpstats,
                   showfliers=False,
                   showcaps=False,
                   medianprops=dict(color="black"),
                   patch_artist=True
                   )

    ax.set_yscale("log")
    return bplot


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    data_dir_stain_separated = config.image_data_path / "slide_statistics"
    report_path = config.reports_path / "boxplots"
    report_path.mkdir(parents=True, exist_ok=True)
    subject_dirs = sorted([f for f in data_dir_stain_separated.iterdir() if f.is_dir() and not f.name.startswith(".")])

    slide_stats = {}

    species_order = ["mouse", "rat", "pig", "human"]

    a = 0.5
    colors = [(102 / 255, 194 / 255, 165 / 255),
              (252 / 255, 141 / 255, 98 / 255),
              (141 / 255, 160 / 255, 203 / 255),
              (231 / 255, 138 / 255, 195 / 255)]

    for subject_dir in subject_dirs:
        subject = subject_dir.stem
        roi_dict = {}
        # print(subject)

        roi_dirs = sorted([f for f in subject_dir.iterdir() if f.is_dir()])
        for roi_dir in roi_dirs:
            roi = roi_dir.stem
            # print(roi)
            roi_dict[roi] = SlideStats.load_from_file_system(roi_dir)

        slide_stats[subject] = roi_dict

    df = merge_to_one_df(slide_stats)
    print(df.columns)
    visualize_species_comparison(df, species_order, colors, report_path)
    # visualize_subject_comparison(df, species_order, colors, report_path)
