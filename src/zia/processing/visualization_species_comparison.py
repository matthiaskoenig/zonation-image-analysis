import re
from collections.abc import Callable
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def visualize_species_comparison(df: pd.DataFrame, report_path: Path = None):
    box_plot_species_comparison(df, "area", "log2 area", np.log2, report_path=report_path)
    box_plot_species_comparison(df, "compactness", "compactness", report_path=report_path)
    box_plot_species_comparison(df, "perimeter", "log2 perimeter", np.log2, report_path=report_path)
    pass


def visualize_subject_comparison(df: pd.DataFrame, report_path: Path = None) -> None:
    for species, species_df in df.groupby("species"):
        box_plot_subject_comparison(species_df, species, "area", "log2 area", np.log2, report_path=report_path)
        box_plot_subject_comparison(species_df, species, "compactness", "compactness", report_path=report_path)


def box_plot_subject_comparison(species_df: pd.DataFrame,
                                species: str,
                                attribute: str,
                                y_label: str,
                                fun: Callable[[pd.Series], pd.Series] = None,
                                report_path: Path = None):
    if fun is None:
        fun = identity

    data = []
    tick_labels = []

    for subject, subject_df in species_df.groupby("subject"):
        data.append(fun(subject_df[attribute]))
        tick_labels.append(subject)

    fig, ax = create_subplots()
    fig.suptitle(species)
    ax.boxplot(data, showfliers=False)

    for i, data_set in enumerate(data):
        x_scatter = np.random.normal(i + 1, 0.05, size=len(data_set))
        ax.scatter(x_scatter, data_set, c="blue", alpha=0.1)

    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(y_label)

    if report_path is not None:
        plt.savefig(report_path / f"subjects_{species}_{attribute}.jpeg")

    plt.show()


def box_plot_species_comparison(df: pd.DataFrame,
                                attribute: str,
                                y_label: str,
                                fun: Callable[[pd.Series], pd.Series] = None,
                                report_path: Path = None):
    data = []
    tick_labels = []
    subject_datas = []

    if fun is None:
        fun = identity

    for species, species_df in df.groupby(by="species"):
        data.append(fun(species_df[attribute]))
        tick_labels.append(species)

        subject_data = []
        subjects = []

        for subject, subject_df in species_df.groupby(by="subject"):
            subject_data.append(fun(subject_df[attribute]))
            subjects.append(subject)

        subject_datas.append(subject_data)

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax: plt.Axes
    ax.boxplot(data, showfliers=False)

    for i, subject_data in enumerate(subject_datas):
        for data in subject_data:
            x_scatter = np.random.normal(i + 1, 0.05, size=len(data))
            ax.scatter(x_scatter, data, c="blue", alpha=0.1)

    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(y_label)

    if report_path is not None:
        plt.savefig(report_path / f"species_{attribute}.jpeg")
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


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")
    data_dir_stain_separated = config.image_data_path / "slide_statistics"
    report_path = config.reports_path / "boxplots"
    report_path.mkdir(parents=True, exist_ok=True)
    subject_dirs = sorted([f for f in data_dir_stain_separated.iterdir() if f.is_dir() and not f.name.startswith(".")])

    slide_stats = {}

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
    visualize_species_comparison(df, report_path)
    visualize_subject_comparison(df, report_path)
