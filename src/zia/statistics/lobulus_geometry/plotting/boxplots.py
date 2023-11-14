from collections.abc import Callable
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize
from zia.statistics.lobulus_geometry.plotting.plot_significance import plot_significance


def identity(x):
    return x


def create_subplots() -> Tuple[plt.Figure, plt.Axes]:
    return plt.subplots(1, 1, dpi=600)


def visualize_species_comparison(df: pd.DataFrame, species_oder: list[str], colors: [List[Tuple[int]]], report_path: Path = None):
    box_plot_species_comparison(df, "area", "area", species_oder, colors, report_path=report_path, log=True)
    box_plot_species_comparison(df, "compactness", "compactness", species_oder, colors, report_path=report_path, limits=(0, 1))
    box_plot_species_comparison(df, "perimeter", "perimeter", species_oder, colors, report_path=report_path, log=True)
    box_plot_species_comparison(df, "minimum_bounding_radius", "minimum bounding radius", species_oder, colors, report_path=report_path, log=False)

    # violin_plot_species_comparison(df, "area", "area", report_path=report_path, axis_cut_off_percentile=0.01)
    # violin_plot_species_comparison(df, "compactness", "compactness", report_path=report_path, limits=(0, 1))
    # violin_plot_species_comparison(df, "perimeter", "perimeter", report_path=report_path)
    pass


def visualize_subject_comparison(df: pd.DataFrame, species_oder: list[str], colors: [List[Tuple[int]]], report_path: Path = None) -> None:
    groupby = df.groupby("species")
    for species, color in zip(species_oder, colors):
        species_df = groupby.get_group(species)
        box_plot_subject_comparison(species_df, species, "area", "area", report_path=report_path, log=True, color=color, ax=None,
                                    test_results=None)
        box_plot_subject_comparison(species_df, species, "compactness", "compactness", report_path=report_path, limits=(0, 1), color=color, ax=None,
                                    test_results=None)
        box_plot_subject_comparison(species_df, species, "perimeter", "perimeter", report_path=report_path, color=color, ax=None,
                                    test_results=None)


def box_plot_roi_comparison(subject_df: pd.DataFrame,
                            roi_lobule_map: dict[str, str],
                            subject: str,
                            attribute: str,
                            y_label: str,
                            report_path: Path = None,
                            log=False,
                            limits=None,
                            color=(0, 0, 0),
                            ax: plt.Axes = None,
                            test_results: pd.DataFrame = None,
                            annotate_n=True
                            ):
    data_dict: Dict = {}

    unit = None

    for roi, subject_df in subject_df.groupby("roi"):
        data_dict[str(roi)] = subject_df[attribute]
        if unit is None:
            unit = set(subject_df[f"{attribute}_unit"]).pop()

    subject_lobule_map = {k: v for k, v in roi_lobule_map.items() if subject in k and k.split("_")[-1] in data_dict.keys()}

    lobule_roi_map = {v: k.split("_")[-1] for k, v in subject_lobule_map.items()}
    # CLL: "1"
    if test_results is not None:
        test_results = test_results.copy(deep=True)

        roi_lobule_map = {v: k for k, v in lobule_roi_map.items()}
        print(roi_lobule_map)

        test_results['group1'] = test_results['group1'].astype(str).map(roi_lobule_map)
        test_results['group2'] = test_results['group2'].astype(str).map(roi_lobule_map)
        print(test_results)

    data_dict = {k: data_dict[lobule_roi_map[k]] for k in sorted(lobule_roi_map.keys())}

    if ax is None:
        fig, ax = create_subplots()
        fig.suptitle(str(roi))

    if not log:
        bplot = ax.boxplot(list(data_dict.values()),
                           showfliers=False,
                           showcaps=False,
                           widths=0.66,
                           medianprops=dict(color="black"),
                           patch_artist=True)
    else:
        bplot = box_plot_log(data_dict, ax)

    for patch in bplot['boxes']:
        patch.set_facecolor(color + (0.3,))

    if test_results is not None:
        plot_significance(ax, list(data_dict.keys()), test_results, log)

    for i, (roi, data) in enumerate(data_dict.items()):
        x_scatter = np.random.normal(i + 1, 0.05, size=len(data))
        ax.scatter(x_scatter,
                   data,
                   color=color + (0.5,),
                   s=1)

        if annotate_n:
            n_axes = ax.inset_axes((0, 0, 1, 0.05), transform=ax.transAxes)
            n_axes.text((i + 1) / len(data_dict) - 1 / 2 * 1 / len(data_dict),
                        0,
                        s=f"n={len(data_dict[roi])}",
                        ha="center",
                        va="bottom",
                        fontsize=6)
            n_axes.axis("off")
            n_axes.patch.set_alpha(0.5)
            n_axes.patch.set_facecolor("white")

    if limits is not None:
        ax.set_ylim(limits)

    ax.set_xticklabels([f"{k}" for k in data_dict.keys()])
    ax.set_ylabel(f"{capitalize(y_label)} ({unit})")

    if report_path is not None:
        plt.savefig(report_path / f"mouse_{subject}_{attribute}.jpeg")
        plt.show()


def box_plot_subject_comparison(species_df: pd.DataFrame,
                                species: str,
                                attribute: str,
                                y_label: str,
                                report_path: Path = None,
                                log=False,
                                limits=None,
                                color=(0, 0, 0),
                                ax: plt.Axes = None,
                                test_results=None,
                                annotate_n=True
                                ):
    data_dict = {}

    unit = None

    for subject, subject_df in species_df.groupby("subject"):
        data_dict[str(subject)] = subject_df[attribute]
        if unit is None:
            unit = set(species_df[f"{attribute}_unit"]).pop()

    if ax is None:
        fig, ax = create_subplots()
        fig.suptitle(species)

    if not log:
        bplot = ax.boxplot(list(data_dict.values()),
                           showfliers=False,
                           showcaps=False,
                           widths=0.66,
                           medianprops=dict(color="black"),
                           patch_artist=True)
    else:
        bplot = box_plot_log(data_dict, ax)

    for patch in bplot['boxes']:
        patch.set_facecolor(color + (0.3,))

    if test_results is not None:
        plot_significance(ax, list(data_dict.keys()), test_results, log)

    for i, (subject, data) in enumerate(data_dict.items()):
        x_scatter = np.random.normal(i + 1, 0.05, size=len(data))
        ax.scatter(x_scatter,
                   data,
                   color=color + (0.5,),
                   s=1)

        if annotate_n:
            n_axes = ax.inset_axes((0, 0, 1, 0.05), transform=ax.transAxes)
            n_axes.text((i + 1) / len(data_dict) - 1 / 2 * 1 / len(data_dict),
                        0,
                        s=f"n={len(data_dict[subject])}",
                        ha="center",
                        va="bottom",
                        fontsize=6)
            n_axes.axis("off")
            n_axes.patch.set_alpha(0.5)
            n_axes.patch.set_facecolor("white")

    if limits is not None:
        ax.set_ylim(limits)

    ax.set_xticklabels([k.replace("_Human", "") for k in data_dict.keys()])
    ax.set_ylabel(f"{capitalize(y_label)} ({unit})")

    if report_path is not None:
        plt.savefig(report_path / f"subjects_{species}_{attribute}.jpeg")
        plt.show()


def box_plot_species_comparison(df: pd.DataFrame,
                                attribute: str,
                                y_label: str,
                                species_order: List[str],
                                colors: List[Tuple[float]],
                                report_path: Path = None,
                                log=False,
                                ax: plt.Axes = None,
                                test_results=None,
                                annotate_n=True):
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

    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=600)
    ax: plt.Axes

    if not log:
        bplot = ax.boxplot([data_dict[s] for s in species_order],
                           showfliers=False,
                           showcaps=False,
                           widths=0.66,
                           medianprops=dict(color="black"),
                           patch_artist=True)
    else:
        bplot = box_plot_log(data_dict, ax)

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color + (0.3,))

    if test_results is not None:
        plot_significance(ax, species_order, test_results, log)

    n_axes = ax.inset_axes((0, -0.07, 1, 0.07), transform=ax.transAxes)
    ax.set_xticks([])
    for i, (species, subject_dict) in enumerate(species_subject_dict.items()):
        for subject, data in subject_dict.items():
            x_scatter = np.random.normal(i + 1, 0.066, size=len(data))
            ax.scatter(x_scatter,
                       data,
                       color=colors[i] + (0.5,),
                       s=1)
        if annotate_n:
            n_axes.text((i + 1) / len(data_dict) - 1 / 2 * 1 / len(data_dict),
                        0,
                        s=f"{len(data_dict[species])}",
                        ha="center",
                        va="bottom",
                        fontsize=8)

            n_axes.fill_betweenx(y=[0, 1], x1=i / len(data_dict), x2=(i + 1) / len(data_dict), color="white" if i % 2 == 0 else "whitesmoke")

        n_axes.set_xlim(left=0, right=1)

    n_axes.set_xticks([(i + 1) / len(data_dict) - 1 / 2 * 1 / len(data_dict) for i in range(len(data_dict))], [capitalize(s) for s in species_order])
    n_axes.set_yticks([])
    # n_axes.set_ylabel("n", rotation=0, va="center")
    ax.set_ylabel(f"{capitalize(y_label)} ({unit})")

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
    ax.set_ylabel(f"{capitalize(y_label)} ({unit})")

    if report_path is not None:
        plt.savefig(report_path / f"species_violin_{attribute}.jpeg")
    plt.show()


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
                   widths=0.66,
                   medianprops=dict(color="black"),
                   patch_artist=True
                   )

    ax.set_yscale("log")
    return bplot


if __name__ == "__main__":
    a = 0.5
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("boxplots")
    # print(df.columns)
    visualize_species_comparison(df, SlideStatsProvider.species_order, SlideStatsProvider.species_colors, report_path)
    # visualize_subject_comparison(df, species_order, colors, report_path)
    # visualize_species_correlation(df,SlideStatsProvider.species_order,SlideStatsProvider.colors,report_path)
