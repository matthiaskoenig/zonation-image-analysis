from pathlib import Path
from typing import Tuple, List, Optional, Callable

import pandas as pd
from matplotlib import pyplot as plt

from zia.statistics.lobulus_geometry.boxplots_species_comparison import identity
from zia.statistics.utils.data_provider import SlideStatsProvider


def scatter_plot_correlation(df: pd.DataFrame,
                             attr: Tuple[str, str],
                             species_order: list[str],
                             colors: [List[Tuple[int]]],
                             labels: Optional[Tuple[str, str]] = None,
                             report_path: Optional[Path] = None,
                             log: Tuple[bool, bool] = (False, False),
                             fun: Tuple[Callable, Callable] = None):
    x_attr, y_attr = attr
    if fun is None:
        x_fun, y_fun = identity, identity
    else:
        x_fun, y_fun = fun
    x_label, y_label = labels if labels is not None else attr
    groupby = df.groupby(by="species")
    units = None

    fig, ax = plt.subplots(1, 1, dpi=600)
    ax: plt.Axes

    for species, c in zip(species_order, colors):
        species_df = groupby.get_group(species)
        x, y = x_fun(species_df[x_attr]), y_fun(species_df[y_attr])
        if units is None:
            units = set(species_df[f"{x_attr}_unit"]).pop(), set(species_df[f"{y_attr}_unit"]).pop()

        ax.scatter(x, y, marker="o", color=c, alpha=0.3, label=species)
        ax.scatter(x, y, marker="o", color="None", edgecolor=c)

    x_unit, y_unit = units
    ax.set_xlabel(f"{x_label} ({x_unit})")
    ax.set_ylabel(f"{y_label} ({y_unit})")

    ax.legend(frameon=False)

    if log is not None:
        x_log, y_log = log

        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")

    if report_path is not None:
        plt.savefig(report_path / f"species_cor_{x_attr}_{y_attr}.jpeg")
    plt.show()


def visualize_species_correlation(df: pd.DataFrame, species_oder: list[str], colors: [List[Tuple[int]]], report_path: Path = None):
    scatter_plot_correlation(df, ("perimeter", "area"), species_oder, colors, report_path=report_path, log=(True, True))
    scatter_plot_correlation(df, ("perimeter", "compactness"), species_oder, colors, report_path=report_path, log=(True, False))
    scatter_plot_correlation(df, ("area", "compactness"), species_oder, colors, report_path=report_path, log=(True, False))


if __name__ == "__main__":
    a = 0.5
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("boxplots")

    visualize_species_correlation(df, SlideStatsProvider.species_order, SlideStatsProvider.colors, report_path)
