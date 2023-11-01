from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.stats import spearmanr, pearsonr


def scatter_plot_correlation(df: pd.DataFrame,
                             color: Tuple[int],
                             attr: Tuple[str, str],
                             log: Tuple[bool, bool] = (False, False),
                             labels: Optional[Tuple[str, str]] = None,
                             limits_x=Tuple[float, float],
                             limits_y=Tuple[float, float],
                             report_path: Optional[Path] = None,
                             ax: plt.Axes = None,
                             scatter=True
                             ):
    x_attr, y_attr = attr

    x_label, y_label = labels if labels is not None else attr

    units = None

    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=300)
        ax: plt.Axes

    x, y = df[x_attr], df[y_attr]
    if units is None:
        units = set(df[f"{x_attr}_unit"]).pop(), set(df[f"{y_attr}_unit"]).pop()

    if scatter:
        ax.scatter(x, y, marker="o", color=color, alpha=0.3)
        ax.scatter(x, y, marker="o", color="None", edgecolor=color)

        ax.set_xlim(limits_x)
        ax.set_ylim(limits_y)

    x_log, y_log = False, False
    if log is not None:
        x_log, y_log = log

        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")

    if x_log and limits_x is not None:
        limits_x = tuple([np.log10(x) for x in list(limits_x)])

    if y_log and limits_y is not None:
        limits_y = tuple([np.log10(y) for y in list(limits_y)])

    if not scatter:
        colors = [color + (0,), color + (1,)]
        cmap = LinearSegmentedColormap.from_list("whatever", colors)
        ax.hexbin(x=x, y=y,
                  xscale="log" if x_log else "linear",
                  yscale="log" if y_log else "linear",
                  gridsize=15,
                  linewidths=0,
                  bins="log",
                  cmap=cmap,
                  extent=limits_x + limits_y
                  )
    x_unit, y_unit = units
    ax.set_xlabel(f"{x_label} ({x_unit})")
    ax.set_ylabel(f"{y_label} ({y_unit})")

    spearman_corr = spearmanr(x, y).statistic

    # spearman_corr_log = spearmanr(np.log(x) if x_log else x, np.log(y) if y_log else y)
    # pearson_corr = pearsonr(x, y)
    # pearson_corr_log = pearsonr(np.log(x) if x_log else x, np.log(y) if y_log else y)

    if spearman_corr < 0:
        x = 0.02
        ha = "left"
    else:
        x = 0.98
        ha = "right"

    ax.text(x=x, y=0.02, s=f"r={spearman_corr:.2f}", fontsize=12, ha=ha, transform=ax.transAxes)

    if report_path is not None:
        plt.savefig(report_path / f"species_cor_{x_attr}_{y_attr}.jpeg")
        plt.show()
