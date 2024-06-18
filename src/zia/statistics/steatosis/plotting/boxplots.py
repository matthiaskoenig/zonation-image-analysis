from collections.abc import Callable
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from zia.statistics.steatosis.utils import map_to_group
from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize
from zia.statistics.lobulus_geometry.plotting.plot_significance import plot_significance
import seaborn as sbn


def identity(x):
    return x


def create_subplots() -> Tuple[plt.Figure, plt.Axes]:
    return plt.subplots(1, 1, dpi=600)


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
        bplot = violin_plot(data_dict.values(), ax)

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
                                annotate_n=True,
                                unit: Optional[str] = None
                                ):
    data_dict = {}

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
        bplot = violin_plot(data_dict.values(), ax)

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
                                colors: Dict[str, Tuple[float]],
                                unit: str,
                                log=False,
                                show_violins: bool = False,
                                ax: plt.Axes = None,
                                test_results=None,
                                annotate_n=True,
                                annotate_group=True,
                                anno_ax_size=0.07
                                ) -> None:
    data_colors = []

    len_groups = len(pd.unique(df["group"]))

    for g in pd.unique(df["group"]):
        for species in species_order:
            if species in g:
                data_colors.append(colors[species])

    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=600)
    ax: plt.Axes

    ax.set_xticks([])
    ax.set_yticks([])

    print(len(data_colors))

    species_gb = df.groupby("species")

    x = 0

    in_axes = []
    for i, sp in enumerate(species_order):

        data_dict = {}
        sp_df = species_gb.get_group(sp)

        for gr, gr_df in sp_df.groupby("group"):
            data_dict[gr] = gr_df[attribute]

        # sp_df = sp_df[sp_df[attribute] < sp_df['area'].quantile(0.995)]
        n_sub_groups = len(data_dict)

        width = n_sub_groups / len_groups
        in_ax = ax.inset_axes((x, 0, width, 1), transform=ax.transAxes)
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

        bplot, vplot = violin_plot(data=data_dict.values(), log=log, ax=in_ax, show_violins=show_violins)

        for patch in bplot['boxes']:
            patch.set_facecolor(colors[sp] + (1 if vplot is not None else 0.3,))

        if vplot is not None:
            for pcol in vplot["bodies"]:
                pcol.set_facecolor(colors[sp] + (0.3,))

        group_gb = sp_df.groupby("group")

        if annotate_n:
            n_axes = in_ax.inset_axes((0, -anno_ax_size, 1, anno_ax_size), transform=in_ax.transAxes)

            for i, (group, group_df) in enumerate(group_gb):
                n = len(group_df)

                if n > 9999:
                    s_n = f"{round(n / 1000)}k"
                else:
                    s_n = str(n)

                n_axes.text((i + 1) / n_sub_groups - 1 / 2 * 1 / n_sub_groups,
                            0,
                            s=s_n,
                            ha="center",
                            va="bottom",
                            fontsize=8)
                n_axes.fill_betweenx(y=[0, 1], x1=i / n_sub_groups, x2=(i + 1) / n_sub_groups, color="white" if i % 2 == 0 else "whitesmoke")
            n_axes.set_xlim(left=0, right=1)
            n_axes.set_yticks([])
            n_axes.set_xticks([0.5], [capitalize(sp)])

        if annotate_group:
            gr_axes = in_ax.inset_axes((0, 1, 1, anno_ax_size), transform=in_ax.transAxes)

            for i, (group, group_df) in enumerate(group_gb):
                diet = pd.unique(group_df["diet"])

                w = f"{diet[0]}W" if not pd.isna(diet[0]) else "Control" if group == "control" else np.nan
                if not pd.isna(w):
                    gr_axes.text((i + 1) / n_sub_groups - 1 / 2 * 1 / n_sub_groups,
                                 0,
                                 s=w,
                                 ha="center",
                                 va="bottom",
                                 fontsize=8)

                gr_axes.fill_betweenx(y=[0, 1], x1=i / n_sub_groups, x2=(i + 1) / n_sub_groups, color="white" if i % 2 == 0 else "whitesmoke")
            gr_axes.set_xlim(left=0, right=1)
            gr_axes.set_yticks([])
            gr_axes.set_xticks([])

    mins = [in_ax.get_ylim()[0] for in_ax in in_axes]
    maxs = [in_ax.get_ylim()[1] for in_ax in in_axes]

    for in_ax in in_axes:
        in_ax.set_ylim(
            bottom=min(mins),
            top=max(maxs)
        )


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def violin_plot(data: List[pd.Series], ax: plt.Axes, log: bool = False, show_violins=True,
                positions: List[float] = None,
                widths: float = None,
                whis=None,
                ) -> Tuple[dict, Optional[dict]]:
    transformer = lambda x: np.log(x) if log else x
    reverser = lambda x: np.exp(x) if log else x

    bxpstats = []
    vpstats = []

    for d in data:
        d = transformer(d)
        q1, median, q3 = np.percentile(d, [25, 50, 75])

        if type(whis) is tuple:
            whislo, whishi = np.percentile(d.values, whis)
        else:
            whislo, whishi = adjacent_values(sorted(d.values), q1, q3)

        bxpstats.append(
            dict(med=reverser(median),
                 q1=reverser(q1),
                 q3=reverser(q3),
                 whislo=reverser(whislo),
                 whishi=reverser(whishi))
        )

        if show_violins:
            min_val = np.min(d)
            max_val = np.max(d)
            mean = np.mean(d)

            kde = gaussian_kde(d)
            val_range = max_val - min_val
            coords = np.linspace(min_val - 0.1 * val_range, max_val + 0.1 * val_range, 100)

            vpstats.append(
                dict(coords=reverser(coords),
                     vals=kde(coords),
                     mean=reverser(mean),
                     median=reverser(median),
                     min=reverser(min_val),
                     max=reverser(max_val)
                     )
            )

    if show_violins:
        violin = ax.violin(vpstats,
                           positions=positions,
                           widths=0.8, showmeans=False,
                           showextrema=False, showmedians=False)
    else:
        violin = None

    if widths is None:
        widths = 0.3 if show_violins else 0.66

    bplot = ax.bxp(bxpstats,
                   positions=positions,
                   showfliers=False,
                   showcaps=False,
                   widths=widths,
                   medianprops=dict(color="white" if show_violins else "black"),
                   patch_artist=True,
                   )

    if log:
        ax.set_yscale("log")

    return bplot, violin
