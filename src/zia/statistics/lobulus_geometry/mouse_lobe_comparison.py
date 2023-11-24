from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd

from zia.statistics.lobulus_geometry.plotting.boxplots import box_plot_subject_comparison, box_plot_roi_comparison
from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize


def plot_mouse_lobe_comparison(slide_stats_df: pd.DataFrame,
                               report_path: Path,
                               attributes: List[str],
                               labels: List[str],
                               logs: List[bool],
                               test_results_path: Path,
                               mouse_lobe_dict: Dict[str, str]):
    slide_stats_df = slide_stats_df[slide_stats_df["species"] == "mouse"]

    fig, axes = plt.subplots(len(attributes), len(set(slide_stats_df["subject"].values)), dpi=300,
                             figsize=(len(attributes) * 2.5, len(SlideStatsProvider.species_order) * 2.5),
                             sharey="row",
                             layout="constrained")

    for i, (subject, subject_df) in enumerate(slide_stats_df.groupby("subject")):
        kruskal_results = pd.read_excel(test_results_path / "test-mouse-lobe-comparison.xlsx", sheet_name=f"kruskal-wallis-mouse-lobes-{subject}",
                                        index_col=False)
        test_results = pd.read_excel(test_results_path / "test-mouse-lobe-comparison.xlsx", sheet_name=f"dunns-mouse-lobes-{subject}",
                                     index_col=False)

        for attr, ax, log, y_label in zip(attributes, axes[:, i], logs, labels):
            if kruskal_results[kruskal_results["attr"] == attr].iloc[0]["pvalue"] < 0.05:
                test_results_attr = test_results[test_results["attr"] == attr]
            else:
                test_results_attr = None
            box_plot_roi_comparison(subject_df,
                                    roi_lobule_map=mouse_lobe_dict,
                                    subject=subject,
                                    attribute=attr,
                                    y_label=y_label,
                                    color=SlideStatsProvider.get_species_colors_as_rgb()[0],
                                    log=log,
                                    ax=ax,
                                    test_results=test_results_attr,
                                    annotate_n=False)

    ax: plt.Axes

    for ax in axes[:, 1:].flatten():
        ax.set_ylabel("")
    for ax in axes[0:-1, :].flatten():
        ax.xaxis.set_ticklabels([])

    for ax in axes[-1, :]:
        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=90)

    for (subject, subject_df), ax in zip(slide_stats_df.groupby("subject"), axes[0, :].flatten()):
        ax.set_title(capitalize(subject), fontweight="bold")

    for (subject, subject_df), ax in zip(slide_stats_df.groupby("subject"), axes[-1, :].flatten()):
        counts = []
        for roi, roi_df in subject_df.groupby("roi"):
            counts.append(len(roi_df))

        in_axes = ax.inset_axes((0, -0.07, 1, 0.07), transform=ax.transAxes)
        in_axes.set_xticks([(i + 1) / len(counts) - 1 / 2 * 1 / len(counts) for i in range(len(counts))],
                           ax.get_xticklabels(),
                           rotation=90)
        ax.set_xticks([])
        in_axes.set_yticks([])

        for i, count in enumerate(counts):
            in_axes.text((i + 1) / len(counts) - 1 / 2 * 1 / len(counts),
                         0,
                         s=f"{count}",
                         ha="center",
                         va="bottom",
                         fontsize=9)

            in_axes.fill_betweenx(y=[0, 1], x1=i / len(counts), x2=(i + 1) / len(counts), color="white" if i % 2 == 0 else "whitesmoke")

        in_axes.set_xlim(left=0, right=1)

    plt.savefig(report_path / "mouse_roi_comparison.png", dpi=600)
    plt.savefig(report_path / "mouse_roi_comparison.svg", dpi=600)

    plt.show()


if __name__ == "__main__":
    mouse_lobe_dict = {
        "MNT-021_0": "LLL",
        "MNT-021_1": "ML",
        "MNT-021_2": "RL",
        "MNT-021_3": "CL",
        "MNT-022_0": "LLL",
        "MNT-022_1": "ML",
        "MNT-022_2": "RL",
        "MNT-022_3": "CL",
        "MNT-023_0": "LLL",
        "MNT-023_1": "ML",
        "MNT-023_2": "RL",
        "MNT-023_3": "N/A",
        "MNT-024_0": "LLL",
        "MNT-024_1": "ML",
        "MNT-024_2": "RL",
        "MNT-024_3": "CL",
        "MNT-025_0": "LLL",
        "MNT-025_1": "ML",
        "MNT-025_2": "RL",
        "MNT-025_3": "CL",
        "MNT-026_0": "LLL",
        "MNT-026_1": "ML",
        "MNT-026_2": "CL",
        "MNT-026_3": "RL",
    }

    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("boxplots")
    df.to_csv(report_path / "slide_statistics_df.csv", index=False)

    attributes = ["perimeter", "area", "compactness", "minimum_bounding_radius"]
    labels = ["perimeter", "area", "compactness", "min bounding radius"]
    logs = [True, True, False, True]

    test_results_path = SlideStatsProvider.get_report_path() / "statistical_test_results"

    df = df[df["species"] == "mouse"]

    plot_mouse_lobe_comparison(df, report_path, attributes, labels, logs, test_results_path, mouse_lobe_dict)
