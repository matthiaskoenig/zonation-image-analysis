import matplotlib.pyplot as plt
import pandas as pd

from zia.statistics.lobulus_geometry.plotting.boxplots import box_plot_subject_comparison, box_plot_roi_comparison
from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize

if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("boxplots")
    df.to_csv(report_path / "slide_statistics_df.csv", index=False)

    attributes = ["perimeter", "area", "compactness", "minimum_bounding_radius"]
    labels = ["perimeter", "area", "compactness", "min bounding radius"]
    logs = [True, True, False, True]

    test_results_path = SlideStatsProvider.get_report_path() / "statistical_test_results"

    df = df[df["species"] == "mouse"]

    fig, axes = plt.subplots(len(attributes), len(set(df["subject"].values)), dpi=300,
                             figsize=(len(attributes) * 2.5, len(SlideStatsProvider.species_order) * 2.5),
                             sharey="row",
                             layout="constrained")

    # df["minimum_bounding_radius"] = df["minimum_bounding_radius"] / 1000
    # df["minimum_bounding_radius_unit"] = "mm"

    annova_result = pd.read_csv(test_results_path / f"anova_mouse_rois.csv", index_col=False)
    test_results = pd.read_csv(test_results_path / f"tukey_mouse_rois.csv", index_col=False)

    for i, (subject, subject_df) in enumerate(df.groupby("subject")):
        for attr, ax, log, y_label in zip(attributes, axes[:, i], logs, labels):
            if annova_result[annova_result["attr"] == attr].iloc[0]["pvalue"] < 0.05:
                test_results_attr = test_results[test_results["attr"] == attr]
            else:
                test_results_attr = None

            box_plot_roi_comparison(subject_df,
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

    for (subject, subject_df), ax in zip(df.groupby("subject"), axes[0, :].flatten()):
        ax.set_title(capitalize(subject), fontweight="bold")

    for (subject, subject_df), ax in zip(df.groupby("subject"), axes[-1, :].flatten()):
        counts = []
        for roi, roi_df in subject_df.groupby("roi"):
            counts.append(len(roi_df))

        in_axes = ax.inset_axes((0, -0.07, 1, 0.07), transform=ax.transAxes)
        in_axes.set_xticks([(i + 1) / len(counts) - 1 / 2 * 1 / len(counts) for i in range(len(counts))],
                           ax.get_xticklabels(),
                           rotation=90)
        ax.set_xticks([])
        in_axes.set_yticks([])
        # in_axes.set_title(capitalize(species), fontweight="bold", )

        for i, count in enumerate(counts):
            in_axes.text((i + 1) / len(counts) - 1 / 2 * 1 / len(counts),
                         0,
                         s=f"{count}",
                         ha="center",
                         va="bottom",
                         fontsize=9)

            in_axes.fill_betweenx(y=[0, 1], x1=i / len(counts), x2=(i + 1) / len(counts), color="white" if i % 2 == 0 else "whitesmoke")

        in_axes.set_xlim(left=0, right=1)

    plt.savefig(report_path / "mouse_roi_comparison.png")
    plt.show()
