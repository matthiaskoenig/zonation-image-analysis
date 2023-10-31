import matplotlib.pyplot as plt
import pandas as pd

from zia.statistics.lobulus_geometry.boxplots_species_comparison import box_plot_species_comparison, box_plot_subject_comparison
from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize

if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("boxplots")
    df.to_csv(report_path / "slide_statistics_df.csv", index=False)

    attributes = ["perimeter", "area", "compactness", "minimum_bounding_radius"]
    labels = ["perimeter", "area", "compactness", "min bounding radius"]
    logs = [True, True, False, False]

    test_results_path = SlideStatsProvider.get_report_path() / "statistical_test_results"

    fig, axes = plt.subplots(len(SlideStatsProvider.species_order), len(attributes), dpi=300,
                             figsize=(len(attributes) * 2.5, len(SlideStatsProvider.species_order) * 2.5),
                             layout="constrained")

    df["minimum_bounding_radius"] = df["minimum_bounding_radius"] / 1000
    df["minimum_bounding_radius_unit"] = "mm"

    species_gb = df.groupby("species")

    for i, species in enumerate(SlideStatsProvider.species_order):
        species_df = species_gb.get_group(species)
        test_results = pd.read_csv(test_results_path / f"tukey_{species}_subjects.csv", index_col=False)

        for attr, ax, log, y_label in zip(attributes, axes[:, i], logs, labels):
            test_results_attr = test_results[test_results["attr"] == attr]
            box_plot_subject_comparison(species_df,
                                        species=species,
                                        attribute=attr,
                                        y_label=y_label,
                                        color=SlideStatsProvider.get_species_colors_as_rgb()[i],
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

    for species, ax in zip(SlideStatsProvider.species_order, axes[0, :].flatten()):
        ax.set_title(capitalize(species), fontweight="bold")

    for species, ax in zip(SlideStatsProvider.species_order, axes[-1, :].flatten()):
        species_df = df[df["species"] == species]

        counts = []
        for subject, subject_df in species_df.groupby("subject"):
            counts.append(len(subject_df))

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

    plt.savefig(report_path / "subject_comparison.png")
    plt.show()
