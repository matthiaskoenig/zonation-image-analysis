import matplotlib.pyplot as plt
import pandas as pd

from zia.statistics.lobulus_geometry.boxplots_species_comparison import box_plot_species_comparison
from zia.statistics.utils.data_provider import SlideStatsProvider

if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("boxplots")
    df.to_csv(report_path / "slide_statistics_df.csv", index=False)

    attributes = ["perimeter", "area", "compactness", "minimum_bounding_radius"]
    labels = ["perimeter", "area", "compactness", "min bounding radius"]
    logs = [True, True, False, False]

    test_results = pd.read_csv(SlideStatsProvider.get_report_path() / "statistical_test_results" / "tukey_species.csv", index_col=False)

    fig, axes = plt.subplots(1, len(attributes), dpi=300,
                             figsize=(len(attributes) * 2.5, 2.5),
                             layout="constrained")

    df["minimum_bounding_radius"] = df["minimum_bounding_radius"] / 1000
    df["minimum_bounding_radius_unit"] = "mm"
    for attr, ax, log, y_label in zip(attributes, axes, logs, labels):
        test_results_attr = test_results[test_results["attr"] == attr]
        box_plot_species_comparison(df,
                                    attr,
                                    y_label=y_label,
                                    species_order=SlideStatsProvider.species_order,
                                    colors=SlideStatsProvider.get_species_colors_as_rgb(),
                                    log=log,
                                    ax=ax,
                                    test_results=test_results_attr,
                                    annotate_n=True)

    plt.savefig(report_path / "species_comparison.png")
    plt.show()
