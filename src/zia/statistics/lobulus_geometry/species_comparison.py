import matplotlib.pyplot as plt
import pandas as pd

from zia.statistics.lobulus_geometry.plotting.boxplots import box_plot_species_comparison
from zia.statistics.utils.data_provider import SlideStatsProvider

if __name__ == "__main__":
    df = SlideStatsProvider.get_slide_stats_df()
    report_path = SlideStatsProvider.create_report_path("boxplots")
    df.to_csv(report_path / "slide_statistics_df.csv", index=False)

    attributes = ["perimeter", "area", "compactness", "minimum_bounding_radius"]
    labels = ["perimeter", "area", "compactness", "min bounding radius"]
    logs = [True, True, False, True]
    test_results_path = SlideStatsProvider.get_report_path() / "statistical_test_results"

    kruskal_result = pd.read_csv(test_results_path / f"kruskal_species.csv", index_col=False)
    test_results = pd.read_csv(test_results_path / "dunns_species.csv", index_col=False)

    fig, axes = plt.subplots(1, len(attributes), dpi=300,
                             figsize=(len(attributes) * 2.5, 2.5),
                             layout="constrained")

    for attr, ax, log, y_label in zip(attributes, axes, logs, labels):
        if kruskal_result[kruskal_result["attr"] == attr].iloc[0]["pvalue"] < 0.05:
            test_results_attr = test_results[test_results["attr"] == attr]
        else:
            test_results_attr = None

        box_plot_species_comparison(df,
                                    attr,
                                    y_label=y_label,
                                    species_order=SlideStatsProvider.species_order,
                                    colors=SlideStatsProvider.get_species_colors_as_rgb(),
                                    log=log,
                                    ax=ax,
                                    test_results=test_results_attr,
                                    annotate_n=True)

    plt.savefig(report_path / "species_comparison.png", dpi=600)
    plt.show()
