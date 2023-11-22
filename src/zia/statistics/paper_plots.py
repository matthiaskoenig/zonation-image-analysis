from zia.log import get_logger
from zia.statistics.lobulus_geometry.correlation_matrix import plot_correlation_matrix
from zia.statistics.lobulus_geometry.descriptive_stats import generate_descriptive_stats
from zia.statistics.lobulus_geometry.mouse_lobe_comparison import plot_mouse_lobe_comparison
from zia.statistics.lobulus_geometry.species_comparison import plot_species_comparison
from zia.statistics.lobulus_geometry.subject_comparison import plot_subject_comparison
from zia.statistics.lobulus_geometry.testing import run_all_tests
from zia.statistics.utils.data_provider import SlideStatsProvider

if __name__ == "__main__":

    log = get_logger(__file__)
    df = SlideStatsProvider.get_slide_stats_df()

    report_path_base = SlideStatsProvider.create_report_path("manuscript")

    report_path_stats_test = report_path_base / "statistical-test"
    report_path_paper_plots = report_path_base / "paper-plots"
    report_path_descriptive_stats = report_path_base / "descriptive-stats"
    report_path_distance_df = report_path_base / "distance-data"
    report_path_valdiation_all = report_path_base / "overview-for-all"

    for p in [report_path_stats_test, report_path_paper_plots, report_path_descriptive_stats, report_path_distance_df, report_path_valdiation_all]:
        p.mkdir(exist_ok=True, parents=True)


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
    attributes = ["perimeter", "area", "compactness", "minimum_bounding_radius"]
    labels = ["perimeter", "area", "compactness", "min bounding radius"]
    logs = [True, True, False, True]

    # run the statistical tests
    log.info("Running statistical test")
    run_all_tests(df, report_path_stats_test, attributes, logs)

    # boxplots
    log.info("Creating box plots")
    plot_species_comparison(df, report_path_paper_plots, attributes, labels, logs, report_path_stats_test)
    plot_subject_comparison(df, report_path_paper_plots, attributes, labels, logs, report_path_stats_test)
    plot_mouse_lobe_comparison(df, report_path_paper_plots, attributes, labels, logs, report_path_stats_test, mouse_lobe_dict)
    plot_correlation_matrix(df, report_path_paper_plots, attributes[0:-1], labels[0:-1], logs[0:-1], binned=True)
    plot_correlation_matrix(df, report_path_paper_plots, attributes[0:-1], labels[0:-1], logs[0:-1], binned=False)

    # descriptive stats
    log.info("Generating descriptive statistics data frame")
    generate_descriptive_stats(df, report_path_descriptive_stats, attributes, logs)

    exit(0)

    # generate the distance dataframe
    log.info("Generating distance intensity data frame")
    distance_df = generate_distance_df(report_path_distance_df, overwrite=False)

    # plot distance related plots
    log.info("Creating gradient plots")
    plot_gradient(report_path_paper_plots, distance_df)
    plot_overview_for_paper(report_path_paper_plots, distance_df)

    # validation plot for every subject and roi
    log.info("Creating overview plots for each subject and ROI")
    plot_validation_for_all(report_path_valdiation_all, distance_df)


