import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

from zia.log import get_logger
from zia.pipeline.common.project_config import get_project_config
from zia.statistics.expression_profile.expression_profile_gradient import generate_distance_df
from zia.statistics.expression_profile.gradient import plot_gradient
from zia.statistics.expression_profile.validation_for_paper import plot_overview_for_paper
from zia.statistics.expression_profile.validation_images import plot_validation_for_all
from zia.statistics.lobulus_geometry.correlation import plot_correlation
from zia.statistics.lobulus_geometry.descriptive_stats import generate_descriptive_stats
from zia.statistics.lobulus_geometry.mouse_lobe_comparison import plot_mouse_lobe_comparison
from zia.statistics.lobulus_geometry.species_comparison import plot_species_comparison
from zia.statistics.lobulus_geometry.subject_comparison import plot_subject_comparison
from zia.statistics.lobulus_geometry.testing import run_all_tests
from zia.statistics.utils.data_provider import SlideStatsProvider


def map_dict(subject, roi, species, mouse_lobe_dict):
    if species != "mouse":
        return roi
    return mouse_lobe_dict[f"{subject}_{roi}"]


def save_slide_statistics(mouse_lobule_dict: Dict[str, str],
                          slide_stats: pd.DataFrame,
                          report_path: Path):
    to_save = slide_stats.copy()

    to_save['roi'] = to_save[['subject', 'roi', 'species']].apply(lambda x: map_dict(x[0], x[1], x[2], mouse_lobule_dict), axis=1)
    to_save.to_excel(report_path / "slide-stats.xlsx")



if __name__ == "__main__":

    exclusion_dict = {
        "UKJ-19-010_Human": ["CYP2D6", "GS"]
    }

    log = get_logger(__file__)

    project_config = get_project_config("control")
    slide_stats_provider = SlideStatsProvider(project_config, exclusion_dict)

    df = slide_stats_provider.get_slide_stats_df()

    report_path_base = SlideStatsProvider.create_report_path("manuscript")
    report_path_stats_test = report_path_base / "statistical-test"
    report_path_paper_plots = report_path_base / "paper-plots"
    report_path_descriptive_stats = report_path_base / "descriptive-stats"
    report_path_distance_df = report_path_base / "distance-data"
    report_path_valdiation_all = report_path_base / "overview-for-all"
    report_path_slide_statistics = report_path_base / "slide-statistics"

    for p in [report_path_stats_test, report_path_paper_plots, report_path_descriptive_stats, report_path_distance_df, report_path_valdiation_all,
              report_path_slide_statistics]:
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

    # write slide stats df
    log.info("Writing slide statistics dataframe")
    save_slide_statistics(mouse_lobe_dict, df, report_path_slide_statistics)

    # run the statistical tests
    log.info("Running statistical test")
    run_all_tests(df, report_path_stats_test, attributes, logs, mouse_lobe_dict)

    # boxplots
    log.info("Creating box plots")
    plot_species_comparison(df, report_path_paper_plots, attributes, labels, logs, report_path_stats_test)
    plot_subject_comparison(df, report_path_paper_plots, attributes, labels, logs, report_path_stats_test)
    plot_mouse_lobe_comparison(df, report_path_paper_plots, attributes, labels, logs, report_path_stats_test, mouse_lobe_dict)
    plot_correlation(df, report_path_paper_plots, attributes[0:-1], labels[0:-1], logs[0:-1], binned=True)
    plot_correlation(df, report_path_paper_plots, attributes[0:-1], labels[0:-1], logs[0:-1], binned=False)
    # descriptive stats
    log.info("Generating descriptive statistics data frame")
    generate_descriptive_stats(df, report_path_descriptive_stats, attributes, mouse_lobe_dict)
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

    # copy README file
    log.info("Copying README file")
    shutil.copy("README.md", report_path_base / "README.md")

    log.info("Creating zip archive")
    shutil.make_archive(str(slide_stats_provider.config.reports_path / "manuscript"), 'zip', str(report_path_base))
