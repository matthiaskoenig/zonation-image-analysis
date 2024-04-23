import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.pipeline_components.portality_mapping_component import PortalityMappingComponent
from zia.statistics.lobulus_geometry.testing import run_all_tests
from zia.statistics.steatosis.get_data import get_steatosis_stats, get_density_stats, get_droplet_df
from zia.statistics.steatosis.species_comparison import plot_species_comparison, plot_species_comparison_density

project_config = get_project_config("steatosis")

report_path_base = project_config.reports_path / "plots" / "manuscript"
report_path_stats_steatosis_test = report_path_base / "steatosis-statistical-test"
report_path_paper_plots = report_path_base / "paper-plots"
report_path_steatosis_boxplots = report_path_base / "steatosis-boxplots"

report_path_descriptive_stats = report_path_base / "descriptive-stats"

for p in [report_path_base, report_path_stats_steatosis_test, report_path_paper_plots, report_path_steatosis_boxplots, report_path_descriptive_stats]:
    p.mkdir(exist_ok=True, parents=True)

attributes = ["perimeter", "area", "min_enclosing_circle"]
labels = ["perimeter", "area", "min bounding radius"]
units = ["µm", "µm$^2$", "µm"]
logs = [True, True, True]

df = get_steatosis_stats()
df = df[df["min_enclosing_circle"] < 25]

wsi_df = get_density_stats()

portality_droplet_df = get_droplet_df(df, overwrite=True)

#portality_droplet_df.to_csv(project_config.image_data_path / PortalityMappingComponent.dir_name / "lobule_distances.csv", index=False)
print(wsi_df)
run_all_tests(df, report_path_stats_steatosis_test, attributes, logs, False)

plot_species_comparison(slide_stats_df=df,
                        report_path=report_path_steatosis_boxplots,
                        attributes=attributes,
                        labels=labels,
                        logs=logs,
                        units=units)

plot_species_comparison_density(slide_stats_df=df,
                                wsi_df=wsi_df,
                                report_path=report_path_steatosis_boxplots)
