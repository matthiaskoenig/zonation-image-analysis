import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.pipeline_components.algorithm.segementation.lobulus_statistics import SlideStats
from zia.pipeline.pipeline_components.portality_mapping_component import PortalityMappingComponent
from zia.pipeline.pipeline_components.segementation_component import SegmentationComponent
from zia.statistics.lobulus_geometry.testing import run_all_tests
from zia.statistics.steatosis.get_data import get_steatosis_stats, get_density_stats, get_droplet_df
from zia.statistics.steatosis.plot_portality_density import plot_droplet_portality
from zia.statistics.steatosis.species_comparison import plot_species_comparison, plot_species_comparison_density
from zia.statistics.steatosis.utils import SPECIES_ORDER, SPECIES_COLORS

project_config = get_project_config("steatosis")

report_path_base = project_config.reports_path / "plots" / "manuscript"
report_path_stats_steatosis_test = report_path_base / "steatosis-statistical-test"
report_path_paper_plots = report_path_base / "paper-plots"
report_path_steatosis_boxplots = report_path_base / "steatosis-boxplots"
report_path_steatosis_portality = report_path_base / "steatosis-portality"

report_path_descriptive_stats = report_path_base / "descriptive-stats"

for p in [report_path_base, report_path_stats_steatosis_test, report_path_paper_plots, report_path_steatosis_boxplots, report_path_descriptive_stats,
          report_path_steatosis_portality]:
    p.mkdir(exist_ok=True, parents=True)

attributes = ["perimeter", "area", "min_enclosing_circle"]
labels = ["perimeter", "area", "min bounding radius"]
units = ["µm", "µm$^2$", "µm"]
logs = [True, True, True]

df = get_steatosis_stats()
df = df[df["min_enclosing_circle"] < 25]

wsi_df = get_density_stats()

portality_droplet_df = get_droplet_df(df, overwrite=False)
"""
plt.hist(portality_droplet_df["pv_dist"])
plt.show()
plt.hist(portality_droplet_df["mean_droplet_area"])
plt.show()
plt.hist(portality_droplet_df["droplet_count"])
plt.show()
"""

# portality_droplet_df.to_csv(project_config.image_data_path / PortalityMappingComponent.dir_name / "lobule_distances.csv", index=False)
# run_all_tests(df, report_path_stats_steatosis_test, attributes, logs, False)

"""for (roi, subject), group_df in portality_droplet_df.groupby(["roi", "subject"]):
    print(roi, subject)
    group_df = group_df.drop_duplicates(subset=["width", "height"])

    min_w, min_h = np.min(group_df["width"]), np.min(group_df["height"])

    # print(np.unique(group_df[['height', 'width']].values, return_counts=True))
    heat_map_data = group_df.pivot(index="height", columns="width", values="total_droplet_area").to_numpy()

    fig, ax = plt.subplots()

    ax.imshow(heat_map_data, cmap=plt.get_cmap("hot"), interpolation="nearest")
    #plt.colorbar()

    slidestats = SlideStats.load_from_file_system(project_config.image_data_path / SegmentationComponent.dir_name / subject / str(roi))

    slidestats.plot_on_axis(ax, offset=(min_h, min_w))
    ax.set_title(f"{subject}, {roi}")
    # Show plot
    plt.show()"""

# plot_species_comparison(slide_stats_df=df,
#                         report_path=report_path_steatosis_boxplots,
#                         attributes=attributes,
#                         labels=labels,
#                         logs=logs,
#                         units=units)
#
# plot_species_comparison_density(slide_stats_df=df,
#                                 wsi_df=wsi_df,
#                                 report_path=report_path_steatosis_boxplots)

group_order = []

for gr in SPECIES_ORDER:
    if gr in ["mouse", "rat"]:
        for w in ["2", "4"]:
            group_order.append(f"{gr} ({w}W HDF)")
    else:
        group_order.append(gr)

colors = []

for gr in SPECIES_ORDER:
    if gr in ["mouse", "rat"]:
        for w in ["2", "4"]:
            colors.append(SPECIES_COLORS[gr])
    else:
        colors.append(SPECIES_COLORS[gr])

plot_droplet_portality(report_path=report_path_steatosis_portality,
                       distance_df=portality_droplet_df,
                       group_order=group_order,
                       colors=colors)
