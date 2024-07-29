from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.pipeline_components.segementation_component import SegmentationComponent
from zia.statistics.paper_plots_control import load_distance_df
from zia.statistics.steatosis.droplet_density_species_comparison import species_comparison_droplet_density
from zia.statistics.steatosis.droplet_species_comparison import droplet_species_comparison
from zia.statistics.steatosis.plot_lobulegeometry_steatosis_corr import plot_lobulegeo_steatosis_correlation
from zia.statistics.steatosis.plot_portality_density import plot_droplet_portality
from zia.statistics.steatosis.utils.get_data import get_steatosis_stats, get_density_stats, get_droplet_df
from zia.statistics.steatosis.lobuli_species_comparison import species_lobuli_comparison
from zia.statistics.steatosis.steatosis_gradient import plot_species_comparison_gradient
from zia.statistics.steatosis.utils.utils import SPECIES_ORDER, SPECIES_COLORS
from zia.statistics.utils.data_provider import SlideStatsProvider

project_config = get_project_config("steatosis")
project_config_control = get_project_config("control")
report_path_plots = project_config.reports_path / "plots"
stats_excel = project_config.reports_path / "descriptive-stats.xlsx"



for p in [report_path_plots]:
    p.mkdir(exist_ok=True, parents=True)

steatosis_attributes = ["perimeter", "area", "min_enclosing_circle"]
steatosis_labels = ["perimeter", "area", "min bounding radius"]
steatosis_units = ["µm", "µm$^2$", "µm"]
steatosis_logs = [True, True, True]

# attributes = ["perimeter", "area", "min_enclosing_circle"]
lobuli_attributes = ["perimeter", "area", "compactness", "minimum_bounding_radius"]
lobuli_labels = ["perimeter", "area", "compactness", "min bounding radius"]
lobuli_units = ["µm", "µm$^2$", "-", "µm"]
lobuli_logs = [True, True, False, True]

df = get_steatosis_stats()

df = df[df["min_enclosing_circle"] < 25]

wsi_df = get_density_stats()

steatosis_portality_df = load_distance_df(project_config)
control_portality_df = load_distance_df(project_config_control)

portality_droplet_df = get_droplet_df(df, steatosis_portality_df, overwrite=False)
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
slide_stat_provider_steatosis = SlideStatsProvider(project_config.image_data_path / SegmentationComponent.dir_name)
slide_stat_provider_control = SlideStatsProvider(project_config_control.image_data_path / SegmentationComponent.dir_name)

slide_stats_df = slide_stat_provider_steatosis.get_slide_stats_df()
slide_stats_df_control = slide_stat_provider_control.get_slide_stats_df()

# for (roi, subject), group_df in portality_droplet_df.groupby(["roi", "subject"]):
#     print(roi, subject)
#
#     group_df = group_df.drop_duplicates(subset=["width", "height"])
#
#     min_w, min_h = np.min(group_df["width"]), np.min(group_df["height"])
#
#     # print(np.unique(group_df[['height', 'width']].values, return_counts=True))
#     heat_map_data = group_df.pivot(index="height", columns="width", values="droplet_count").to_numpy()
#
#     fig, ax = plt.subplots()
#
#     im = ax.imshow(heat_map_data, cmap=plt.get_cmap("hot"), interpolation="nearest", vmin=0, vmax=5)
#     bar = plt.colorbar(im)
#
#     slidestats = SlideStats.load_from_file_system(project_config.image_data_path / SegmentationComponent.dir_name / subject / str(roi))
#
#
#
#     slidestats.plot_on_axis(ax, offset=(min_h, min_w))
#     ax.set_title(f"{subject}, {roi}")
#     # Show plot
#     plt.show()



droplet_species_comparison(droplet_stats_df=df,
                           report_path=report_path_plots,
                           stats_excel=stats_excel,
                           attributes=steatosis_attributes,
                           labels=steatosis_labels,
                           logs=steatosis_logs,
                           units=steatosis_units)

species_comparison_droplet_density(droplet_stats_df=df,
                                   wsi_df=wsi_df,
                                   report_path=report_path_plots,
                                   stats_excel=stats_excel)

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

plot_droplet_portality(report_path=report_path_plots,
                       distance_df=portality_droplet_df,
                       group_order=group_order,
                       colors=colors)

plot_lobulegeo_steatosis_correlation(
    portality_droplet_df=portality_droplet_df,
    reportpath=report_path_plots,
    slide_stats_df_steatosis=slide_stats_df,
    slide_stats_df_control=slide_stats_df_control,
    attributes=lobuli_attributes,
    logs=lobuli_logs,
    labels=lobuli_labels,
    units=lobuli_units

)

species_lobuli_comparison(report_path=report_path_plots,
                          stats_excel=stats_excel,
                          slide_stats_df_steatosis=slide_stats_df,
                          slide_stats_df_control=slide_stats_df_control,
                          attributes=lobuli_attributes,
                          logs=lobuli_logs,
                          labels=lobuli_labels,
                          units=lobuli_units)

plot_species_comparison_gradient(report_path=report_path_plots,
                                 control_portality_df=control_portality_df,
                                 steatosis_portality_df=steatosis_portality_df
                                 )
