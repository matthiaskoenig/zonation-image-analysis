from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from zia import BASE_PATH
from zia.config import read_config, Configuration
from zia.io.wsi_tifffile import read_ndpi
from zia.pipeline.pipeline_components.algorithm.segementation.lobulus_statistics import SlideStats
from zia.pipeline.pipeline_components.portality_mapping_component import open_protein_arrays
from zia.statistics.expression_profile.validation_images import plot_mixed_channel, plot_boundaries, plot_distances, get_level_seven_array, plot_he, \
    get_zarr_path
from zia.statistics.utils.data_provider import SlideStatsProvider, capitalize


def plot_overview_for_paper(report_path: Path,
                            project_config: Configuration,
                            slide_stats_dict: Dict[str, Dict[str, SlideStats]],
                            distance_df: pd.DataFrame):
    subjects = ["MNT-023", "NOR-021", "SSES2021 10", "UKJ-19-026_Human"]
    rois = [0, 0, 0, 0]
    config = read_config(BASE_PATH / "configuration.ini")

    distance_gb = distance_df.groupby(["subject", "roi"])

    slide_paths = []
    for subject, roi in zip(subjects, rois):
        slide_dir = config.image_data_path / "rois_registered" / f"{subject}" / f"{roi}"
        for file in slide_dir.iterdir():
            if file.is_file() and "HE" in file.stem:
                slide_paths.append(file)

    slide_arrays = [read_ndpi(slide_path)
                    for slide_path in slide_paths]

    he_arrays = [get_level_seven_array(slides) for slides in slide_arrays]

    protein_arrays = [open_protein_arrays(zarr_path=get_zarr_path(project_config, subject, roi),
                                          level=slide_stats_dict[subject][str(roi)].meta_data["level"],
                                          excluded=[])
                      for (subject, roi) in zip(subjects, rois)]

    shapes = [arr["HE"].shape for arr in protein_arrays]

    max_h = 0
    w = 0
    for shape in shapes:
        if shape[0] > max_h:
            max_h = shape[0]
            w = shape[1]

    fig, axes = plt.subplots(6, 4, figsize=(8.3, 8.3 * 1.1 * max_h / w), dpi=300, height_ratios=[0.05, 0.2325, 0.2325, 0.2325, 0.2325, 0.02])

    plt.subplots_adjust(hspace=0, wspace=0)

    for i, (subject, roi, protein_array, he_array) in enumerate(zip(subjects, rois, protein_arrays, he_arrays)):
        subject_df = distance_gb.get_group((subject, roi))
        template = np.zeros_like(protein_array["CYP2E1"], dtype=float)
        slide_stats = slide_stats_dict[subject][str(roi)]
        plot_he(axes[1, i], he_array)
        plot_mixed_channel(axes[2, i], protein_array)
        plot_boundaries(axes[3, i], protein_array["CYP2E1"], slide_stats)
        plot_distances(axes[4, i], subject_df, template, slide_stats)

    for species, color, ax in zip(SlideStatsProvider.species_order, SlideStatsProvider.species_colors, axes[0, :]):
        ax.set_facecolor(color)
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(x=0.5, y=0.5, s=capitalize(species), ha="center", va="center", transform=ax.transAxes, fontweight="bold")
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

    for ax, protein_array in zip(axes[-1, :].flatten(), protein_arrays):
        # ax.axis("off")
        h, w = protein_array["HE"].shape
        pixel_width = 0.22724690376093626  # µm level 0
        p_factor = 2 ** 7 * pixel_width
        rular_width = 1000 / p_factor / h

        ax.plot([0, rular_width], [0.95, 0.95], color="black",
                marker="none", linewidth=2)
        ax.text(x=rular_width / 2, y=0.7, s="1 mm", ha="center", va="top", fontsize=8)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    cax = axes[4, -1].inset_axes((-2.95, 0.9, 0.35, 0.1))
    fig.colorbar(cm.ScalarMappable(cmap="magma"), cax=cax, orientation="horizontal")
    cax.set_xticks([0, 1], ["PP", "PV"], fontsize=8)

    for ax in axes[1:, :].flatten():
        ax.axis("off")

    handles = [
        Line2D([], [], color="greenyellow", marker="none", label="lobule boundary"),
        Patch(facecolor=to_rgba("fuchsia", 0.5), edgecolor=to_rgb("fuchsia"), label="portal vessel"),
        Patch(facecolor=to_rgba("aqua", 0.5), edgecolor=to_rgb("aqua"), label="central vessel")
    ]

    axes[3, -1].legend(handles=handles, frameon=False, bbox_to_anchor=(-3, 1.15), loc="upper left", prop=dict(size=8))

    for let, ax in zip(("A", "B", "C", "D"), axes[1:5, -1]):
        ax.text(x=-3, y=1, s=f"{let}", fontsize=11, fontweight="bold", transform=ax.transAxes,
                va="top", ha="right")

    plt.savefig(report_path / "representative_subjects.png", bbox_inches="tight", dpi=600)
    plt.savefig(report_path / "representative_subjects.svg", bbox_inches="tight", dpi=600)

    plt.show()


if __name__ == "__main__":
    config = SlideStatsProvider.config
    report_path = config.reports_path / "supplementary_images"
    distance_df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)
    plot_overview_for_paper(report_path, distance_df)
