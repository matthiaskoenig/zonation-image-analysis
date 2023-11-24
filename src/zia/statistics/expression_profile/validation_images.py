from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pandas as pd
import zarr
from imagecodecs.numcodecs import Jpeg2k
from matplotlib import cm
from matplotlib.colors import to_rgba, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from zia import BASE_PATH
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.io.wsi_tifffile import read_ndpi
from zia.processing.lobulus_statistics import SlideStats
from zia.statistics.expression_profile.expression_profile_gradient import open_protein_arrays
from zia.statistics.utils.data_provider import SlideStatsProvider

numcodecs.register_codec(Jpeg2k)


def plot_distances(ax: plt.Axes,
                   subject_df: pd.DataFrame,
                   template: np.ndarray,
                   slide_stats: SlideStats) -> np.ndarray:
    arrays = []
    for protein, protein_df in subject_df.groupby("protein"):
        template[template == 0] = np.nan

        template[protein_df["height"].values, protein_df["width"].values] = protein_df["pv_dist"]

        arrays.append(template)

    idx = np.argmax([arr[arr != np.nan].size for arr in arrays])

    template = arrays[idx]
    ax.imshow(template, cmap="magma")
    slide_stats.plot_on_axis(ax,
                             lobulus_ec="white",
                             cvessel_ec="none",
                             cvessel_fc="none",
                             pvessel_fc="none",
                             pvessel_ec="none")


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = 255 - arr
    min_, max_ = np.min(arr), np.max(arr)

    return (((arr - min_) / (max_ - min_)) * 255).astype(np.uint8)


def plot_mixed_channel(ax: plt.Axes, protein_arrays: Dict[str, np.ndarray]):
    keys = ["CYP2E1", "GS", "CYP3A4"]
    arrays = [arr for key, arr in protein_arrays.items() if key in keys]
    arrays = [normalize(arr) for arr in arrays]

    merged = cv2.merge(arrays)

    # b, g, r = cv2.split(merged)

    # alpha = (~((b == 0) & (g == 0) & (r == 0))).astype(np.uint8) * 255

    # print(alpha)

    # merged = cv2.merge([b, g, r, alpha])

    ax.imshow(merged)


def plot_boundaries(ax: plt.axes, roi_protein: np.ndarray, slide_stats: SlideStats):
    ax: plt.Axes
    ax.imshow(255 - roi_protein, cmap="binary_r")
    slide_stats.plot_on_axis(ax,
                             lobulus_ec="greenyellow",
                             cvessel_ec="aqua",
                             cvessel_fc="aqua",
                             pvessel_ec="fuchsia",
                             pvessel_fc="fuchsia",
                             linewidth=0.5
                             )
    pass


def get_level_seven_array(slide_arrays: List[zarr.Array]) -> np.ndarray:
    level_0 = slide_arrays[0]
    last_level = slide_arrays[-1]

    level_diff = round(np.log2(level_0.shape[0]) - np.log2(last_level.shape[0]))

    current_array = np.array(last_level, dtype=np.uint8)
    for i in range(7 - level_diff):
        current_array = cv2.pyrDown(current_array)

    graysacle = cv2.cvtColor(current_array, cv2.COLOR_RGB2GRAY)
    blurred_gs = cv2.GaussianBlur(graysacle, (3, 3), 0)
    ret2, th2 = cv2.threshold(blurred_gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    output = np.ones_like(current_array, dtype=np.uint8) * 255

    return cv2.bitwise_and(current_array, current_array, output, mask=th2)


def plot_he(ax: plt.Axes, he_array: np.ndarray):
    ax.imshow(he_array)


def plot_validation_for_all(report_path: Path, distance_df: pd.DataFrame):
    config = read_config(BASE_PATH / "configuration.ini")

    slide_stats_dict = SlideStatsProvider.get_slide_stats()
    for (subject, roi), subject_df in distance_df.groupby(["subject", "roi"]):
        fig, axes = plt.subplots(2, 4, figsize=(8.3, 8.3 * 1.1 / 4), dpi=300, height_ratios=[0.96, 0.04])
        plt.subplots_adjust(hspace=0, wspace=0)
        slide_stats = slide_stats_dict[str(subject)][str(roi)]
        protein_arrays = open_protein_arrays(
            address=config.image_data_path / "stain_separated" / f"{subject}.zarr",
            path=f"{ZarrGroups.STAIN_1.value}/{roi}",
            level=slide_stats.meta_data["level"],
            excluded=[]
        )

        slide_path = None
        slide_dir = config.image_data_path / "rois_registered" / f"{subject}" / f"{roi}"
        for file in slide_dir.iterdir():
            if file.is_file() and "HE" in file.stem:
                slide_path = file
        if slide_path is not None:
            slide = read_ndpi(slide_path)
            he_array = get_level_seven_array(slide)
            plot_he(axes[0, 0], he_array)

        template = np.zeros_like(protein_arrays["CYP2E1"], dtype=float)
        plot_distances(axes[0, 3], subject_df, template, slide_stats)
        plot_boundaries(axes[0, 2], protein_arrays["CYP2E1"], slide_stats)
        plot_mixed_channel(axes[0, 1], protein_arrays)

        for ax in axes[0, :]:
            ax.axis("off")

        for ax in axes[1, :]:
            ax.axis("off")

        h, w = protein_arrays["HE"].shape
        pixel_width = 0.22724690376093626  # µm level 0
        p_factor = 2 ** 7 * pixel_width
        rular_width = 1000 / p_factor / h

        axes[1, 0].plot([0, rular_width], [0.95, 0.95], color="black",
                        marker="none", linewidth=5)
        axes[1, 0].text(x=rular_width / 2, y=0.6, s="1 mm", ha="center", va="top", fontsize=8)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)

        handles = [
            Line2D([], [], color="greenyellow", marker="none", label="lobule\nboundary"),
            Patch(facecolor=to_rgba("fuchsia", 0.5), edgecolor=to_rgb("fuchsia"), label="portal\nvessel"),
            Patch(facecolor=to_rgba("aqua", 0.5), edgecolor=to_rgb("aqua"), label="central\nvessel")
        ]

        cax = axes[1, -1].inset_axes((0.70, 0, 0.25, 1))
        fig.colorbar(cm.ScalarMappable(cmap="magma"), cax=cax, orientation="horizontal")
        cax.set_xticks([0, 1], ["PP", "PV"])

        axes[1, 2].legend(handles=handles,
                          frameon=False,
                          prop=dict(size=6),
                          ncols=3,
                          loc="center")
        fig.suptitle(f"Subject: {subject}, ROI: {roi}",
                     y=1.01)

        plt.savefig(report_path / f"distance_{subject}_{roi}.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    config = SlideStatsProvider.config
    report_path = config.reports_path / "supplementary_images"
    report_path.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)

    plot_validation_for_all(report_path, df)