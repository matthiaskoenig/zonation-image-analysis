from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pandas as pd
from imagecodecs.numcodecs import Jpeg2k
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from zia import BASE_PATH
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.statistics.expression_profile.expression_profile_gradient import open_protein_arrays
from zia.statistics.utils.data_provider import SlideStatsProvider

numcodecs.register_codec(Jpeg2k)


def plot_distances(ax: plt.Axes,
                   subject_df: pd.DataFrame,
                   template: np.ndarray) -> np.ndarray:
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


def plot_boundaries(ax: plt.axes, roi_protein: np.ndarray):
    ax: plt.Axes
    ax.imshow(255 - roi_protein, cmap="binary_r")
    slide_stats.plot_on_axis(ax,
                             lobulus_ec="cornflowerblue",
                             cvessel_ec="yellowgreen",
                             cvessel_fc="yellowgreen",
                             pvessel_ec="palegreen",
                             pvessel_fc="palegreen"
                             )
    pass


if __name__ == "__main__":

    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "supplementary_images"
    report_path.mkdir(exist_ok=True, parents=True)
    protein_order = ["CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4", "GS", "HE"]
    species_order = SlideStatsProvider.species_order
    colors = SlideStatsProvider.species_colors
    df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)
    species_gb = df.groupby("species")

    slide_stats_dict = SlideStatsProvider.get_slide_stats()
    for (subject, roi), subject_df in df.groupby(["subject", "roi"]):
        fig, axes = plt.subplots(2, 3, figsize=(8.3, 8.3 * 1.1 / 3), dpi=300, layout="constrained", height_ratios=[0.96, 0.04])
        fig.suptitle(f"Subject: {subject}, ROI: {roi}", fontsize=18)

        slide_stats = slide_stats_dict[str(subject)][str(roi)]
        protein_arrays = open_protein_arrays(
            address=config.image_data_path / "stain_separated" / f"{subject}.zarr",
            path=f"{ZarrGroups.STAIN_1.value}/{roi}",
            level=slide_stats.meta_data["level"],
            excluded=[]
        )

        template = np.zeros_like(protein_arrays["CYP2E1"], dtype=float)
        plot_distances(axes[0, 2], subject_df, template)
        plot_boundaries(axes[0, 1], protein_arrays["CYP2E1"])
        plot_mixed_channel(axes[0, 0], protein_arrays)

        for ax in axes[0, :]:
            ax.axis("off")

        for ax in axes[1, :]:
            ax.axis("off")
            h, w = protein_arrays["HE"].shape
            pixel_width = 0.22724690376093626  # µm level 0
            p_factor = 2 ** 7 * pixel_width
            rular_width = 1000 / p_factor / h

            ax.plot([0, rular_width], [0.95, 0.95], color="black",
                    marker="none", linewidth=5)
            ax.text(x=rular_width / 2, y=0.6, s="1 mm", ha="center", va="top", fontsize=12)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        handles = [
            Line2D([], [], color="cornflowerblue", marker="none", label="lobule\nboundary"),
            Patch(color="palegreen", label="portal\nvessel"),
            Patch(color="yellowgreen", label="central\nvessel")
        ]

        cax = axes[1, -1].inset_axes((0.70, 0, 0.25, 1))
        fig.colorbar(cm.ScalarMappable(cmap="magma"), cax=cax, orientation="horizontal")
        cax.set_xticks([0, 1], ["PP", "PV"])

        axes[0, 1].legend(handles=handles,
                          frameon=False,
                          prop=dict(size=6),
                          ncols=3,
                          bbox_to_anchor=(0.5, 1), loc="lower center")

        plt.savefig(report_path / f"distance_{subject}_{roi}.png", pad_inches=0)
        # plt.show()
        plt.close(fig)
