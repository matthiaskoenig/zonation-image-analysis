import cv2
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import pandas as pd
import zarr
from imagecodecs.numcodecs import Jpeg2k

from zia import BASE_PATH
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.config import read_config
from zia.statistics.utils.data_provider import SlideStatsProvider

numcodecs.register_codec(Jpeg2k)


def plot_distances(subject_df: pd.DataFrame, template: np.ndarray) -> np.ndarray:
    arrays = []
    for protein, protein_df in subject_df.groupby("protein"):
        template[template == 0] = np.nan

        template[protein_df["height"].values, protein_df["width"].values] = protein_df["pv_dist"]

        arrays.append(template)

    idx = np.argmax([arr[arr != np.nan].size for arr in arrays])

    template = arrays[idx]
    return template


if __name__ == "__main__":

    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "distance_coloured"
    report_path.mkdir(exist_ok=True, parents=True)
    protein_order = ["CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4", "GS", "HE"]
    species_order = SlideStatsProvider.species_order
    colors = SlideStatsProvider.species_colors
    df = pd.read_csv(config.reports_path / "lobule_distances.csv", sep=",", index_col=False)
    species_gb = df.groupby("species")

    slide_stats_dict = SlideStatsProvider.get_slide_stats()
    for (subject, roi), subject_df in df.groupby(["subject", "roi"]):
        arrays = []
        mins = []

        slide_stats = slide_stats_dict[str(subject)][str(roi)]
        zarr_store = config.image_data_path / "stain_separated" / f"{subject}.zarr"
        roi_protein = np.array(zarr.open(store=zarr_store, path=f"stain_1/{roi}/CYP2E1/7"))

        template = np.zeros_like(roi_protein, dtype=float)
        template = plot_distances(subject_df, template)

        to_plot = (template * 255).astype(np.uint8)

        result = cv2.applyColorMap(to_plot, cv2.COLORMAP_MAGMA)

        cv2.imwrite(str(report_path / f"distance_{subject}_{roi}.png"), result)

        # fig = plot_pic(template, slide_stats, title=f"Subject: {subject}, ROI: {roi}")
        # fig.savefig(report_path / f"distance_{subject}_{roi}.png", bbox_inches='tight', pad_inches=0)
        # fig: plt.Figure
        # plt.show()
        # plt.close(fig)
