from pathlib import Path

import cv2
import pandas as pd

from zia import BASE_PATH
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.statistics.expression_profile.expression_profile_gradient import open_protein_arrays
from zia.statistics.utils.data_provider import SlideStatsProvider


def plot_dab_stains(report_path: Path, overwrite=True) -> pd.DataFrame:
    if overwrite == False and (report_path / "lobule_distances.csv").exists():
        return pd.read_csv(report_path / "lobule_distances.csv", index_col=False)

    config = read_config(BASE_PATH / "configuration.ini")
    report_path = config.reports_path / "dab-stains"
    report_path.mkdir(exist_ok=True)

    slide_stats_dict = SlideStatsProvider.get_slide_stats()

    for subject, roi_dict in slide_stats_dict.items():

        for roi, slide_stats in roi_dict.items():
            protein_arrays = open_protein_arrays(
                address=config.image_data_path / "stain_separated" / f"{subject}.zarr",
                path=f"{ZarrGroups.STAIN_1.value}/{roi}",
                level=slide_stats.meta_data["level"],
                excluded=[]
            )

            for key, arr in protein_arrays.items():
                cv2.imwrite(str(report_path / f"{subject}_{roi}_{key}.png"), arr)


if __name__ == "__main__":
    config = SlideStatsProvider.config

    report_path = config.reports_path

    plot_dab_stains(report_path=report_path)
