import cv2
import numpy as np
import pandas as pd

from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.pipeline.common.project_config import get_project_config
from zia.pipeline.pipeline_components.roi_extraction_component import RoiExtractionComponent
from zia.statistics.lobulus_geometry.species_comparison import plot_species_comparison
from image_utils.io.tiffile import read_ndpi

project_config = get_project_config("steatosis")

steatosis_stats_path = project_config.image_data_path / "SteatosisStats"

diet = {
    "FLR-167": "2",
    "FLR-168": "2",
    "FLR-169": "2",
    "FLR-170": "2",
    "FLR-171": "2",
    "FLR-172": "2",

    "FLR-179": "4",
    "FLR-180": "4",
    "FLR-181": "4",
    "FLR-199": "4",
    "FLR-200": "4",
    "FLR-201": "4",

    "MNT-031": "2",
    "MNT-032": "2",
    "MNT-027": "2",
    "MNT-033": "2",
    "MNT-034": "2",
    "MNT-035": "2",
    "MNT-036": "2",

    "MNT-041": "4",
    "MNT-042": "4",
    "MNT-043": "4",
    "MNT-044": "4",
    "MNT-045": "4",
    "MNT-046": "4",
}


def get_species_from_subject(subject: str) -> str:
    if "FLR" in subject:
        return "rat"
    if "MNT" in subject:
        return "mouse"
    if "Human" in subject:
        return "human"
    return "unknown"


def get_steatosis_stats() -> pd.DataFrame:
    dfs = []
    for subject_dir in steatosis_stats_path.iterdir():
        subject = subject_dir.stem
        for roi_dir in subject_dir.iterdir():
            roi = roi_dir.stem

            df = pd.read_csv(roi_dir / "steatosis_statistics.csv")

            df["subject"] = subject
            df["roi"] = roi

            species = get_species_from_subject(subject)
            df["species"] = species

            if species in ["mouse", "rat"]:
                df["diet"] = diet[subject]

            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_density_stats(px_size=0.2272) -> pd.DataFrame:
    data_points = []

    for subject_dir in (project_config.image_data_path / RoiExtractionComponent.dir_name).iterdir():
        for roi_dir in subject_dir.iterdir():

            for wsi_path in roi_dir.iterdir():
                if not "HE" in wsi_path.stem:
                    continue
                wsis = read_ndpi(wsi_path)

                lowest_key = max(wsis.keys())

                image = cv2.cvtColor(wsis[lowest_key][:], cv2.COLOR_RGB2GRAY)

                blur = cv2.GaussianBlur(image, (15, 15), 5)

                _, th = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

                area = np.count_nonzero(th == 0) * (px_size * 2 ** lowest_key) ** 2

                data_points.append(
                    dict(
                        subject=subject_dir.stem,
                        roi=roi_dir.stem,
                        species=get_species_from_subject(subject_dir.stem),
                        area=area,
                        diet=diet.get(subject_dir.stem)
                    )
                )

    return pd.DataFrame(data_points)
