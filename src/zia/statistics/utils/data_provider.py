import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import pandas as pd

from zia import BASE_PATH
from zia.config import read_config
from zia.pipeline.pipeline_components.algorithm.segementation.lobulus_statistics import SlideStats


def capitalize(s: str) -> str:
    return s[0].upper() + s[1:]


def merge_to_one_df(slide_stats: Dict[str, Dict[str, SlideStats]]) -> pd.DataFrame:
    dfs = []
    for subject, rois in slide_stats.items():

        for roi, slide_stat in rois.items():
            slide_stat_df = slide_stat.to_dataframe()
            slide_stat_df["species"] = slide_stat.meta_data.get("species")
            slide_stat_df["subject"] = subject
            slide_stat_df["roi"] = roi

            dfs.append(slide_stat_df)

    return pd.concat(dfs, ignore_index=True)


def get_slide_stats(slide_stats_dir: Path) -> Dict[str, Dict[str, SlideStats]]:

    if not slide_stats_dir.exists():
        raise FileNotFoundError("The slide statistic directory does not exist.")

    subject_dirs = sorted([f for f in slide_stats_dir.iterdir() if f.is_dir() and not f.name.startswith(".")])

    slide_stats = {}

    for subject_dir in subject_dirs:
        subject = subject_dir.stem
        roi_dict = {}

        roi_dirs = sorted([f for f in subject_dir.iterdir() if f.is_dir()])
        for roi_dir in roi_dirs:
            roi = roi_dir.stem
            roi_dict[roi] = SlideStats.load_from_file_system(roi_dir)

        slide_stats[subject] = roi_dict

    return slide_stats


class SlideStatsProvider:
    species_order = ["mouse", "rat", "pig", "human"]
    protein_order = ["HE", "GS", "CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4"]
    species_colors = ["#77AADD", "#EE8866", "#DDDDDD", "#44BB99"]

    def __init__(self, slide_stat_directory: Path):
        self.slide_stats_dict = get_slide_stats(slide_stat_directory)

    @classmethod
    def get_species_colors_as_rgb(cls) -> List[Tuple[float]]:
        return [(tuple(int(h.strip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))) for h in cls.species_colors]

    def get_slide_stats_df(self):
        return merge_to_one_df(self.slide_stats_dict)
