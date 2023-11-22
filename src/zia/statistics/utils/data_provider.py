import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import pandas as pd

from zia import BASE_PATH
from zia.config import read_config
from zia.processing.lobulus_statistics import SlideStats


def capitalize(s: str) -> str:
    return s[0].upper() + s[1:]


def get_species_from_name(subject) -> Optional[str]:
    """Metadata for image"""
    rat_pattern = re.compile("NOR-\d+")
    pig_pattern = re.compile("SSES2021 \d+")
    mouse_pattern = re.compile("MNT-\d+")
    human_pattern = re.compile("UKJ-19-\d+_Human")
    if re.search(pig_pattern, subject):
        return "pig"
    if re.search(mouse_pattern, subject):
        return "mouse"
    if re.search(rat_pattern, subject):
        return "rat"
    if re.search(human_pattern, subject):
        return "human"

    return None


def _merge_to_one_df(slide_stats: Dict[str, Dict[str, SlideStats]]) -> pd.DataFrame:
    dfs = []
    for subject, rois in slide_stats.items():
        species = get_species_from_name(subject)

        for roi, slide_stat in rois.items():
            slide_stat_df = slide_stat.to_dataframe()
            slide_stat_df["species"] = species
            slide_stat_df["subject"] = subject
            slide_stat_df["roi"] = roi

            dfs.append(slide_stat_df)

    return pd.concat(dfs, ignore_index=True)


class SlideStatsProvider:
    config = read_config(BASE_PATH / "configuration.ini")
    species_order = ["mouse", "rat", "pig", "human"]
    protein_order = ["HE", "GS", "CYP1A2", "CYP2D6", "CYP2E1", "CYP3A4"]
    species_colors = ["#77AADD", "#EE8866", "#DDDDDD", "#44BB99"]

    exclusion_dict = {
        "UKJ-19-010_Human": ["CYP2D6", "GS"]
    }

    @classmethod
    def get_species_colors_as_rgb(cls) -> List[Tuple[float]]:
        return [(tuple(int(h.strip("#")[i:i + 2], 16)/255 for i in (0, 2, 4))) for h in cls.species_colors]

    @classmethod
    def get_report_path(cls) -> Path:
        return cls.config.reports_path

    @classmethod
    def create_report_path(cls, dir_name: str) -> Path:
        report_path = SlideStatsProvider.config.reports_path / dir_name
        report_path.mkdir(parents=True, exist_ok=True)
        return report_path

    @classmethod
    def get_slide_stats_df(cls):
        return _merge_to_one_df(cls.get_slide_stats())

    @classmethod
    def get_slide_stats(cls) -> Dict[str, Dict[str, SlideStats]]:
        data_dir_stain_separated = cls.config.image_data_path / "slide_statistics"
        subject_dirs = sorted([f for f in data_dir_stain_separated.iterdir() if f.is_dir() and not f.name.startswith(".")])

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
