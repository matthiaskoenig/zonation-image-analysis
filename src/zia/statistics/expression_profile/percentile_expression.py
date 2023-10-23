from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import shapely
import shapely.ops
import zarr
from shapely import Geometry

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.mask_generatation.image_analysis import MaskGenerator
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.processing.lobulus_statistics import SlideStats, LobuleStatistics
from zia.statistics.utils.data_provider import SlideStatsProvider, get_species_from_name


def open_protein_arrays(address: Path, path: str, level: PyramidalLevel) -> Dict[str, np.ndarray]:
    group = zarr.open(store=address, path=path)
    return {key: 255 - np.array(val.get(f"{level}")) for key, val in group.items()}


def swap_xy(geometry: Geometry):
    return shapely.ops.transform(lambda x, y: (y, x), geometry)


def create_lobule_df(no_pixels: int, min_i: int, max_i: int, percentage_max: list[float], data: list[float], idx: int) -> pd.DataFrame:
    return pd.DataFrame(
        dict(lobule=idx,
             n_pixels=no_pixels,
             min_intensity=min_i,
             max_intensity=max_i,
             percentage_max=percentage_max,
             data=data)
    ).explode(["data", "percentage_max"])


def analyse_protein_expression_for_lobule(protein_array: np.ndarray, lobule_stats: LobuleStatistics, idx: int) -> Optional[pd.DataFrame]:
    mask = np.zeros_like(protein_array, dtype=np.uint8)

    MaskGenerator.draw_polygons(mask, swap_xy(lobule_stats.polygon), (0, 0), True)
    for p in lobule_stats.vessels_central:
        MaskGenerator.draw_polygons(mask, swap_xy(p), (0, 0), False)
    for p in lobule_stats.vessels_portal:
        MaskGenerator.draw_polygons(mask, swap_xy(p), (0, 0), False)

    mask = mask.astype(bool)
    area = mask.sum()

    pixels = protein_array[mask]
    empty_pixels = pixels[pixels == 0]
    if empty_pixels.size / area > 0.2:
        print("empty lobule on slide")
        return None

    if pixels.size == 0:
        print(f"to small: {pixels.size}")
        return None

    p_min, p_max = np.min(pixels), np.max(pixels)

    if p_max == p_min:
        return None

    norm_pixels = (pixels - p_min) / (p_max - p_min)

    percentages = list(range(0, 101, 5))

    percentiles = [(norm_pixels > (p / 100)).sum() / area for p in percentages]

    return create_lobule_df(area, p_min, p_max, percentages, percentiles, idx)


def analyse_protein_expression(protein_array: np.ndarray, slide_stats: SlideStats) -> pd.DataFrame:
    dfs = [analyse_protein_expression_for_lobule(protein_array, ls, idx) for idx, ls in enumerate(slide_stats.lobule_stats)]
    dfs = list(filter(lambda x: x is not None, dfs))
    return pd.concat(dfs)


def analyse_lobuli(slide_stats: SlideStats, protein_arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    protein_dfs = []
    for protein, protein_array in protein_arrays.items():
        print(protein)
        df = analyse_protein_expression(protein_array, slide_stats)
        df["protein"] = protein
        protein_dfs.append(df)
    return pd.concat(protein_dfs)


if __name__ == "__main__":
    config = read_config(BASE_PATH / "configuration.ini")

    slide_stats_dict = SlideStatsProvider.get_slide_stats()

    subject_dfs = []
    for subject, roi_dict in slide_stats_dict.items():
        roi_dfs = []
        for roi, slide_stats in roi_dict.items():
            protein_arrays = open_protein_arrays(
                address=config.image_data_path / "stain_separated" / f"{subject}.zarr",
                path=f"{ZarrGroups.STAIN_1.value}/{roi}",
                level=slide_stats.meta_data["level"]
            )
            df = analyse_lobuli(slide_stats, protein_arrays)
            df["roi"] = roi
            roi_dfs.append(df)

        subject_df = pd.concat(roi_dfs)
        subject_df["subject"] = subject
        species = get_species_from_name(subject)
        subject_df["species"] = species
        subject_dfs.append(subject_df)

    final_df = pd.concat(subject_dfs)

    print(final_df.head())

    final_df.to_csv(config.reports_path / "lobule_percentiles.csv", sep=",", index=False)
