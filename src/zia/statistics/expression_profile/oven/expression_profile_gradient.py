from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import shapely
import shapely.ops
import zarr
from shapely import Geometry, Polygon

from zia import BASE_PATH
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.pipelines.mask_generatation.image_analysis import MaskGenerator
from zia.config import read_config
from zia.data_store import ZarrGroups
from zia.processing.lobulus_statistics import SlideStats, LobuleStatistics
from zia.statistics.utils.data_provider import SlideStatsProvider, get_species_from_name


@np.vectorize
def dist(d_p: float, d_c: float) -> float:
    if d_c == 0 and d_p == 0:
        return np.nan
    return 1 - d_c / (d_p + d_c)


def open_protein_arrays(address: Path, path: str, level: PyramidalLevel) -> Dict[str, np.ndarray]:
    group = zarr.open(store=address, path=path)
    return {key: 255 - np.array(val.get(f"{level}")) for key, val in group.items()}


def swap_xy(geometry: Geometry):
    return shapely.ops.transform(lambda x, y: (y, x), geometry)


def offset(geometry: Geometry, offset: Tuple[int, int]):
    return shapely.ops.transform(lambda x, y: (x - offset[0], y - offset[1]), geometry)


def create_lobule_df(height: np.ndarray, width: np.ndarray, d_portal: np.ndarray, d_central: np.ndarray, pv_dist: np.ndarray, intensity: np.ndarray,
                     idx: int) -> pd.DataFrame:
    return pd.DataFrame(
        dict(lobule=idx,
             width=width,
             height=height,
             d_portal=d_portal,
             d_central=d_central,
             pv_dist=pv_dist,
             intensity=intensity
             )
    ).explode(["width", "height", "d_portal", "d_central", "intensity"])


def analyse_protein_expression_for_lobule(protein_array: np.ndarray, lobule_stats: LobuleStatistics, idx: int, meta: Dict) -> Optional[pd.DataFrame]:
    poly_boundary: Polygon = swap_xy(lobule_stats.polygon)
    minx, miny, maxx, maxy = (max(0, int(x)) for x in poly_boundary.bounds)
    # print(minx, maxx, miny, maxy)

    # print(protein_array.shape)
    lobule_array = protein_array[miny:maxy, minx:maxx]

    mask_central = np.ones_like(lobule_array, dtype=np.uint8)
    vessel_central = [swap_xy(v) for v in lobule_stats.vessels_central]
    max_i = np.max(lobule_array)

    for p in vessel_central:
        MaskGenerator.draw_polygons(mask_central, p, (minx, miny), False)

    mask_central_with_max = mask_central.copy()
    mask_central_with_max[lobule_array == max_i] = False

    # plot_pic(mask_central_with_max)

    mask_portal = np.zeros_like(lobule_array, dtype=np.uint8)
    vessel_portal = [offset(swap_xy(v), (minx, miny)) for v in lobule_stats.vessels_portal]
    MaskGenerator.draw_polygons(mask_portal, poly_boundary, (minx, miny), True)
    for p in vessel_portal:
        MaskGenerator.draw_polygons(mask_portal, p, (minx, miny), True)

    # plot_pic(mask_portal)

    dist_central = cv2.distanceTransform(mask_central_with_max.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)
    dist_portal = cv2.distanceTransform(mask_portal.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)

    # plot_pic(dist_central)
    # plot_pic(dist_portal)

    mask = (mask_central ^ mask_portal).astype(bool)
    area = (~mask).sum()

    pixels = lobule_array[~mask]

    # image = lobule_array.copy()
    # image[mask] = 0
    # plot_pic(image)

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

    pw = meta["pixel_size"]
    level = meta["level"]

    factor = 2 ** level * pw

    intensity = lobule_array[~mask]

    d_central = dist_central[~mask] * factor
    d_portal = dist_portal[~mask] * factor
    pv_dist = dist(d_portal, d_central)

    # print(d_central)
    # print(d_portal)

    positions = np.argwhere(mask == False)
    height = positions[:, 0] + miny
    width = positions[:, 1] + minx

    return create_lobule_df(height, width, d_portal, d_central, pv_dist, intensity, idx)


def analyse_protein_expression(protein_array: np.ndarray, slide_stats: SlideStats) -> pd.DataFrame:
    dfs = [analyse_protein_expression_for_lobule(protein_array, ls, idx, slide_stats.meta_data) for idx, ls in enumerate(slide_stats.lobule_stats)]
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

    final_df.to_csv(config.reports_path / "lobule_distances.csv", sep=",", index=False)
