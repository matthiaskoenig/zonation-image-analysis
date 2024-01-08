from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np
import pandas as pd
import shapely
import shapely.ops
import zarr
from shapely import Geometry, Polygon

from zia import BASE_PATH
from zia.pipeline.common.resolution_levels import PyramidalLevel
from zia.config import read_config
from zia.oven.data_store import ZarrGroups
from zia.pipeline.pipeline_components.algorithm.segementation.lobulus_statistics import SlideStats, LobuleStatistics
from zia.pipeline.common.geometry_utils import GeometryDraw
from zia.statistics.utils.data_provider import SlideStatsProvider, get_species_from_name
from imagecodecs.numcodecs import Jpeg
from numcodecs import register_codec
register_codec(Jpeg)

@np.vectorize
def dist(d_p: float, d_c: float) -> float:
    if d_c == 0 and d_p == 0:
        return 1
    return 1 - d_c / (d_p + d_c)


def open_protein_arrays(address: Path, path: str, level: PyramidalLevel, excluded: List[str]) -> Dict[str, np.ndarray]:
    group = zarr.open(store=address, path=path)
    return {key: 255 - np.array(val.get(f"{level}")) for key, val in group.items() if not key in excluded}


def swap_xy(geometry: Geometry):
    return shapely.ops.transform(lambda x, y: (y, x), geometry)


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


def analyse_protein_expression_for_lobule(protein_array: np.ndarray, foreground_mask: np.ndarray, lobule_stats: LobuleStatistics, idx: int,
                                          meta: Dict, sum_array: np.ndarray) -> \
        Optional[pd.DataFrame]:
    poly_boundary: Polygon = swap_xy(lobule_stats.polygon)
    minx, miny, maxx, maxy = (max(0, int(x)) for x in poly_boundary.bounds)

    lobule_array = protein_array[miny:maxy, minx:maxx]
    lobule_sum = sum_array[miny:maxy, minx:maxx]
    lobule_foreground = foreground_mask[miny:maxy, minx:maxx]

    mask_lobule = np.zeros_like(lobule_array, dtype=np.uint8)
    mask_portal = np.zeros_like(lobule_array, dtype=np.uint8)
    mask_central = np.zeros_like(lobule_array, dtype=np.uint8)

    GeometryDraw.draw_geometry(mask_lobule, poly_boundary, (minx, miny), True)

    vessel_central = [swap_xy(v) for v in lobule_stats.vessels_central]

    for p in vessel_central:
        GeometryDraw.draw_geometry(mask_central, p, (minx, miny), True)

    vessel_portal = [swap_xy(v) for v in lobule_stats.vessels_portal]
    for p in vessel_portal:
        GeometryDraw.draw_geometry(mask_portal, p, (minx, miny), True)

    mask_central_distance = ~mask_central.astype(bool)
    max_i = np.percentile(lobule_sum, 99)

    mask_central_distance[lobule_sum > max_i] = False

    # plot_pic(mask_lobule, "lobule")
    # plot_pic(mask_portal, "portal")
    mask_portal_distance = (mask_lobule.astype(bool) & ~mask_portal.astype(bool))

    # plot_pic(mask_portal_distance, "mask portal dist")
    # plot_pic(mask_central_distance, "mask central dist")

    dist_central = cv2.distanceTransform(mask_central_distance.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)
    dist_portal = cv2.distanceTransform(mask_portal_distance.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)

    # plot_pic(dist_central, "central dist")
    # plot_pic(dist_portal, "portal dist")

    mask = np.ones_like(lobule_array, dtype=bool)
    mask[mask_central.astype(bool)] = False
    mask[mask_portal.astype(bool)] = False
    mask[~mask_lobule.astype(bool)] = False

    # plot_pic(mask, "final mask")

    area = mask.sum()

    pixels = lobule_array[mask]
    foreground = lobule_foreground[mask]

    empty_pixels = foreground[foreground == False]
    if empty_pixels.size / area > 0.2:
        # print("empty lobule on slide")
        return None

    if pixels.size == 0:
        # print(f"to small: {pixels.size}")
        return None

    p_min, p_max = np.min(pixels), np.max(pixels)

    if p_max == p_min:
        return None

    pw = meta["pixel_size"]
    level = meta["level"]

    factor = 2 ** level * pw

    intensity = lobule_array[mask]

    d_central = dist_central[mask] * factor
    d_portal = dist_portal[mask] * factor
    pv_dist = dist(d_portal, d_central)

    positions = np.argwhere(mask)
    height = positions[:, 0] + miny
    width = positions[:, 1] + minx

    return create_lobule_df(height, width, d_portal, d_central, pv_dist, intensity, idx)


def analyse_protein_expression(protein_array: np.ndarray, fore_ground_mask: np.ndarray, slide_stats: SlideStats,
                               sum_array: np.ndarray) -> pd.DataFrame:
    dfs = [analyse_protein_expression_for_lobule(protein_array, fore_ground_mask, ls, idx, slide_stats.meta_data, sum_array) for idx, ls in
           enumerate(slide_stats.lobule_stats)]
    dfs = list(filter(lambda x: x is not None, dfs))
    return pd.concat(dfs)


def normalize_and_weight(array: np.ndarray) -> np.ndarray:
    min_, max_ = np.min(array), np.max(array)
    w = 1 - 1 / (max_ - min_)

    return w * (array - min_) / (max_ - min_)


def create_sum_array(protein_arrays) -> np.ndarray:
    normalized_arrays = [normalize_and_weight(arr) for arr in protein_arrays.values()]
    return 1 / len(normalized_arrays) * np.sum(normalized_arrays, axis=0)


def analyse_lobuli(slide_stats: SlideStats, protein_arrays: Dict[str, np.ndarray], foreground_masks: Dict[str, np.ndarray]) -> pd.DataFrame:
    protein_dfs = []
    sum_array = create_sum_array(protein_arrays)
    for protein, protein_array in protein_arrays.items():
        df = analyse_protein_expression(protein_array, foreground_masks[protein], slide_stats, sum_array)
        df["protein"] = protein
        protein_dfs.append(df)
    return pd.concat(protein_dfs)


def normalize_arrays(protein_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normalized = {}

    gs = protein_arrays.get("GS")
    if gs is not None:
        bg = np.percentile(gs[gs > 0], 20)
    else:
        cyp3a4 = protein_arrays["CYP3A4"]
        bg = np.percentile(cyp3a4[cyp3a4 > 0], 10)

    for key, arr in protein_arrays.items():
        max_i = np.percentile(arr[arr > 0], 99)
        norm = (arr - bg) / (max_i - bg)
        norm[norm < 0] = 0
        normalized[key] = norm
    return normalized


def get_foreground_mask(protein_arrays: Dict[str, np.ndarray]):
    return {key: arr > 0 for key, arr in protein_arrays.items()}


def generate_distance_df(report_path: Path, overwrite=True) -> pd.DataFrame:
    if overwrite == False and (report_path / "lobule_distances.csv").exists():
        return pd.read_csv(report_path / "lobule_distances.csv", index_col=False)

    config = read_config(BASE_PATH / "configuration.ini")

    slide_stats_dict = get_slide_stats()

    subject_dfs = []

    for subject, roi_dict in slide_stats_dict.items():
        roi_dfs = []

        excluded = SlideStatsProvider.exclusion_dict.get(subject)
        for roi, slide_stats in roi_dict.items():
            protein_arrays = open_protein_arrays(
                address=config.image_data_path / "stain_separated" / f"{subject}.zarr",
                path=f"{ZarrGroups.STAIN_1.value}/{roi}",
                level=slide_stats.meta_data["level"],
                excluded=excluded if excluded is not None else []
            )
            foreground_masks = get_foreground_mask(protein_arrays)
            normalized_arrays = normalize_arrays(protein_arrays)
            df = analyse_lobuli(slide_stats, normalized_arrays, foreground_masks)
            df["roi"] = roi
            roi_dfs.append(df)

        subject_df = pd.concat(roi_dfs)
        subject_df["subject"] = subject
        species = get_species_from_name(subject)
        subject_df["species"] = species
        subject_dfs.append(subject_df)

    final_df = pd.concat(subject_dfs)
    final_df["roi"] = pd.to_numeric(final_df["roi"])

    final_df = final_df.round({"d_portal": 3,
                               "d_central": 3,
                               "pv_dist": 3,
                               "intensity": 3})

    final_df.to_csv(report_path / "lobule_distances.csv", sep=",", index=False)

    return final_df


if __name__ == "__main__":
    config = SlideStatsProvider.config

    report_path = config.reports_path

    generate_distance_df(report_path=report_path)
