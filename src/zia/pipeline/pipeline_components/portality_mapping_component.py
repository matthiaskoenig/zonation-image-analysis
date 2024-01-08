from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import zarr
from shapely import Geometry, Polygon, LineString, Point, GeometryCollection
from shapely.ops import transform

from zia.log import get_logger
from zia.pipeline.common.geometry_utils import GeometryDraw, off_set_geometry
from zia.pipeline.common.project_config import Configuration
from zia.pipeline.common.resolution_levels import PyramidalLevel
from zia.pipeline.file_management.file_management import SlideFileManager
from zia.pipeline.pipeline_components.algorithm.segementation.lobulus_statistics import SlideStats, LobuleStatistics
from zia.pipeline.pipeline_components.pipeline import IPipelineComponent
from zia.pipeline.pipeline_components.segementation_component import SegmentationComponent
from zia.pipeline.pipeline_components.stain_separation_component import StainSeparationComponent, Stain

log = get_logger(__file__)


class PortalityMappingComponent(IPipelineComponent):
    """Pipeline step for lobuli segmentation."""
    dir_name = "PortalityMap"

    def __init__(self, project_config: Configuration, file_manager: SlideFileManager, exclusion_dict=None,
                 overwrite: bool = False):
        super().__init__(project_config, file_manager, PortalityMappingComponent.dir_name, overwrite)
        if exclusion_dict is None:
            exclusion_dict = {}
        self.exclusion_dict = exclusion_dict

    def run(self) -> None:
        if not self.overwrite and (self.image_data_path / "lobule_distances.csv").exists():
            log.info("SlideStatistics data frame already exists.")

        log.info("Started generating portality map data frame.")
        slide_stats_df = self.generate_distance_df()

        log.info("Saving data frame.")
        slide_stats_df.to_csv(self.image_data_path / "lobule_distances.csv")

    def get_roi_dirs(self, subject: str) -> Dict[str, Path]:
        subject_path = self.project_config.image_data_path / SegmentationComponent.dir_name / subject
        if not subject_path.exists():
            raise FileNotFoundError(f"Segmentation result path not found for subject {subject}.")
        return {p.stem: p for p in subject_path.iterdir()}

    def generate_distance_df(self) -> pd.DataFrame:

        subject_dfs = []
        for subject, _ in self.file_manager.group_by_subject().items():
            roi_dfs = []
            excluded = self.exclusion_dict.get(subject)
            for roi, slide_stats_path in self.get_roi_dirs(subject).items():
                slide_stats = SlideStats.load_from_file_system(slide_stats_path)
                zarr_paths = self.get_zarr_path(subject, roi)
                protein_arrays = open_protein_arrays(
                    zarr_paths,
                    level=slide_stats.meta_data["level"],
                    excluded=excluded if excluded is not None else []
                )
                foreground_masks = get_foreground_mask(protein_arrays)
                normalized_arrays = normalize_arrays(protein_arrays)
                df = analyse_lobuli(slide_stats, normalized_arrays, foreground_masks)
                df["roi"] = roi
                df["species"] = slide_stats.meta_data["subject"]
                roi_dfs.append(df)
                # save the slide stats updated with the local maxima
                slide_stats.to_geojson(slide_stats_path)

            subject_df = pd.concat(roi_dfs)
            subject_df["subject"] = subject
            subject_dfs.append(subject_df)

        final_df = pd.concat(subject_dfs)
        final_df["roi"] = pd.to_numeric(final_df["roi"])

        final_df = final_df.round({"d_portal": 3,
                                   "d_central": 3,
                                   "pv_dist": 3,
                                   "intensity": 3})

        return final_df

    def get_zarr_path(self, subject: str, lobe_id: str) -> Dict[str, Path]:
        base_path = self.project_config.image_data_path / StainSeparationComponent.dir_name / f"{Stain.ONE.value}" / subject / lobe_id
        if not base_path.exists():
            raise FileNotFoundError(f"No stain separation directory exists for subject {subject} and lobe {lobe_id}.")

        return {p.stem: p for p in base_path.iterdir()}


@np.vectorize
def dist(d_p: float, d_c: float) -> float:
    if d_c == 0 and d_p == 0:
        return 1
    return 1 - d_c / (d_p + d_c)


def open_protein_arrays(zarr_path: Dict[str, Path], level: PyramidalLevel, excluded: List[str]) -> Dict[str, np.ndarray]:
    return {protein: 255 - np.array(zarr.open_array(store=path, path=f"{level}"))
            for protein, path in zarr_path.items()
            if protein not in excluded}


def swap_xy(geometry: Geometry):
    return transform(lambda x, y: (y, x), geometry)


def create_lobule_df(height: np.ndarray,
                     width: np.ndarray,
                     d_portal: np.ndarray,
                     d_central: np.ndarray,
                     pv_dist: np.ndarray,
                     intensity: np.ndarray,
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


def set_local_lobule_maxima(local_maxima: np.ndarray, lobule_stats: LobuleStatistics, off_set: Tuple[int, int]) -> None:
    contours, _ = cv2.findContours(local_maxima.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Extract polygons for each contour
    geometries = []
    for contour in contours:
        if len(contour) >= 3:
            # Create Polygon for contours with 3 or more points
            contour = np.squeeze(contour)
            polygon = off_set_geometry(swap_xy(Polygon(contour)), off_set)
            geometries.append(polygon)
        elif len(contour) == 2:
            # Create LineString for contours with 2 points
            contour = np.squeeze(contour)
            linestring = off_set_geometry(swap_xy(LineString(contour)), off_set)
            geometries.append(linestring)
        elif len(contour) == 1:
            # Create Point for contours with 1 point
            point = off_set_geometry(swap_xy(Point(contour.squeeze())), off_set)
            geometries.append(point)

    lobule_stats.local_maxima = GeometryCollection(geometries)


def analyse_protein_expression_for_lobule(protein_array: np.ndarray, foreground_mask: np.ndarray, lobule_stats: LobuleStatistics, idx: int,
                                          meta: Dict, sum_array: np.ndarray) -> Optional[pd.DataFrame]:
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

    local_maxima = lobule_sum > max_i
    set_local_lobule_maxima(local_maxima, lobule_stats, (-miny, -minx))

    mask_central_distance[local_maxima] = False

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

    gs = protein_arrays.get("gs")
    if gs is not None:
        bg = np.percentile(gs[gs > 0], 20)
    else:
        cyp3a4 = protein_arrays["cyp3a4"]
        bg = np.percentile(cyp3a4[cyp3a4 > 0], 10)

    for key, arr in protein_arrays.items():
        max_i = np.percentile(arr[arr > 0], 99)
        norm = (arr - bg) / (max_i - bg)
        norm[norm < 0] = 0
        normalized[key] = norm
    return normalized


def get_foreground_mask(protein_arrays: Dict[str, np.ndarray]):
    return {key: arr > 0 for key, arr in protein_arrays.items()}
