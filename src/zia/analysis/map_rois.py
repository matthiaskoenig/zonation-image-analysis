from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import ImageDraw, ImageFont

from zia import BASE_PATH
from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic
from zia.config import read_config
from zia.data_store import DataStore
from zia.io.wsi_openslide import read_full_image_from_slide
from zia.path_utils import FileManager, filter_factory
import cv2


def calculate_hu_distance(roi0: Roi, roi1: Roi):
    cs0, rs0 = roi0.get_bound(PyramidalLevel.SEVEN)
    cs1, rs1 = roi1.get_bound(PyramidalLevel.SEVEN)

    img0 = np.zeros(shape=(rs0.stop - rs0.start, cs0.stop - cs0.start),
                    dtype=np.uint8)
    img1 = np.zeros(shape=(rs1.stop - rs1.start, cs1.stop - cs1.start),
                    dtype=np.uint8)

    cnr0 = np.array(roi0.get_polygon_for_level(PyramidalLevel.SEVEN, offset=(
        cs0.start, rs0.start)).exterior.coords, np.int32)
    cnr1 = np.array(roi1.get_polygon_for_level(PyramidalLevel.SEVEN, offset=(
        cs1.start, rs1.start)).exterior.coords, np.int32)

    cv2.fillPoly(img0, pts=[cnr0], color=255)
    cv2.fillPoly(img1, pts=[cnr1], color=255)

    #plot_pic(img0)
    #plot_pic(img1)

    d = cv2.matchShapes(img0, img1, cv2.CONTOURS_MATCH_I2, 0)
    return d


def get_mapping_from_distances(distances: np.ndarray):
    mappings = {}

    while len(mappings) < distances.shape[0]:
        r, m = np.unravel_index(np.nanargmin(distances), distances.shape)
        mappings[r] = m
        distances[r, :] = np.nan
        distances[:, m] = np.nan

    return mappings


def _draw_rois(
    liver_rois: List[Roi],
    mapping: Optional[Dict[int, int]],
    image_path: Path,
    data_store: DataStore,
) -> None:
    """Draw rois."""
    region = read_full_image_from_slide(data_store.image, 7)
    draw = ImageDraw.Draw(region)
    font = ImageFont.truetype("arial.ttf", 40)
    print(len(liver_rois))
    for i, liver_roi in enumerate(liver_rois):
        polygon = liver_roi.get_polygon_for_level(
            PyramidalLevel.SEVEN
        )
        poly_points = polygon.exterior.coords
        text_coords = polygon.centroid
        draw.polygon(list(poly_points), outline="red", width=3)
        draw.text((text_coords.x, text_coords.y),
                  str(mapping.get(i)) if mapping else str(i), font=font)

    region.save(image_path, "PNG")


if __name__ == "__main__":

    file_manager = FileManager(
        configuration=read_config(BASE_PATH / "configuration.ini"),
        filter=filter_factory(negative=False)
    )

    report_path = file_manager.results_path / "ordered_rois"
    report_path.mkdir(exist_ok=True)

    image_info_by_subject = file_manager.image_info_grouped_by_subject()
    for subject in image_info_by_subject:
        print(subject)
        protein_roi_dict = {}
        for image_path in image_info_by_subject[subject]:
            data_store = DataStore(image_path)

            protein_roi_dict[data_store.image_info.metadata.protein] = data_store.rois

        reference_rois = protein_roi_dict["he"]
        protein_mappings = {}
        if len(reference_rois) > 1:
            for protein, rois in protein_roi_dict.items():
                print(protein)
                distances = np.empty(shape=(len(reference_rois), len(rois)))
                for i, roi0 in enumerate(reference_rois):
                    for k, roi1 in enumerate(rois):
                        distances[i, k] = calculate_hu_distance(roi0, roi1)
                print(distances)
                protein_mappings[protein] = get_mapping_from_distances(distances)

        for image_path in image_info_by_subject[subject]:
            data_store = DataStore(image_path)
            protein = data_store.image_info.metadata.protein
            print(protein)
            _draw_rois(data_store.rois, protein_mappings.get(protein),
                       report_path / f"{data_store.image_info.metadata.image_id}.png",
                       data_store)
