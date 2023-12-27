"""Run the image processing pipeline."""

import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from shapely import Polygon, make_valid

from zia.io.wsi_openslide import read_full_image_from_slide, read_wsi
from zia.log import get_logger
from zia.pipeline.common.annotations import Annotation, AnnotationParser, AnnotationType
from zia.pipeline.common.project_config import Configuration
from zia.pipeline.common.resolution_levels import PyramidalLevel
from zia.pipeline.common.roi import Roi
from zia.pipeline.file_management.file_management import Slide, SlideFileManager
from zia.pipeline.pipeline_components.pipeline import IPipelineComponent

logger = get_logger(__name__)


class RoiDetectionComponent(IPipelineComponent):
    """Pipeline step for ROI processing."""
    dir_name = "RoiDetection"

    def __init__(self, project_config: Configuration, file_manager: SlideFileManager, overwrite: bool = False):
        super().__init__(project_config, file_manager, RoiDetectionComponent.dir_name, overwrite)

    def exist(self, roi_file: Path) -> bool:
        if not self.overwrite and roi_file.exists():
            logger.info(f"The ROI file {roi_file.stem} already exists and overwrite is False.")
            return True
        return False

    def run(self):
        for slide in self.file_manager.slides:
            self.detect_rois_for_slide(slide)

    def get_annotations_path(self, slide: Slide) -> Path:
        p = self.project_config.annotations_path / f"{slide.name}.geojson"
        if not p.exists():
            raise FileNotFoundError(f"No common file found for {slide.name}.")
        return p

    def validate_annotations(self, liver_annotations: List[Annotation], lobe_annotations: List[Annotation], slide: Slide) -> None:
        if len(liver_annotations) == 0:
            raise Exception(f"No common of type liver found for slide {slide.name}")

        if 1 < len(liver_annotations) != len(lobe_annotations):
            raise Exception(f"The number of liver common requires a matching number of lobe annotations for slide {slide.name}.")

    def detect_rois_for_slide(self, slide: Slide) -> None:
        """Run analysis."""
        roi_file = self.get_image_data_path() / f"{slide.name}.geojson"
        # prevent from overwriting existing data
        if self.exist(roi_file):
            return

        start_time = time.time()

        geojson_path = self.get_annotations_path(slide)

        annotations = AnnotationParser.parse_geojson(path=geojson_path)

        liver_annotations = AnnotationParser.get_annotation_by_type(annotations, AnnotationType.LIVER)

        lobe_annotations = AnnotationParser.get_annotation_by_types(annotations, AnnotationType.get_lobe_annnotations())

        self.validate_annotations(liver_annotations, lobe_annotations, slide)

        liver_rois = self.find_rois(slide, liver_annotations, lobe_annotations)

        # writing this as a subdirectory to the zarr store because this is reused.
        if len(liver_rois) > 0:
            self.save_rois(liver_rois, roi_file)

        else:
            logger.warning(f"[{slide.name}]\tNo ROIs found.")

        end_time = time.time()
        run_time = end_time - start_time

        logger.info(f"[{slide.subject}\t{slide.protein}]\tFinished ROI detection in {run_time:.2f} s. Found {len(liver_rois)} ROI.")


    def save_rois(self, rois: List[Roi], roi_file: Path) -> None:
        Roi.write_to_geojson(rois, roi_file)

    def find_rois(self,
                  slide: Slide,
                  annotations: List[Annotation],
                  lobe_annotations: List[Annotation]
                  ) -> List[Roi]:

        open_slide = read_wsi(self.project_config.data_path / slide.species / slide.protein.upper() / f"{slide.name}.{self.project_config.extension}")
        region = read_full_image_from_slide(open_slide, PyramidalLevel.SEVEN)

        cv2image = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)

        # pad the image to handle edge cutting tissue regions
        padding = 10

        padded_copy = cv2.copyMakeBorder(
            cv2image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255
        )

        blur = cv2.GaussianBlur(padded_copy, (5, 5), 5, borderType=cv2.BORDER_REPLICATE)

        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10), (-1, -1))
        eroded = cv2.erode(thresh, element)
        dilated = cv2.dilate(eroded, element)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-padding, -padding)
        )

        # get contour array to shapely coordinates
        contours = filter_shapes(contours)
        shapes = [
            Polygon(transform_contour_to_shapely_coords(contour))
            for contour in contours
        ]

        # drop shapes which are within a bigger shape
        shapes.sort(key=lambda x: x.area, reverse=True)
        kept = []
        remaining = shapes[1:]  # removing image bounding box
        reduce_shapes(kept, remaining)

        # find the contour the organ shape that contains the common geometry

        contour_shapes = extract_organ_shapes(kept, annotations)

        if len(contour_shapes) == 0:
            logger.warning(f"[{slide.subject}\t{slide.protein}]\tNo organ contour matches with the common geometries")

        # reduce the polygons to have fewer points

        reduced_polys = reduce_polygons(contour_shapes, slide)

        lobe_roi_dict = map_rois(reduced_polys, lobe_annotations)

        liver_rois = [Roi(poly, cv2image.shape[:2], PyramidalLevel.SEVEN, AnnotationType.LIVER, lobe_id)
                      for lobe_id, poly in lobe_roi_dict.items()]

        return liver_rois


def reduce_polygons(polygons: List[Polygon], slide: Slide) -> List[Polygon]:
    # reduce the polygons to have fewer points

    reduced_polys = []

    # reduces the number of points in the polygon
    # this may result in invalid polygons which have to be fixed
    for polygon in polygons:
        len_before = len(polygon.exterior.coords)
        tolerance = 2
        reduced_geometry = polygon.simplify(tolerance, False)
        reduced_polygon = Polygon(reduced_geometry)

        len_after = len(reduced_polygon.exterior.coords)
        factor = len_before / len_after
        logger.debug(f"[{slide.subject}\t{slide.protein}]\tReduced polygon vertices by factor {factor:.1f}")
        if not reduced_polygon.is_valid:
            logger.warning(f"[{slide.subject}\t{slide.protein}]\tInvalid Polygon encountered after reduction.")
            reduced_polygon = make_valid(reduced_polygon)
            logger.debug(f"[{slide.subject}\t{slide.protein}]\tMade Polygon valid.")

        reduced_polys.append(reduced_polygon)
    return reduced_polys


def extract_organ_shapes(
        shapes: List[Polygon], organ_annotations: List[Annotation]
) -> List[Polygon]:
    extracted = []
    for shape in shapes:
        for anno in organ_annotations:
            if shape.contains(anno.get_resized_geometry(PyramidalLevel.SEVEN)) and shape not in extracted:
                extracted.append(shape)
    return extracted


def filter_shapes(contours):
    return [contour for contour in contours if contour.shape[0] >= 4]


def transform_contour_to_shapely_coords(contour):
    return tuple([(x, y) for x, y in contour[:, 0, :]])


def reduce_shapes(
        kept_shapes:
        List[Polygon], remaining_shapes: List[Polygon]
) -> None:
    not_in_bigger_shape = []
    big_shape = remaining_shapes.pop(0)
    kept_shapes.append(big_shape)
    for smaller_shape in remaining_shapes:
        if not smaller_shape.within(big_shape):
            not_in_bigger_shape.append(smaller_shape)

    if len(not_in_bigger_shape) == 1:
        kept_shapes.append(not_in_bigger_shape[0])
    elif len(not_in_bigger_shape) == 0:
        return
    else:
        reduce_shapes(kept_shapes, not_in_bigger_shape)


def map_rois(contour_shapes: List[Polygon], lobe_annotations: List[Annotation]):
    if len(lobe_annotations) == 0 and len(contour_shapes) == 1:
        return {"0": contour_shapes[0]}

    if len(lobe_annotations) != len(contour_shapes):
        logger.warning(
            f"The number of tissue ROIs ({len(contour_shapes)}) does not match the number of lobule annotations ({len(lobe_annotations)}).")

    lobe_roi_dict = {}
    for shape in contour_shapes:
        for anno in lobe_annotations:
            if shape.contains(anno.get_resized_geometry(PyramidalLevel.SEVEN)):
                lobe_roi_dict[anno.annotation_class.split("_")[1]] = shape

    if len(contour_shapes) != len(lobe_roi_dict):
        logger.warning("Not all ROIs were matched with a lobe common")

    return lobe_roi_dict
