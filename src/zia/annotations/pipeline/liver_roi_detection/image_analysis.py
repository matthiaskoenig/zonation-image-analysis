import logging
from typing import List, Optional

import cv2
import numpy as np
from shapely import Polygon, make_valid

from zia.annotations.annotation.annotations import (
    Annotation,
    AnnotationParser,
    AnnotationType,
)
from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.util import PyramidalLevel
from zia.annotations.open_slide_image.data_store import DataStore
from zia.console import console
from zia.io.wsi_openslide import read_full_image_from_slide

logger = logging.getLogger(__name__)


class RoiSegmentation:
    visuals = []

    @classmethod
    def find_rois(
            cls,
            data_store: DataStore,
            annotations: Optional[List[Annotation]],
            annotation_type: AnnotationType,
    ) -> List[Roi]:
        region = read_full_image_from_slide(data_store.image, PyramidalLevel.SEVEN)

        cv2image = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)

        # pad the image to handle edge cutting tissue regions
        padding = 10

        padded_copy = cv2.copyMakeBorder(
            cv2image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255
        )

        blur = cv2.GaussianBlur(padded_copy, (5, 5), 5, borderType=cv2.BORDER_REPLICATE)

        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-padding, -padding)
        )

        # get contour array to shapely coordinates
        contours = RoiSegmentation.filter_shapes(contours)
        shapes = [
            Polygon(RoiSegmentation.transform_contour_to_shapely_coords(contour))
            for contour in contours
        ]

        # drop shapes which are within a bigger shape
        shapes.sort(key=lambda x: x.area, reverse=True)
        kept = []
        remaining = shapes[1:]  # removing image bounding box
        RoiSegmentation.reduce_shapes(kept, remaining)
        # load

        liver_annotations = AnnotationParser.get_annotation_by_type(
            annotations, annotation_type
        )

        if len(liver_annotations) == 0:
            logger.warning("No annotations of type 'Liver' where found.")
            return []

        # find the contour the organ shape that contains the annotation geometry

        contour_shapes = RoiSegmentation._extract_organ_shapes(kept, liver_annotations)

        if len(contour_shapes) == 0:
            logger.warning("No organ contour matches with the annotation geometries.")

        # reduce the polygons to have fewer points

        reduced_polys = []

        # reduces the number of points in the polygon
        # this may result in invalid polygons which have to be fixed
        for polygon in contour_shapes:
            len_before = len(polygon.exterior.coords)
            tolerance = 2
            reduced_geometry = polygon.simplify(tolerance, False)
            reduced_polygon = Polygon(reduced_geometry)

            len_after = len(reduced_polygon.exterior.coords)
            factor = len_before / len_after
            console.print(f"Reduced polygon vertices by factor {factor:.1f}")
            if not reduced_polygon.is_valid:
                console.print(f"Invalid Polygon encountered after reduction for '{data_store.name}'")
                reduced_polygon = make_valid(reduced_polygon)
                console.print(f"Made Polygon valid for '{data_store.name}'")

            reduced_polys.append(reduced_polygon)

        liver_rois = [
            Roi(poly, PyramidalLevel.SEVEN, AnnotationType.LIVER) for poly in reduced_polys
        ]

        return liver_rois

    @classmethod
    def _extract_organ_shapes(
            cls, shapes: List[Polygon], organ_annotations: List[Annotation]
    ) -> List[Polygon]:
        extracted = [
            shape
            for shape in shapes
            for anno in organ_annotations
            if shape.contains(anno.get_resized_geometry(PyramidalLevel.SEVEN))
        ]
        return extracted

    @classmethod
    def filter_shapes(cls, contours):
        return [contour for contour in contours if contour.shape[0] >= 4]

    @classmethod
    def transform_contour_to_shapely_coords(cls, contour):
        return tuple([(x, y) for x, y in contour[:, 0, :]])

    @classmethod
    def reduce_shapes(
            cls, kept_shapes: List[Polygon], remaining_shapes: List[Polygon]
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
            RoiSegmentation.reduce_shapes(kept_shapes, not_in_bigger_shape)
