import logging
from typing import List, Optional

import cv2
import numpy as np
import skimage.measure
from openslide import OpenSlide
from shapely import Polygon

from zia.annotations.annotation.annotations import (
    Annotation,
    AnnotationParser,
    AnnotationType,
)
from zia.io.wsi_openslide import read_full_image_from_slide
from zia.annotations.annotation.roi import Roi
from zia.annotations.annotation.util import PyramidalLevel

logger = logging.getLogger(__name__)


class RoiSegmentation:
    visuals = []

    @classmethod
    def find_rois(
            cls,
            image: OpenSlide,
            annotations: Optional[List[Annotation]],
            annotation_type: AnnotationType,
    ) -> List[Roi]:
        region = read_full_image_from_slide(image, 7)

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

        ## reduce the polygons to have fewer points

        reduced_polys = []

        for polygon in contour_shapes:
            coords = np.array(polygon.exterior.coords)
            len_before = len(coords)
            tolerance = 2
            reduced_coords = skimage.measure.approximate_polygon(coords, tolerance)
            len_after = len(reduced_coords)
            print(len_after / len_before)
            reduced_polys.append(Polygon(reduced_coords))

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
            if shape.contains(anno.get_resized_geometry(128))
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
