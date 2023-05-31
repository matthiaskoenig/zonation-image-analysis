from typing import List

import cv2
import numpy as np
from openslide import OpenSlide
from shapely import Polygon
import logging

from zia.annotations.annotation.annotations import Annotation, AnnotationParser, \
    AnnotationType
from zia.annotations.annotation.geometry_utils import read_full_image_from_slide
from zia.annotations.annotation.roi import Roi, PyramidalLevel

logger = logging.getLogger(__name__)
class RoiSegmentation:

    @classmethod
    def find_rois(cls, image: OpenSlide, annotations: List[Annotation],
                  annotation_type: AnnotationType):
        region = read_full_image_from_slide(image, 7)

        cv2image = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)

        blur = cv2.GaussianBlur(cv2image, (5, 5), 5)

        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get contour array to shapely coordinates
        contours = RoiSegmentation.filter_shapes(contours)
        shapes = [Polygon(RoiSegmentation.transform_contour_to_shapely_coords(contour))
                  for contour in contours]

        # drop shapes which are within a bigger shape
        shapes.sort(key=lambda x: x.area, reverse=True)
        kept = []
        remaining = shapes[1:]  # removing image bounding box
        RoiSegmentation.reduce_shapes(kept, remaining)
        # load

        liver_annotations = AnnotationParser.get_annotation_by_type(annotations,
                                                                    annotation_type)

        if len(liver_annotations) == 0:
            logger.warning("No annotations of type 'Liver' where found.")
            return None

        liver_anno = liver_annotations[0]

        # find the contour the organ shape that contains the annotation geometry

        contour_shapes = [shape for shape in kept if
                          shape.contains(liver_anno.get_resized_geometry(128))]

        if len(contour_shapes) == 0:
            logger.warning("No organ contour matches with the annotation geometry.")

        liver_roi = Roi(contour_shapes[0], PyramidalLevel.SEVEN, AnnotationType.LIVER)

        return liver_roi

    @classmethod
    def filter_shapes(cls, contours):
        return [contour for contour in contours if contour.shape[0] >= 4]

    @classmethod
    def transform_contour_to_shapely_coords(cls, contour):
        return tuple([(x, y) for x, y in contour[:, 0, :]])

    @classmethod
    def reduce_shapes(cls, kept_shapes: List[Polygon],
                      remaining_shapes: List[Polygon]) -> None:
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
