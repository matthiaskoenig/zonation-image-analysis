from typing import List, Optional

import cv2
import numpy as np
from shapely import Polygon, make_valid

from zia.pipeline.annotation import (
    Annotation,
    AnnotationParser,
    AnnotationType,
)
from zia.pipeline.annotation import Roi
from zia.pipeline.annotation import PyramidalLevel
from zia.data_store import DataStore
from zia.io.wsi_openslide import read_full_image_from_slide
from zia.log import get_logger

logger = get_logger(__name__)


class RoiSegmentation:
    visuals = []

    @classmethod
    def find_rois(
        cls,
        data_store: DataStore,
        annotations: Optional[List[Annotation]],
        annotation_type: AnnotationType,
    ) -> List[Roi]:
        image_id = data_store.image_info.metadata.image_id

        region = read_full_image_from_slide(data_store.image, PyramidalLevel.SEVEN)

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

        lobe_annotations = AnnotationParser.get_annotation_by_types(annotations, [])

        if len(liver_annotations) == 0:
            logger.warning(f"[{image_id}]\tNo annotations of type 'Liver' where found",
                           extra={"image_id": image_id})
            return []

        # find the contour the organ shape that contains the common geometry

        contour_shapes = RoiSegmentation._extract_organ_shapes(kept, liver_annotations)

        if len(contour_shapes) == 0:
            logger.warning(
                f"[{image_id}]\tNo organ contour matches with the annotation geometries",
                extra={"image_id": image_id})

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
            logger.info(
                f"[{image_id}]\tReduced polygon vertices by factor {factor:.1f}")
            if not reduced_polygon.is_valid:
                logger.warning(
                    f"[{image_id}]\tInvalid Polygon encountered after reduction."
                )
                reduced_polygon = make_valid(reduced_polygon)
                logger.info(f"[{image_id}]\tMade Polygon valid.")

            reduced_polys.append(reduced_polygon)

        for poly in reduced_polys:
            print(poly.bounds)

        liver_rois = [
            Roi(poly, cv2image.shape[:2], PyramidalLevel.SEVEN, AnnotationType.LIVER)
            for poly in reduced_polys
        ]

        for roi in liver_rois:
            print(roi.get_bound(PyramidalLevel.SEVEN))

        return liver_rois

    @classmethod
    def _extract_organ_shapes(
        cls, shapes: List[Polygon], organ_annotations: List[Annotation]
    ) -> List[Polygon]:
        extracted = []
        for shape in shapes:
            for anno in organ_annotations:
                if shape.contains(anno.get_resized_geometry(PyramidalLevel.SEVEN)) and shape not in extracted:
                    extracted.append(shape)
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
