import logging
from typing import List, Tuple, Union

import cv2
import numpy as np
import shapely
from shapely import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.validation import make_valid

from zia.annotations.annotation.annotations import Annotation
from zia.annotations.annotation.geometry_utils import rescale_coords
from zia.annotations.annotation.slicing import get_tile_slices
from zia.annotations.annotation.util import PyramidalLevel
from zia.data_store import DataStore, ZarrGroups
from zia.log import create_message

logger = logging.getLogger(__name__)


class MaskGenerator:

    @classmethod
    def create_mask(cls, data_store: DataStore, annotations: List[Annotation]) -> None:
        image_id = data_store.image_info.metadata.image_id
        # iterate over the range of interests
        for roi_no, roi in enumerate(data_store.rois):
            logger.info(f"[{image_id}]\tStarted Mask Generation for ROI {roi_no}")

            cs, rs = roi.get_bound(PyramidalLevel.ZERO)
            x_min, x_max, y_min, y_max = cs.start, cs.stop, rs.start, rs.stop

            # shape of the roi
            shape = (rs.stop - rs.start, cs.stop - cs.start)

            # create pyramidal group to persist mask
            pyramid_dict = data_store.create_pyramid_group(
                ZarrGroups.LIVER_MASK, roi_no, shape, bool
            )

            # get a list of slice that slices the area of the roi in tiles
            slices = get_tile_slices(shape)

            # get the roi polygon and offset it to the origin of the created array
            roi_poly = roi.get_polygon_for_level(
                PyramidalLevel.ZERO, offset=(x_min, y_min)
            )

            # one roi was not valid in terms if self intersections... this fixes that
            if not roi_poly.is_valid:
                logger.warning(
                    create_message(image_id, "Non valid polygon encountered."))
                roi_poly = make_valid(roi_poly)
                logger.info(create_message(image_id, "Made Polygon valid."))

            # iterate over the tiles
            for rs, cs in slices:
                # tile shape -> don't change to tile size, because tiles at the edges of the roi are not squares but rectangles
                tile_shape = (rs.stop - rs.start, cs.stop - cs.start)

                # transient mask for tile
                base_mask = np.zeros(shape=tile_shape, dtype=np.uint8)

                # draw the roi poly on the mask
                cls._draw_geometry_on_tile(
                    polygon=roi_poly, mask=base_mask, slices=(rs, cs), color=True
                )

                # draw annotations on the mask
                for annotation in annotations:
                    polygon = annotation.get_resized_geometry(
                        level=PyramidalLevel.ZERO, offset=(x_min, y_min)
                    )
                    if isinstance(polygon, (Polygon, MultiPolygon, LineString)):
                        cls._draw_geometry_on_tile(
                            polygon=polygon,
                            mask=base_mask,
                            slices=(rs, cs),
                            color=False,
                        )
                    else:
                        logger.warning(
                            create_message(image_id,
                                           f"Different geometry type encountered in annotations: {type(polygon)}."))

                # TODO: refactor out to generalize and make reusable for other components
                # create a dict to store the downsampled tile masks
                down_sample_masks = {0: base_mask}
                for i in range(len(pyramid_dict) - 1):
                    down_sample_masks[i+1] = cv2.pyrDown(down_sample_masks[i])

                # persist tile pyramidacilly
                for i, level_array in pyramid_dict.items():
                    factor = 2 ** i

                    # resize the slice for level
                    new_rs = slice(int(rs.start / factor), int(rs.stop / factor))
                    new_cs = slice(int(cs.start / factor), int(cs.stop / factor))

                    level_array[new_rs, new_cs] = down_sample_masks[i].astype(bool)

            logger.info(create_message(image_id,
                                       f"Finished mask creation for ROI {roi_no}."))

    @classmethod
    def _draw_polygons(
        cls,
        mask: np.ndarray,
        polygons: Union[Polygon | MultiPolygon],
        offset: Tuple[int, int],
        color: bool,
    ) -> None:
        if isinstance(polygons, Polygon):
            cls._draw_polygon(polygons, mask, offset, color)

        elif isinstance(polygons, MultiPolygon):
            for polygon in polygons.geoms:
                cls._draw_polygon(polygon, mask, offset, color)
        else:
            logger.warning(f"Non polygon type geometry encountered {type(polygons)}")

    @classmethod
    def _draw_polygon(
        cls, polygon: Polygon, mask: np.ndarray, offset: Tuple[int, int], color: bool
    ) -> None:
        if polygon.is_empty:
            return
        tile_poly_coords = rescale_coords(
            polygon.exterior.coords, offset=offset
        )  # (cs.start, rs.start)
        tile_poly_coords = np.array(tile_poly_coords, dtype=np.int32)
        cv2.fillPoly(mask, [tile_poly_coords], 1 if color else 0)

    @classmethod
    def _draw_line_string(
        cls,
        line_string: LineString,
        mask: np.ndarray,
        offset: Tuple[int, int],
        color: bool,
    ) -> None:
        if line_string.is_empty:
            return
        line_string_coords = rescale_coords(
            line_string.coords, offset=offset
        )  # (cs.start, rs.start)
        line_string_coords = np.array(line_string_coords, dtype=np.int32)
        cv2.polylines(
            mask,
            [line_string_coords],
            isClosed=False,
            color=1 if color else 0,
            thickness=1,
        )

    @classmethod
    def _intersect_polygons_with_tile(
        cls, polygons: Union[Polygon, MultiPolygon], tile_polygon: Polygon
    ) -> Union[Polygon | MultiPolygon]:
        i_polygons: List[Union[MultiPolygon | Polygon]] = []
        i_line_strings: List[LineString] = []

        if polygons.intersects(tile_polygon):

            intersection = polygons.intersection(tile_polygon)

            if isinstance(intersection, (Polygon, MultiPolygon)):
                i_polygons.append(intersection)

            elif isinstance(intersection, LineString):
                i_line_strings.append(intersection)

            elif isinstance(intersection, GeometryCollection):
                for geometry in intersection.geoms:
                    if isinstance(geometry, (Polygon, MultiPolygon)):
                        i_polygons.append(geometry)

                    elif isinstance(geometry, LineString):
                        i_line_strings.append(geometry)

                    else:
                        logger.warning(
                            f"Geometry Collection geometry type not yet handled: {type(intersection)}")

            else:
                logger.warning(
                    f"Intersection geometry type not yet handled: {type(intersection)}")

        return i_polygons, i_line_strings

    @classmethod
    def _draw_geometry_on_tile(
        cls,
        polygon: Union[MultiPolygon | Polygon],
        mask: np.ndarray,
        slices: Tuple[slice, slice],
        color: bool,
    ):
        rs, cs = slices
        # shapely polygon of for the tile
        tile_polygon = shapely.geometry.box(cs.start, rs.start, cs.stop, rs.stop)

        # calculate intersection of the tile polygon and the roi polygon(s)
        polygons, line_strings = cls._intersect_polygons_with_tile(
            polygons=polygon, tile_polygon=tile_polygon
        )

        for polygon in polygons:
            cls._draw_polygons(
                polygons=polygon, mask=mask, offset=(cs.start, rs.start), color=color
            )

        for line_string in line_strings:
            cls._draw_line_string(
                line_string, mask=mask, offset=(cs.start, rs.start), color=color
            )
        # plot_pic(base_mask)
