import logging
from typing import Tuple, Union

import cv2
import numpy as np
import shapely
import shapely.ops
from shapely import LineString, Polygon, Geometry, LinearRing, Point
from shapely.geometry.base import BaseMultipartGeometry, BaseGeometry

logger = logging.getLogger(__name__)


class GeometryDraw:

    @classmethod
    def off_set_geometry(cls, geometry: Geometry, offset: Tuple[int, int]):
        return shapely.ops.transform(lambda x, y: (x - offset[0], y - offset[1]), geometry)

    @classmethod
    def draw_geometry(
            cls,
            mask: np.ndarray,
            geometry: Union[BaseGeometry | BaseMultipartGeometry],
            offset: Tuple[int, int],
            color: bool,
    ) -> None:
        if isinstance(geometry, BaseMultipartGeometry):
            cls._draw_multipart_recursively(geometry, mask, offset, color)

        elif isinstance(geometry, BaseGeometry):
            cls._draw_geometry(geometry, mask, offset, color)
        else:
            logger.warning(f"Non polygon type geometry encountered {type(geometry)}")

    @classmethod
    def _draw_polygon(
            cls, polygon: Polygon, mask: np.ndarray, color: bool
    ) -> None:
        if polygon.is_empty:
            return
        tile_poly_coords = polygon.exterior.coords,
        tile_poly_coords = np.array(tile_poly_coords, dtype=np.int32)
        cv2.fillPoly(mask, [tile_poly_coords], 1 if color else 0)

    @classmethod
    def _draw_line_string(
            cls,
            line_string: LineString,
            mask: np.ndarray,
            color: bool,
    ) -> None:
        if line_string.is_empty:
            return

        line_string_coords = np.array(line_string.coords, dtype=np.int32)
        cv2.polylines(
            mask,
            [line_string_coords],
            isClosed=True if isinstance(line_string, LinearRing) else False,
            color=1 if color else 0,
            thickness=1,
        )

    @classmethod
    def _draw_point(cls, point: Point, mask: np.ndarray, color: bool) -> None:
        if point.is_empty:
            return
        
        h, w = mask.shape[:2]
        mask[min(round(point.y), h-1), min(round(point.x), w-1)] = 1 if color else 0

    @classmethod
    def _draw_multipart_recursively(cls, geometry: BaseMultipartGeometry, mask: np.ndarray, offset: Tuple[int, int], color: bool):
        for geometry in geometry.geoms:
            if isinstance(geometry, BaseMultipartGeometry):
                cls._draw_multipart_recursively(geometry, mask, offset, color)
            else:
                cls._draw_geometry(geometry, mask, offset, color)

    @classmethod
    def _draw_geometry(cls, geometry, mask, offset, color):
        geometry = cls.off_set_geometry(geometry, offset)
        if isinstance(geometry, Polygon):
            cls._draw_polygon(geometry, mask, color)
        elif isinstance(geometry, LineString):
            cls._draw_line_string(geometry, mask, color)
        elif isinstance(geometry, Point):
            cls._draw_point(geometry, mask, color)
        else:
            logger.warning(f"Encountered geometry type {type(geometry)} is not supported.")
