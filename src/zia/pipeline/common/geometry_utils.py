"""Utilities for manipulating geometry and coordinates."""

from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely import Geometry, Polygon, LineString, LinearRing, Point, GeometryCollection
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from zia.log import get_logger

logger = get_logger(__file__)


def off_set_geometry(geometry: Geometry, offset: Tuple[int, int]):
    return shapely.ops.transform(lambda x, y: (x - offset[0], y - offset[1]), geometry)


def rescale_coords(
        coords: List[Tuple[float, float]],
        factor: float = 1.0,
        offset: Tuple[int, int] = (0, 0),
) -> List[Tuple[float, float]]:
    """Rescaling of coordinates by factor.

    :param factor: muliplicative factor
    :param offset: offset to shift coordinates
    """
    return [((x - offset[0]) * factor, (y - offset[1]) * factor) for x, y in coords]


class GeometryDraw:

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
        mask[min(round(point.y), h - 1), min(round(point.x), w - 1)] = 1 if color else 0

    @classmethod
    def _draw_multipart_recursively(cls, geometry: BaseMultipartGeometry, mask: np.ndarray, offset: Tuple[int, int], color: bool):
        for geometry in geometry.geoms:
            if isinstance(geometry, BaseMultipartGeometry):
                cls._draw_multipart_recursively(geometry, mask, offset, color)
            else:
                cls._draw_geometry(geometry, mask, offset, color)

    @classmethod
    def _draw_geometry(cls, geometry, mask, offset, color):
        geometry = off_set_geometry(geometry, offset)
        if isinstance(geometry, Polygon):
            cls._draw_polygon(geometry, mask, color)
        elif isinstance(geometry, LineString):
            cls._draw_line_string(geometry, mask, color)
        elif isinstance(geometry, Point):
            cls._draw_point(geometry, mask, color)
        else:
            logger.warning(f"Encountered geometry type {type(geometry)} is not supported.")


class AxGeometryDraw:

    @classmethod
    def draw_geometry(
            cls,
            ax: plt.Axes,
            geometry: BaseMultipartGeometry,
            offset: Tuple[int, int] = (0, 0),
            **kwargs,
    ) -> None:
        if isinstance(geometry, BaseMultipartGeometry):
            cls._draw_multipart_recursively(geometry, ax, offset, **kwargs)

        elif isinstance(geometry, BaseGeometry):
            cls._draw_geometry(geometry, ax, offset, **kwargs)
        else:
            logger.warning(f"Non polygon type geometry encountered {type(geometry)}")

    @classmethod
    def _draw_polygon(
            cls, polygon: Polygon, ax: plt.Axes, **kwargs
    ) -> None:
        if polygon.is_empty:
            return

        fc = kwargs.get("facecolor")
        ec = kwargs.get("edgecolor")
        lw = kwargs.get("linewidth")

        x, y = polygon.exterior.xy
        ax.fill(y, x, facecolor=fc, edgecolor=ec, linewidth=lw)

    @classmethod
    def _draw_line_string(
            cls,
            line_string: LineString,
            ax: plt.Axes,
            **kwargs,
    ) -> None:
        if line_string.is_empty:
            return

        fc = kwargs.get("facecolor")
        lw = kwargs.get("linewidth")

        x, y = line_string.xy

        ax.plot(y, x, color=fc, linewidth=lw)

    @classmethod
    def _draw_point(cls, point: Point, ax: plt.Axes, **kwargs) -> None:
        if point.is_empty:
            return

        fc = kwargs.get("facecolor")
        ec = kwargs.get("edgecolor")
        lw = kwargs.get("linewidth")
        ms = kwargs.get("markersize")

        x, y = point.xy

        ax.scatter(y, x, c=fc, edgecolors=ec, linewidths=lw, s=ms)

    @classmethod
    def _draw_multipart_recursively(cls, geometry: BaseMultipartGeometry, ax: plt.Axes, offset: Tuple[int, int], **kwargs):
        for geometry in geometry.geoms:
            if isinstance(geometry, BaseMultipartGeometry):
                cls._draw_multipart_recursively(geometry, ax, offset, **kwargs)
            else:
                cls._draw_geometry(geometry, ax, offset, **kwargs)

    @classmethod
    def _draw_geometry(cls, geometry: BaseGeometry, ax: plt.Axes, offset: Tuple[int, int], **kwargs):
        geometry = off_set_geometry(geometry, offset)
        if isinstance(geometry, Polygon):
            cls._draw_polygon(geometry, ax, **kwargs)
        elif isinstance(geometry, LineString):
            cls._draw_line_string(geometry, ax, **kwargs)
        elif isinstance(geometry, Point):
            cls._draw_point(geometry, ax, **kwargs)
        else:
            logger.warning(f"Encountered geometry type {type(geometry)} is not supported.")
