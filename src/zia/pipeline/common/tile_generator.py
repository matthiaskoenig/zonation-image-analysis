from typing import List, Tuple, Iterator

import numpy as np
import shapely
import zarr
from shapely import Polygon

from zia.log import get_logger
from zia.statistics.expression_profile.geometry_utils.polygon_drawing import GeometryDraw

log = get_logger(__file__)


class TileGenerator:
    def __init__(self, slices: List[Tuple[slice, slice]],
                 array: zarr.Array,
                 roi_shape: Tuple[int, int, int],
                 tile_size: int,
                 roi_polygon: Polygon):
        self.slices = slices
        self.array = array
        self.tile_size = tile_size
        self.roi_shape = roi_shape
        self.roi_polygon = roi_polygon

    def get_tiles(self) -> Iterator[np.ndarray]:
        for rs, cs in self.slices:
            yield self._prepare_tile(rs, cs)

    def _prepare_tile(self, rs: slice, cs: slice):
        arr = self.array[rs, cs]
        mask = self._create_foreground_mask(rs, cs)
        arr[~mask] = [255, 255, 255]

        return arr

    def _create_foreground_mask(self, rs: slice, cs: slice) -> np.ndarray:
        # tile shape -> don't change to tile size, because tiles at the edges of the roi are not squares but rectangles
        tile_shape = (rs.stop - rs.start, cs.stop - cs.start)

        # transient mask for tile
        base_mask = np.zeros(shape=tile_shape, dtype=np.uint8)

        tile_polygon = shapely.geometry.box(cs.start, rs.start, cs.stop, rs.stop)

        intersection = self.roi_polygon.intersection(tile_polygon)

        GeometryDraw.draw_geometry(mask=base_mask, geometry=intersection, offset=(cs.start, rs.start), color=True)

        return base_mask.astype(bool)

