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
from zia.annotations.open_slide_image.data_store import DataStore, ZarrGroups
from zia.annotations.workflow_visualizations.util.image_plotting import plot_pic


IMAGE_NAME = "J-12-00348_NOR-021_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006"

logger = logging.getLogger(__name__)


class MaskGenerator:
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
            print(f"Non polygon type geometry encountered {type(polygons)}")

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

        intersection = polygons.intersection(tile_polygon)
        # print(type(intersection))

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
                    print(
                        f"Geometry Collection geometry type not yet handled: {type(intersection)}"
                    )

        else:
            print(f"Intersection geometry type not yet handled: {type(intersection)}")

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

    @classmethod
    def create_mask(cls, data_store: DataStore, annotations: List[Annotation]) -> None:
        # iterate over the range of interests
        for roi_no, roi in enumerate(data_store.rois):
            cs, rs = roi.get_bound(PyramidalLevel.ZERO)
            x_min, x_max, y_min, y_max = cs.start, cs.stop, rs.start, rs.stop

            # shape of the roi
            shape = (rs.stop - rs.start, cs.stop - cs.start)

            # create zeros array in zarr group with shape of roi
            mask_array = data_store.create_mask_array(
                ZarrGroups.LIVER_MASK, roi_no, shape
            )

            # get a list of slice that slices the area of the roi in tiles
            slices = get_tile_slices(shape)

            # get the roi polygon and offset it to the origin of the created array
            roi_poly = roi.get_polygon_for_level(
                PyramidalLevel.ZERO, offset=(x_min, y_min)
            )

            # one roi was not valid in terms if self intersections... this fixes that
            if not roi_poly.is_simple:
                roi_poly = make_valid(roi_poly)
                print(f"not simple: {type(roi_poly)}")

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
                        print(f"different geometry type encountered. {type(polygon)}")
                mask_array[rs, cs] = base_mask.astype(bool)
            plot_pic(mask_array[::16, ::16])


if __name__ == "__main__":
    pass
