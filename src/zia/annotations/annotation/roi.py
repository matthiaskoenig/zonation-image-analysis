from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional

import geojson as gj
from shapely import Polygon
from shapely.geometry import shape

from zia.annotations.annotation.annotations import AnnotationType
from zia.annotations.annotation.geometry_utils import rescale_coords
from zia.annotations.annotation.util import PyramidalLevel


class Roi:
    def __init__(
            self, polygon: Polygon, image_size: Optional[Tuple[int, int]], level: PyramidalLevel, annotation_type: AnnotationType
    ):
        self._geometry = Roi.normalize_coords(polygon, image_size) if image_size is not None else polygon
        self._level = level
        self.annotation_type = annotation_type

    def get_polygon_for_level(self, level: PyramidalLevel, offset=(0, 0)) -> Polygon:
        factor = 2 ** (self._level - level)
        offset = tuple(x / factor for x in offset)
        return Polygon(rescale_coords(self._geometry.exterior.coords, factor, offset))

    def _to_geojson_feature(self) -> gj.Feature:
        geojson_dict = self._geometry.__geo_interface__
        polygon = gj.Polygon(coordinates=geojson_dict["coordinates"])
        properties = {"level": self._level,
                      "annotationType": self.annotation_type
                      }

        return gj.Feature(geometry=polygon, properties=properties)

    @classmethod
    def write_to_geojson(cls, rois: List[Roi], path: Path):
        features = [roi._to_geojson_feature() for roi in rois]
        feature_collection = gj.FeatureCollection(features)

        with open(path, "w") as f:
            json.dump(feature_collection, f)

    @classmethod
    def load_from_file(cls, path: Path) -> List[Roi]:
        with open(path, "r") as f:
            feature_collection = gj.load(f)

        return [
            Roi._parse_feature(feature) for feature in feature_collection["features"]
        ]

    @classmethod
    def _parse_feature(cls, feature: dict) -> Roi:
        if not isinstance(feature.get("geometry"), gj.Polygon):
            raise ImportError("The parsed geojson geometry for a ROI must be a Polygon")

        properties: dict = feature.get("properties")
        if "level" not in properties.keys():
            raise KeyError(
                "The geojson object must contain a the element 'properties.level'"
            )
        if "annotationType" not in properties.keys():
            raise KeyError(
                "The geojson object must contain a the element 'properties.level'"
            )

        geometry = shape(feature.get("geometry"))
        level = PyramidalLevel.get_by_numeric_level(properties.get("level"))
        annotation_type = AnnotationType.get_by_string(properties.get("annotationType"))

        return Roi(geometry, None, level, annotation_type)

    def get_bound(self, level: PyramidalLevel) -> Tuple[slice, slice]:
        """
        returns slice for tuple of from ((slice(min_x, max_x), slice(min_y, max_y))
        """
        poly = self.get_polygon_for_level(level)
        bounds = poly.bounds
        # bounds where generated from padded image. setting negative bounds to zero to have valid slices
        xs = slice(int(bounds[0]), int(bounds[2]))
        ys = slice(int(bounds[1]), int(bounds[3]))
        return xs, ys

    @classmethod
    def normalize_coords(cls, polygon: Polygon, img_size: Tuple[int, int]) -> Polygon:
        """
        The polygons are generated from a padded image. Therfore, teh bounderies may be located outside
        of the image. This method aligns thos boundaries with the image boundaries.
        @param polygon: the polygon to normalize to image
        @param img_size: the size of the image
        @return: new polygon with aligned boundaries
        """
        return Polygon([cls.normalize(coord, img_size) for coord in polygon.exterior.coords])

    @classmethod
    def normalize(cls, coord: Tuple[int, int], img_size: Tuple[int, int]) -> Tuple[int, int]:
        x, y = coord
        h, w = img_size

        if x <= 0:
            x = 0
        elif x >= w:
            x = w

        if y <= 0:
            y = 0
        elif y >= h:
            y = h

        return x, y
