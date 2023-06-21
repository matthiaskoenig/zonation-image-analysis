import json
from typing import List, Tuple

import geojson as gj
from shapely import Polygon
from shapely.geometry import shape

from zia.annotations.annotation.annotations import AnnotationType
from zia.annotations.annotation.geometry_utils import rescale_coords
from zia.annotations.annotation.util import PyramidalLevel


class Roi:
    def __init__(
        self, polygon: Polygon, level: PyramidalLevel, annotation_type: AnnotationType
    ):
        self._geometry = polygon
        self._level = level
        self.annotation_type = annotation_type

    def get_polygon_for_level(self, level: PyramidalLevel, offset=(0, 0)) -> Polygon:
        factor = 2 ** (self._level - level)
        offset = tuple(x / factor for x in offset)
        return Polygon(rescale_coords(self._geometry.exterior.coords, factor, offset))

    def _to_geojson_feature(self) -> gj.Feature:
        geojson_dict = self._geometry.__geo_interface__
        polygon = gj.Polygon(coordinates=geojson_dict["coordinates"])
        properties = {"level": self._level, "annotationType": self.annotation_type}

        return gj.Feature(geometry=polygon, properties=properties)

    @classmethod
    def write_to_geojson(cls, rois: List["Roi"], path: str):
        features = [roi._to_geojson_feature() for roi in rois]
        feature_collection = gj.FeatureCollection(features)

        with open(path, "w") as f:
            json.dump(feature_collection, f)

    @classmethod
    def load_from_file(cls, path: str) -> List["Roi"]:
        with open(path, "r") as f:
            feature_collection = gj.load(f)

        return [
            Roi._parse_feature(feature) for feature in feature_collection["features"]
        ]

    @classmethod
    def _parse_feature(cls, feature: dict) -> "Roi":
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

        return Roi(geometry, level, annotation_type)

    def get_bound(self, level: PyramidalLevel) -> Tuple[slice, slice]:
        """
        returns slice for tuple of from ((slice(min_x, max_x), slice(min_y, max_y))
        """
        poly = self.get_polygon_for_level(level)
        bounds = poly.bounds
        # bounds where generated from padded image. setting negative bounds to zero to have valid slices
        norm_bounds = tuple([b if b >= 0 else 0 for b in bounds])
        xs = slice(int(norm_bounds[0]), int(norm_bounds[2]))
        ys = slice(int(norm_bounds[1]), int(norm_bounds[3]))
        return xs, ys
