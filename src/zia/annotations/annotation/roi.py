import json
from dataclasses import dataclass
from enum import IntEnum
from typing import List

import geojson
from shapely import Polygon
import geojson as gj

from zia.annotations.annotation.annotations import AnnotationType
from zia.annotations.annotation.geometry_utils import rescale_coords


class PyramidalLevel(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7

    @classmethod
    def get_by_numeric_level(cls, level: int):
        return getattr(cls, level)


class Roi:
    def __init__(self,
                 polygon: Polygon,
                 level: PyramidalLevel,
                 annotation_type: AnnotationType):
        self._geometry = polygon
        self._level = level
        self.annotation_type = annotation_type

    def get_polygon_for_level(self, level: PyramidalLevel) -> Polygon:
        factor = 2 ** (level - self._level)
        return Polygon(rescale_coords(self._geometry.exterior.coords, factor))

    def _to_geojson_feature(self) -> gj.Feature:
        polygon = gj.Polygon(self._geometry.exterior.coords)
        properties = {
            "level": self._level,
            "annotationType": self.annotation_type
        }

        return gj.Feature(geometry=polygon, properties=properties)

    @classmethod
    def write_to_geojson(cls, rois: List["Roi"], path: str):
        features = [roi._to_geojson_feature() for roi in rois]
        feature_collection = gj.FeatureCollection(features)

        with open(path, "w") as f:
            json.dump(feature_collection, f)

    @classmethod
    def load_from_file(cls, path: str) -> "Roi":
        with open(path, "r") as f:
            geojson_dict = gj.load(path)

        if not isinstance(geojson_dict.get("geometry"), gj.Polygon):
            raise ImportError("The parsed geojson geometry for a ROI must be a Polygon")

        properties: dict = geojson_dict.get("properties")
        if "level" not in properties.keys():
            raise KeyError(
                "The geojson object must contain a the element 'properties.level'")
        if "annotationType" not in properties.keys():
            raise KeyError(
                "The geojson object must contain a the element 'properties.level'")

        geometry = geojson_dict.get("geometry")
        level = PyramidalLevel.get_by_numeric_level(properties.get("level"))
        annotation_type = AnnotationType.get_by_string(properties.get("annotationType"))

        return Roi(geometry, level, annotation_type)
