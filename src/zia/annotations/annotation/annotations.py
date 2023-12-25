"""Annotations in whole-slide image.

Annotations have been generated with QuPath and stored in geojson.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set, Union

import geojson as gj
from shapely import LineString, MultiPolygon, Polygon
from shapely.geometry import shape

from zia.annotations.annotation.geometry_utils import rescale_coords
from zia.annotations.annotation.util import PyramidalLevel


logger = logging.getLogger(__name__)


class AnnotationType(str, Enum):
    """Types of annotations."""

    LUNG = "lung"
    KIDNEY = "kidney"
    LIVER = "liver"
    BUBBLE = "bubble"
    FOLD = "fold"
    VESSEL = "vessel"
    DARK = "dark"
    LIGHT = "light"
    SCRATCH = "scratch"
    SHADOW = "shadow"
    OTHER = "other"
    TEAR = "tear"
    BLUR = "blur",
    LOBE_0 = "lobe_0",
    LOBE_1 = "lobe_1",
    LOBE_2 = "lobe_2",
    LOBE_3 = "lobe_3"

    UNKNOWN = "-"

    @classmethod
    def get_lobe_annnotations(cls) -> List[AnnotationType]:
        return [cls.LOBE_0, cls.LOBE_1, cls.LOBE_2, cls.LOBE_3]

    @classmethod
    def get_by_string(cls, string: str):
        return getattr(cls, string.upper())

    @classmethod
    def get_artifacts(cls) -> Set[AnnotationType]:
        return {
            cls.BUBBLE,
            cls.FOLD,
            cls.DARK,
            cls.LIGHT,
            cls.SCRATCH,
            cls.SHADOW,
            cls.OTHER,
            cls.TEAR,
            cls.BLUR,
        }


@dataclass
class Annotation:
    geometry: Union[Polygon, MultiPolygon]
    annotation_class: AnnotationType

    """
    downsizes the geometry of the annotation by the level. The level should
    correspond to the level of the pyramidal image.
    """

    def get_resized_geometry(
        self, level: PyramidalLevel, offset=(0, 0)
    ) -> Optional[Union[Polygon, MultiPolygon | LineString]]:
        factor = 1 / 2**level
        if isinstance(self.geometry, Polygon):
            return Polygon(
                rescale_coords(self.geometry.exterior.coords, factor, offset)
            )
        if isinstance(self.geometry, MultiPolygon):
            return MultiPolygon(
                [
                    Polygon(rescale_coords(poly.exterior.coords, factor, offset))
                    for poly in self.geometry.geoms
                ]
            )
        if isinstance(self.geometry, LineString):
            return LineString(rescale_coords(self.geometry.coords, factor, offset))

        logger.warning(f"Another geometry type encountered: '{type(self.geometry)}'")


"""
Parses a geojson file and returns a list of Annotations, which contains
a shapely geometry and the annotation type (annotation class assigned in QuPath)
"""


class AnnotationParser:
    @classmethod
    def parse_geojson(cls, path: Path) -> List[Annotation]:
        with open(path) as f:
            data = gj.load(f)

        features = data["features"]

        if len(features) == 0:
            raise ImportError("The feature collection is empty")

        return AnnotationParser._create_annotations(features)

    @classmethod
    def _get_anno_type_from_feature(cls, feature: dict) -> AnnotationType:
        classification = feature.get("properties").get("classification")
        if classification is None:
            return AnnotationType.UNKNOWN
        return AnnotationType.get_by_string(classification.get("name"))

    @classmethod
    def _get_anno_geometry_from_feature(
        cls, feature: dict
    ) -> Union[Polygon, MultiPolygon]:
        return shape(feature["geometry"])

    @classmethod
    def _create_annotations(cls, features: List[dict]) -> List[Annotation]:
        return [AnnotationParser._create_annotation(feature) for feature in features]

    @classmethod
    def _create_annotation(cls, feature: dict) -> Annotation:
        return Annotation(
            AnnotationParser._get_anno_geometry_from_feature(feature),
            AnnotationParser._get_anno_type_from_feature(feature),
        )

    @classmethod
    def get_annotation_by_type(
        cls,
        features: List[Union[Polygon | MultiPolygon | LineString]],
        annotation_type: AnnotationType,
    ) -> List[Annotation]:
        return list(filter(lambda x: x.annotation_class == annotation_type, features))

    @classmethod
    def get_annotation_by_types(
        cls,
        features: List[Union[Polygon | MultiPolygon | LineString]],
        annotation_types: List[AnnotationType],
    ) -> List[Annotation]:
        nested_list = [
            AnnotationParser.get_annotation_by_type(features, anno_type)
            for anno_type in annotation_types
        ]
        return [anno for sub_list in nested_list for anno in sub_list]


class AnnotationClassMissingException(Exception):
    def __init__(self, feature_id: str):
        self.feature_id = feature_id
        super().__init__(
            f"Annotation classification is missing for feature with id '{feature_id}'."
        )


if __name__ == "__main__":
    PATH_TO_FILE = (
        "..\J-12-00350_NOR-022_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006.geojson"
    )

    shapes = AnnotationParser.parse_geojson(PATH_TO_FILE)
    liver_annotations = AnnotationParser.get_annotation_by_type(
        shapes, AnnotationType.LIVER
    )
