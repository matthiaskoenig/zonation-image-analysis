import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Tuple, Union

import geojson as gj
import shapely.affinity
from shapely import MultiPolygon, Polygon
from shapely.geometry import shape
from strenum import StrEnum

from zia.annotations.annotation.geometry_utils import rescale_coords


PATH_TO_FILE = (
    "..\J-12-00350_NOR-022_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006.geojson"
)

logger = logging.getLogger(__name__)


class AnnotationType(StrEnum):
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
    BLUR = "blur"

    UNKNOWN = "-"

    @classmethod
    def get_by_string(cls, string: str):
        return getattr(cls, string.upper())

    @classmethod
    def get_artifacts(cls) -> Set["AnnotationType"]:
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
        self, factor, offset=(0, 0)
    ) -> Optional[Union[Polygon, MultiPolygon]]:
        if isinstance(self.geometry, Polygon):
            return Polygon(
                Annotation._rescale_coords(
                    self.geometry.exterior.coords, factor, offset
                )
            )
        if isinstance(self.geometry, MultiPolygon):
            return MultiPolygon(
                [
                    Polygon(
                        Annotation._rescale_coords(poly.exterior.coords, factor, offset)
                    )
                    for poly in self.geometry.geoms
                ]
            )

        logger.warning(
            f"Another geometry type encountered, "
            f"which was not drawn: '{type(self.geometry)}'"
        )

    @classmethod
    def _rescale_coords(
        cls, coords, level: int, offset: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        return rescale_coords(coords, 1 / level, offset)


"""
Parses a geojson file and returns a list of Annotations, which contains
a shapely geometry and the annotation type (annotation class assigned in QuPath)
"""


class AnnotationParser:
    @classmethod
    def parse_geojson(cls, file_name: str) -> List[Annotation]:
        with open(file_name) as f:
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
        features: List[Union[Polygon | MultiPolygon]],
        annotation_type: AnnotationType,
    ) -> List[Annotation]:
        return list(filter(lambda x: x.annotation_class == annotation_type, features))

    @classmethod
    def get_annotation_by_types(
        cls,
        features: List[Union[Polygon | MultiPolygon]],
        annotation_types: Set[AnnotationType],
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
    shapes = AnnotationParser.parse_geojson(PATH_TO_FILE)
    liver_annotations = AnnotationParser.get_annotation_by_type(
        shapes, AnnotationType.LIVER
    )