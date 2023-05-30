from dataclasses import dataclass
from strenum import StrEnum
from typing import Union, List, Tuple

from shapely import Polygon, MultiPolygon
import geojson as gj
from shapely.geometry import shape

PATH_TO_FILE = "..\J-12-00350_NOR-022_Lewis_CYP2E1- 1 300_Run 14_ML, Sb, rk_MAA_006.geojson"


class AnnotationType(StrEnum):
    LUNG = "Lung"
    KIDNEY = "Kidney"
    LIVER = "Liver"
    BUBBLE = "Bubble"
    FOLD = "Fold"
    VESSEL = "Vessel"

    @classmethod
    def get_by_string(cls, string: str):
        return getattr(cls, string.upper())


@dataclass
class Annotation:
    geometry: Union[Polygon, MultiPolygon]
    annotation_class: AnnotationType

    """
    downsizes the geometry of the annotation by the level. The level should
    correspond to the level of the pyramidal image.
    """
    def get_resized_geometry(self, factor) -> Union[Polygon, MultiPolygon]:
        if isinstance(self.geometry, Polygon):
                return Polygon(Annotation._rescale_coords(self.geometry.exterior.coords, factor))
        if isinstance(self.geometry, MultiPolygon):
            return MultiPolygon([Polygon(Annotation._rescale_coords(poly.coords, factor))] for poly in self.geoms)

    @classmethod
    def _rescale_coords(cls, coords, level: int) -> List[Tuple[float, float]]:
        return [(x / level, y / level) for x, y in
                coords]


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
        name = feature.get("properties").get("classification").get("name")
        return AnnotationType.get_by_string(name)

    @classmethod
    def _get_anno_geometry_from_feature(cls, feature: dict) -> Union[
        Polygon, MultiPolygon]:
        return shape(feature["geometry"])

    @classmethod
    def _create_annotations(cls, features: List[dict]) -> List[Annotation]:
        return [AnnotationParser._create_annotation(feature) for feature in features]

    @classmethod
    def _create_annotation(cls, feature: dict):
        return Annotation(AnnotationParser._get_anno_geometry_from_feature(feature),
                          AnnotationParser._get_anno_type_from_feature(feature))

    @classmethod
    def get_annotation_by_type(cls, features: List[Union[Polygon | MultiPolygon]],
                               annotation_type: AnnotationType):
        return list(filter(lambda x: x.annotation_class == annotation_type, features))


if __name__ == "__main__":
    shapes = AnnotationParser.parse_geojson(PATH_TO_FILE)
    liver_annotations = AnnotationParser.get_annotation_by_type(shapes,
                                                                AnnotationType.LIVER)
