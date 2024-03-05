import json
import uuid
from dataclasses import asdict
from typing import List, Optional, Dict

import geojson
from shapely import Polygon
from shapely.geometry import shape

from zia.annotations.labelstudio.json_model import DataItem, Prediction, Result, KeypointValue, Annotation, PolygonValue


def create_keypoint_result(id: Optional[str], polygon: Polygon, label: str, h: int, w: int) -> Dict:
    return asdict(Result(
        original_width=w,
        original_height=h,
        image_rotation=0,
        value=KeypointValue(
            x=polygon.centroid.x / w * 100,
            y=polygon.centroid.y / h * 100,
            width=1,
            keypointlabels=[label]
        ),
        id=id,
        from_name="keypoint",
        to_name="image",
        type="keypoint"
    ))


def create_polygon_result(id: Optional[str], polygon: Polygon, label: str, h: int, w: int) -> Dict:
    return asdict(Result(
        original_width=w,
        original_height=h,
        image_rotation=0,
        value=PolygonValue(
            points=[(x / w * 100, y / h * 100) for x, y in polygon.exterior.coords],
            polygonlabels=[label]
        ),
        id=id,
        from_name="polygon",
        to_name="image",
        type="polygon"
    ))


