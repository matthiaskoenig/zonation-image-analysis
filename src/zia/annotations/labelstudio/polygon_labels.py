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
            width=0.2,
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
        type="keypoint"
    ))


def create_polygon_keypoint_annotation(polygons: List[Polygon], labels: List[str], h: int, w: int) -> Annotation:
    results = [create_keypoint_result(str(i), polygon, label, h, w) for i, (polygon, label) in enumerate(zip(polygons, labels))]

    return Annotation(
        unique_id=str(uuid.uuid4()),
        result=results,
        was_cancelled=False,
        ground_truth=False,
        completed_by=1
    )


def create_polygon_keypoint_prediction(task_id: int, polygons: List[Polygon], labels: List[str], h: int, w: int) -> Prediction:
    results = [create_keypoint_result(str(i), polygon, label, h, w) for i, (polygon, label) in enumerate(zip(polygons, labels))]

    return Prediction(
        task=task_id,
        result=results,
        model_version="initial-clustering",
        score=0.5
    )


if __name__ == "__main__":

    with open(r"C:\Users\jonas\Development\git\zonation-image-analysis\sample_data\polygons\FLR-168_0_sample_0.geojson", "r") as f:
        fcol: geojson.FeatureCollection = geojson.load(f)

    labels, polygons = [], []

    for f in fcol["features"]:
        p = shape(f["geometry"])
        label = f["properties"]["label"]

        if str(label) == "1":
            labels.append("makrosteatosis")
            polygons.append(p)

    data_item = create_polygon_keypoint_annotation(polygons, labels, 2000, 2000)

    with open("sample_dict.json", "w+") as f:
        json.dump(asdict(data_item), f)
