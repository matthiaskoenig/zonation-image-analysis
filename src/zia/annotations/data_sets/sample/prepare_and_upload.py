import json
import uuid

import geojson
from shapely.geometry import shape
from tqdm import tqdm

from zia.annotations.config.project_config import ResourcePaths
from zia.annotations.labelstudio.client import LabelStudioClient
from zia.annotations.labelstudio.json_model import Prediction
from zia.annotations.labelstudio.polygon_labels import create_keypoint_result, create_polygon_result


def prepare_and_upload():
    resource_paths = ResourcePaths("sample_data")
    client = LabelStudioClient()

    with open(resource_paths.base_path / "sample_data.json", "r") as f:
        sample_data = json.load(f)

    h = sample_data["config"]["height"]
    w = sample_data["config"]["width"]

    image_task_id_dict = client.get_task_list(view=0, project=3, resolve_uri=False)

    for image, task_id in tqdm(image_task_id_dict.items(), desc="creating and uploading predictions", unit="task"):
        image_path = resource_paths.image_path / image

        image_id = image_path.stem

        polygon_path = resource_paths.polygon_path / f"{image_id}.geojson"

        with open(polygon_path, "r") as f:
            fcol: geojson.FeatureCollection = geojson.load(f)

        labels, polygons = [], []

        for f in fcol["features"]:
            p = shape(f["geometry"])
            label = f["properties"]["label"]

            if str(label) == "0":
                labels.append("makrosteatosis")
                polygons.append(p)

        keypoint_results = [create_keypoint_result(str(uuid.uuid4()), polygon, label, h, w) for label, polygon in zip(labels, polygons)]
        polygon_result = [create_polygon_result(str(uuid.uuid4()), polygon, label, h, w) for label, polygon in zip(labels, polygons)]

        prediction = Prediction(
            task=task_id,
            model_version="init_cluster",
            result=keypoint_results + polygon_result,
            mislabeling=0,
            project=3
        )

        client.create_prediction(prediction)


if __name__ == "__main__":
    prepare_and_upload()
