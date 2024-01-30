import json
import uuid
from pathlib import Path
from typing import List

import geojson
from label_studio_sdk import Project
from shapely.geometry import shape
from tqdm import tqdm

from zia.annotations.config.project_config import ResourcePaths
from zia.annotations.labelstudio.client import LabelStudioClient
from zia.annotations.labelstudio.polygon_labels import create_keypoint_result, create_polygon_result
from zia.log import get_logger

log = get_logger(__file__)


def prepare_and_upload(datasets: List[str]):
    client = LabelStudioClient.create_sdk_client()

    projects = client.get_projects()

    if len(projects) == 0:
        raise Exception("No projects found in labelstudio")

    label_studio_project: Project = projects[0]

    data_config_dict = {}
    resource_paths_dict = {}

    for dataset in datasets:
        resource_paths = ResourcePaths(dataset)
        resource_paths_dict[dataset] = resource_paths

        with open(resource_paths.base_path / "data_config.json", "r") as f:
            data_config_dict[dataset] = json.load(f)

    for task in tqdm(label_studio_project.tasks, desc="creating and uploading predictions", unit="tasks"):

        data_path = Path(task["storage_filename"])

        if not data_path.exists():
            raise FileNotFoundError(f"The file {data_path} does not exist")

        dataset_name = data_path.parent.parent.name

        if dataset_name not in datasets:
            continue

        h = data_config_dict[dataset_name]["config"]["height"]
        w = data_config_dict[dataset_name]["config"]["width"]

        image_id = data_path.stem

        polygon_path = resource_paths_dict[dataset_name].polygon_path / f"{image_id}.geojson"

        with open(polygon_path, "r") as f:
            fcol: geojson.FeatureCollection = geojson.load(f)

        labels, polygons = [], []

        for f in fcol["features"]:
            p = shape(f["geometry"])
            label = f["properties"]["label"]
            labels.append(label)
            polygons.append(p)

        keypoint_results = [create_keypoint_result(str(uuid.uuid4()), polygon, label, h, w) for label, polygon in zip(labels, polygons)]
        polygon_result = [create_polygon_result(str(uuid.uuid4()), polygon, label, h, w) for label, polygon in zip(labels, polygons)]

        # delete existing predictions
        if len(task["predictions"]) != 0:
            for prediction in task["predictions"]:
                prediction_id = prediction["id"]
                client.make_request("DELETE", f"/api/predictions/{prediction_id}")

        label_studio_project.create_prediction(task_id=task["id"], result=polygon_result + keypoint_results, model_version="initial_import")


if __name__ == "__main__":
    prepare_and_upload(["sample_data", "jena_data"])
