import json
import uuid
from multiprocessing.connection import Client
from pathlib import Path
from typing import List, Tuple, Dict

import geojson
from label_studio_sdk import Project
from shapely import Polygon
from shapely.geometry import shape
from tqdm import tqdm

from zia.annotations.config.project_config import ResourcePaths
from zia.annotations.labelstudio.client import LabelStudioClient
from zia.annotations.labelstudio.polygon_labels import create_keypoint_result, create_polygon_result
from zia.log import get_logger

log = get_logger(__file__)


def load_polygons(path: Path) -> Tuple[List[str], List[Polygon]]:
    with open(path, "r") as f:
        fcol: geojson.FeatureCollection = geojson.load(f)

    labels, polygons = [], []

    for f in fcol["features"]:
        p = shape(f["geometry"])
        label = f["properties"]["label"]
        labels.append(label)
        polygons.append(p)

    return labels, polygons


def create_image_name_task_dict(tasks: dict) -> Dict[str, str]:
    image_names_task_dict = {}
    for task in tasks:
        image_name = Path(task["data"]["image"]).stem
        image_names_task_dict[image_name] = task["id"]

    return image_names_task_dict


def create_tasks_for_image(dataset_name: str,
                           resource_paths: ResourcePaths,
                           image_name_task_dict: Dict[str, str],
                           project: Project) -> None:
    tasks_to_create = []
    for image_path in resource_paths.image_path.iterdir():

        if image_path.stem in image_name_task_dict:
            continue

        tasks_to_create.append(
            {
                "data": {
                    "image": f"/data/local-files/?d={dataset_name}/image/{image_path.name}",
                    "pet": "rat"
                },
                "meta": {
                    "dataset": dataset_name,
                    "group": "steatosis" if not "control" in image_path.stem else "control"
                },
                "is_labeled": False,
                "overlap": 0
            }
        )

    if len(tasks_to_create) != 0:
        log.info(f"Creating tasks for {len(tasks_to_create)} images.")
        project.import_tasks(tasks=tasks_to_create)
        image_name_task_dict = create_image_name_task_dict(project.tasks)

    for image_path in tqdm(list(resource_paths.image_path.iterdir()), desc="updating task meta data", unit="tasks"):
        task_id = image_name_task_dict.get(image_path.stem)
        project.update_task(int(task_id),
                            meta={"dataset": dataset_name})


def get_dataset_storagepath_dict(client: Client, project_id: int) -> Dict[str, Path]:
    response = client.make_request(
        method="GET",
        url=f"/api/storages/localfiles?project={project_id}",
    )

    res_json = response.json()

    storage_dict = {}
    for storage in res_json:
        storage_path = Path(storage["path"])
        storage_dict[storage_path.parent.name] = storage_path
    return storage_dict


def prepare_and_upload(datasets: List[str]):
    client = LabelStudioClient.create_sdk_client()

    projects = client.get_projects()

    if len(projects) == 0:
        raise Exception("No projects found in labelstudio")

    project_id = projects[0].id

    label_studio_project: Project = client.get_project(project_id)

    image_name_task_dict = create_image_name_task_dict(label_studio_project.tasks)

    # create tasks for all images if they do not exist.
    for dataset_name in datasets:
        resource_paths = ResourcePaths(dataset_name)
        create_tasks_for_image(dataset_name,
                               resource_paths,
                               image_name_task_dict,
                               label_studio_project)

    data_config_dict = {}
    resource_paths_dict = {}

    for dataset_name in datasets:
        resource_paths = ResourcePaths(dataset_name)
        resource_paths_dict[dataset_name] = resource_paths

        with open(resource_paths.base_path / "data_config.json", "r") as f:
            data_config_dict[dataset_name] = json.load(f)

    for task in tqdm(label_studio_project.tasks, desc="creating and uploading predictions", unit="tasks"):

        image_id = Path(task["data"]["image"]).stem
        dataset_name = task["meta"]["dataset"]

        if dataset_name not in datasets:
            continue

        h = data_config_dict[dataset_name]["config"]["height"]
        w = data_config_dict[dataset_name]["config"]["width"]

        polygon_path = resource_paths_dict[dataset_name].polygon_path / f"{image_id}.geojson"

        labels, polygons = load_polygons(polygon_path)

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
