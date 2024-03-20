from pathlib import Path
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
from label_studio_sdk import Project
from matplotlib import pyplot as plt
from tqdm import tqdm

from zia.annotations.config.project_config import ResourcePaths
from zia.annotations.labelstudio.client import LabelStudioClient

dataset_paths = ResourcePaths("sample_data")

target_path = Path(r"/media/jkuettner/Extreme Pro/exchange/training_data/sampledata")


def transform_to_image_key_point(keypoint: Dict[str, Any]) -> Tuple[int, int]:
    x = keypoint["value"]["x"]
    y = keypoint["value"]["y"]

    ori_h = keypoint["original_height"]
    ori_w = keypoint["original_width"]

    new_x = round(x / 100 * ori_w)
    new_y = round(y / 100 * ori_h)

    return new_x, new_y


def create_mask(image: np.ndarray, image_key_points: List[Tuple[int, int]]) -> np.ndarray:
    gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gs, 200, 255, cv2.THRESH_BINARY)

    for _ in range(2):
        th = cv2.GaussianBlur(th, (31, 31), 0)
        _, th = cv2.threshold(th, 255 // 2, 255, cv2.THRESH_BINARY)

    # create markers arraym -> key points with colors 2,... n+2, unknown: 0, background: 1
    markers = np.zeros_like(th, dtype=np.int32)

    markers[th == 0] = 1

    for i, keypoint in enumerate(image_key_points):
        cv2.circle(markers, keypoint, 5, (i + 2,), cv2.FILLED)

    markers = cv2.watershed(image, markers)

    mask = np.zeros_like(gs, dtype=np.uint8)
    mask[markers > 1] = 255

    for _ in range(1):
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        _, mask = cv2.threshold(mask, 255 // 2, 255, cv2.THRESH_BINARY)

    return mask


def get_image_name(task: Dict[str, Any]) -> str:
    return task["data"]["subject"] + "_" + task["data"]["tile"] + ".png"


if __name__ == '__main__':

    for p0 in ["test", "train", "val"]:
        for p1 in ["image", "mask"]:
            p = target_path / p0 / p1
            p.mkdir(parents=True, exist_ok=True)

    client = LabelStudioClient.create_sdk_client(production=True)
    resource_paths = ResourcePaths("sample_data")
    project: Project = list(filter(lambda p: p.params["title"].lower() == "macrosteatosis", client.get_projects()))[0]

    from sklearn.model_selection import train_test_split

    train_tasks, test_tasks = train_test_split(project.get_tasks(), test_size=0.2, shuffle=True, random_state=42)

    val_tasks, test_tasks = train_test_split(test_tasks, test_size=0.5, shuffle=True, random_state=42)

    for tasks, use in zip([train_tasks, val_tasks, test_tasks], ["train", "val", "test"]):
        for task in tqdm(tasks, desc=f"Creating masks from tasks for {use} data", unit="tasks"):
            if len(task["annotations"]) == 0:
                continue

            image_name = task["data"]["subject"] + "_" + task["data"]["tile"] + ".png"

            annotation = task["annotations"][0]

            result = annotation["result"]

            key_points = list(filter(lambda r: r["type"] in ["keypoint", "keypointlabels"], result))

            image_key_points = [transform_to_image_key_point(key_point) for key_point in key_points]

            image = cv2.imread(str(resource_paths.image_path / image_name))
            mask = create_mask(image, image_key_points)

            image = cv2.pyrDown(image)
            mask = cv2.resize(mask, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(target_path / use / "image" / image_name), image)
            cv2.imwrite(str(target_path / use / "mask" / image_name), mask)
