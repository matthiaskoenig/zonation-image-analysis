import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from label_studio_sdk import Client
from label_studio_sdk.data_manager import Filters, Column

from zia.annotations.config.project_config import read_labelstudio_config
import requests

from zia.annotations.labelstudio.json_model import Prediction, Annotation, Task


class LabelStudioClient:

    @classmethod
    def create_sdk_client(cls, production=False) -> Client:
        config = read_labelstudio_config("Production" if production else "Test")
        return Client(url=config.base_url, api_key=config.api_token)

    def __init__(self, client: Client):
        self.client = client

    def _create_header(self) -> Dict[str, str]:
        return {
            "Authorization": "Token " + self.api_token,
            "Content-Type": "application/json"
        }

    def create_prediction(self, payload: Prediction):
        response = requests.post(
            url=self.base_url + "/predictions/",
            headers=self._create_header(),
            data=json.dumps(asdict(payload))
        )

        if response.status_code == 201:
            return response.json()["id"]

        raise ConnectionError(response.text)

    def create_annotation(self, payload: Annotation, task_id: int):
        response = requests.post(
            url=self.base_url + f"/tasks/{task_id}/annotations/",
            headers=self._create_header(),
            data=payload
        )

        if response.status_code == 201:
            return response.json()["id"]

        raise ConnectionError(response.text)

    def get_image_task_dict(self, project: int) -> Dict[str, int]:

        project = self.client.get_project(project)

        image_task_id_dict = {}
        for task in project.tasks:
            image_name = Path(task["data"]["image"]).stem
            image_task_id_dict[image_name] = task["id"]

        return image_task_id_dict


if __name__ == "__main__":
    client = LabelStudioClient(LabelStudioClient.create_sdk_client())

    result = client.get_image_task_dict(1)

