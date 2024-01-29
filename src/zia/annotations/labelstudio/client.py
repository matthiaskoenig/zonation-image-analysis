import json
from dataclasses import asdict
from typing import Dict, List

from zia.annotations.config.project_config import read_labelstudio_config
import requests

from zia.annotations.labelstudio.json_model import Prediction, Annotation, Task


class LabelStudioClient:

    def __init__(self, production=False):
        config = read_labelstudio_config("Production" if production else "Test")
        self.api_token = config.api_token
        self.base_url = config.base_url

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

    def get_task_list(self, view: int, project: int, resolve_uri: bool) -> Dict[str, int]:
        response = requests.get(
            url=self.base_url + "/tasks/",
            params=dict(view=view, project=project, resolve_uri=resolve_uri),
            headers=self._create_header()
        )

        if response.status_code == 200:
            data = response.json()

            task_data = {}

            for entry in data["tasks"]:
                image = "-".join(entry["data"]["img"].split("-")[1:])
                task_data[image] = entry["id"]

            return task_data

        raise ConnectionError(response.text)


if __name__ == "__main__":
    client = LabelStudioClient()
    result = client.get_task_list(0, 3, False)
