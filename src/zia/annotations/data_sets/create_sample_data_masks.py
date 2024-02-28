import cv2
import numpy as np
from label_studio_sdk import Project

from zia.annotations.config.project_config import ResourcePaths
from zia.annotations.labelstudio.client import LabelStudioClient
from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic

if __name__ == '__main__':
    client = LabelStudioClient.create_sdk_client()
    resource_paths = ResourcePaths("sample_data")
    project: Project = list(filter(lambda p: p.params["title"].lower() == "macrosteatosis", client.get_projects()))[0]

    for task in project.get_tasks():
        if len(task["annotations"]) == 0:
            continue

        #print(task)

        image = task["data"]["subject"] + "_" + task["data"]["sample"] + ".png"

        annotation = task["annotations"][0]

        result = annotation["result"]

        polygons = list(filter(lambda r: r["type"] == "polygon", result))

        print(image, len(polygons))

        template = np.zeros(shape=(1024, 1024)).astype(np.uint8)

        for polygon in polygons:

            points = np.array(polygon["value"]["points"])

            image_coords = points / 100 * 1024

            cv2.fillPoly(template, np.expand_dims(image_coords, 0).astype(np.int32), color=(255,))

        cv2.imwrite(str(resource_paths.mask_path / image), template)
        # plot_pic(template)
