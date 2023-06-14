import os
from typing import Optional

from zia import DATA_PATH, ZARR_PATH, RESULTS_PATH


class FileManager:
    data_path = DATA_PATH
    zarr_path = ZARR_PATH
    results_path = RESULTS_PATH

    def __init__(self):
        self.initialize()

    def intitialize(self):
        for path in [self.results_path, self.zarr_path]:
            if not os.path.exists(path):
                os.mkdir(path)

    def get_annotation_path(self, image_name: str) -> Optional[str]:
        json_file = image_name + ".geojson"
        base = self.data_path / "annotations_species_comparison"
        for sub_folder in os.listdir(base):
            image_path = os.path.join(base, sub_folder, "objectsjson", json_file)
            if os.path.isfile(image_path):
                return image_path

    def get_image_path(self, image_name: str) -> Optional[str]:
        image_file = image_name + ".ndpi"
        base = self.data_path / "cyp_species_comparison" / "all"
        for species_folder in os.listdir(base):
            species_path = os.path.join(base, species_folder)
            for cyp_folder in os.listdir(species_path):
                image_path = os.path.join(species_path, cyp_folder, image_file)
                if os.path.isfile(image_path):
                    return image_path

    def get_roi_geojson_paths(self, image_name: str) -> str:
        json_file = image_name + ".geojson"
        base = self.results_path / "annotations_liver_roi"
        for species_folder in os.listdir(base):
            json_path = os.path.join(base, species_folder, json_file)
            if os.path.isfile(json_path):
                return json_path

    def get_zarr_file(self, image_name: str):
        return ZARR_PATH / f"{image_name}.zarr"
