import os
from enum import Enum
from typing import Iterator, Optional, Tuple

from zia import DATA_PATH, REPORT_PATH, RESULTS_PATH, ZARR_PATH


class ResultDir(Enum):
    ANNOTATIONS_LIVER_ROI = "annotations_liver_roi"
    LIVER_MASK = "liver_mask"

    @classmethod
    def values(cls) -> Iterator["ResultDir"]:
        for const in ResultDir.__members__.values():
            yield const


class FileManager:
    data_path = DATA_PATH
    zarr_path = ZARR_PATH
    report_path = REPORT_PATH
    results_path = RESULTS_PATH

    def __init__(self):
        self.initialize()

    def get_image_names(self) -> Iterator[Tuple[str, str]]:
        """
        returns an iterator of Tuples all the image files in the data directory
        with their respective species. The image name is returned without
        the extension.
        """
        base = self.data_path / "cyp_species_comparison" / "control"
        for species_folder in os.listdir(base):
            species_path = os.path.join(base, species_folder)
            for cyp_folder in os.listdir(species_path):
                cyp_path = os.path.join(species_path, cyp_folder)
                for image_file in os.listdir(cyp_path):
                    yield species_folder, os.path.splitext(image_file)[0]

    def get_annotation_path(self, image_name: str) -> Optional[str]:
        json_file = image_name + ".geojson"
        base = self.data_path / "annotations_species_comparison"
        for sub_folder in os.listdir(base):
            image_path = os.path.join(base, sub_folder, "objectsjson", json_file)
            if os.path.isfile(image_path):
                return image_path
        return None

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
        base = self.results_path / ResultDir.ANNOTATIONS_LIVER_ROI.value
        for species_folder in os.listdir(base):
            json_path = os.path.join(base, species_folder, json_file)
            if os.path.isfile(json_path):
                return json_path

    def get_zarr_file(self, image_name: str):
        return ZARR_PATH / f"{image_name}.zarr"

    def get_results_path(self, result_dir: ResultDir, species: str, file_name: str):
        report_folder = os.path.join(self.results_path, result_dir.value)
        if not os.path.exists(report_folder):
            os.mkdir(report_folder)

        species_folder = os.path.join(report_folder, species)
        if not os.path.exists(species_folder):
            os.mkdir(species_folder)

        return os.path.join(species_folder, file_name)

    def get_report_path(self, report_dir: ResultDir, species, file_name: str):
        report_folder = os.path.join(self.report_path, report_dir.value)
        if not os.path.exists(report_folder):
            os.mkdir(report_folder)

        species_folder = os.path.join(report_folder, species)
        if not os.path.exists(species_folder):
            os.mkdir(species_folder)

        return os.path.join(species_folder, file_name)

    def initialize(self):
        for path in [self.results_path, self.zarr_path]:
            if not os.path.exists(path):
                os.mkdir(path)
