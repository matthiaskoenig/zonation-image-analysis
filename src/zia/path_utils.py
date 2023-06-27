"""Module handling project paths."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator

from zia.config import Configuration
from zia.console import console


class ResultsDirectories(str, Enum):
    """Reused directories."""

    ANNOTATIONS_LIVER_ROI = "annotations_liver_roi"
    LIVER_MASK = "liver_mask"
    MASKED_PNG_IMAGES = "masked_png_images"


class FileManager:
    """Class for managing set of images."""

    def __init__(self, configuration: Configuration):
        """Initialize the file manager."""
        self.data_path: Path = configuration.data_path
        self.zarr_path: Path = configuration.zarr_path
        self.annotation_path: Path = configuration.annotations_path
        self.report_path: Path = configuration.reports_path
        self.results_path: Path = configuration.results_path

    def image_paths(self, extension="ndpi") -> Iterator[Path]:
        """Get list of images for given extension."""

        return self.data_path.glob(f"**/*.{extension}")

    def get_zarr_path(self, image: Path) -> Path:
        """Get Zarr Path for image."""
        return self.zarr_path / f"{image.stem}.zarr"

    def get_geojson_path(self, image: Path) -> Path:
        """Get geojson annotation path for image."""
        # resolve relative path
        directory = self.annotation_path / image.relative_to(self.data_path).parent
        return directory / f"{image.stem}.geojson"

    #
    # def get_roi_geojson_paths(self, image_name: str) -> str:
    #     json_file = image_name + ".geojson"
    #     base = self.results_path / ResultDir.ANNOTATIONS_LIVER_ROI.value
    #     for species_folder in os.listdir(base):
    #         json_path = os.path.join(base, species_folder, json_file)
    #         if os.path.isfile(json_path):
    #             return json_path

    def info(self):
        console.rule(style="white")
        image_paths = list(self.image_paths())
        console.print(f"FileManage: {len(image_paths)} images")
        console.rule(style="white")

        for k, p in enumerate(image_paths):
            zarr_path = self.get_zarr_path(p).relative_to(self.zarr_path)
            geojson_path = self.get_geojson_path(p).relative_to(self.annotation_path)
            # console.print(f"{p.relative_to(self.data_path)} | {zarr_path} | {geojson_path}")
            console.print(f"[{k}] {self.image_metadata(p)}")
        console.rule(style="white")

    def image_metadata(self, image: Path) -> ImageMetadata:
        """Metadata for image"""
        return ImageMetadata(
            image_id=image.stem,
            protein=image.parent.name,
            species=image.parent.parent.name,
            negative="negative" in image.stem.lower(),
        )


@dataclass
class ImageMetadata:
    """Metadata resolved from path information."""

    species: str
    protein: str
    image_id: str
    negative: bool


if __name__ == "__main__":
    from zia import BASE_PATH
    from zia.config import read_config

    configuration = read_config(BASE_PATH / "configuration.ini")
    file_manager = FileManager(configuration)
    file_manager.info()
