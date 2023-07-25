"""Module handling project paths."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, List, Any
import re

from zia.config import Configuration
from zia.console import console


class ResultsDirectories(str, Enum):
    """Reused directories."""

    ANNOTATIONS_LIVER_ROI = "annotations_liver_roi"
    LIVER_MASK = "liver_mask"
    MASKED_PNG_IMAGES = "masked_png_images"


@dataclass
class ImageInfo:
    path: Path
    zarr_path: Path
    annotations_path: Path
    metadata: ImageMetadata
    roi_path: Optional[Path]


class FileManager:
    """Class for managing set of images."""

    def __init__(self, configuration: Configuration, filter: Optional[Callable] = None):
        """Initialize the file manager."""
        self._configuration: Configuration = configuration
        self.data_path: Path = configuration.data_path
        self.zarr_path: Path = configuration.zarr_path
        self.annotation_path: Path = configuration.annotations_path
        self.report_path: Path = configuration.reports_path
        self.results_path: Path = configuration.results_path

        self.filter: Optional[Callable] = filter

    def get_images(self) -> List[ImageInfo]:
        """Get image information."""
        images: List[ImageInfo] = []
        for p in self.image_paths():
            images.append(
                ImageInfo(
                    path=p,
                    zarr_path=self.get_zarr_path(p),
                    annotations_path=self.get_geojson_path(p),
                    metadata=image_metadata(p),
                    roi_path=None
                )
            )
        return images

    def image_paths(self, extension="ndpi") -> List[Path]:
        """Get list of images for given extension."""
        paths = self.data_path.glob(f"**/*.{extension}")
        image_filter = self.filter if self.filter else (lambda x: True)
        return [p for p in paths if image_filter(p)]

    def get_zarr_path(self, image: Path) -> Path:
        """Get Zarr Path for image."""
        return self.zarr_path / f"{image.stem}.zarr"

    def get_geojson_path(self, image: Path) -> Path:
        """Get geojson annotation path for image."""
        # resolve relative path
        directory = self.annotation_path / image.relative_to(self.data_path).parent.parent / "objectsjson"
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
        console.print(f"filter: {self.filter}")
        console.rule(style="white")

        for k, p in enumerate(image_paths):
            zarr_path = self.get_zarr_path(p).relative_to(self.zarr_path)
            geojson_path = self.get_geojson_path(p).relative_to(self.annotation_path)
            # console.print(f"{p.relative_to(self.data_path)} | {zarr_path} | {geojson_path}")
            console.print(f"[{k}] {image_metadata(p)}")
        console.rule(style="white")


def image_metadata(image: Path) -> ImageMetadata:
    """Metadata for image"""
    rat_pattern = re.compile("NOR-\d+")
    pig_pattern = re.compile("SSES2021 \d+")
    mouse_pattern = re.compile("MNT-\d+")
    human_pattern = re.compile("UKJ-19-\d+_Human")

    image_id = image.stem
    species = image.parent.parent.name.lower()
    if species == "pig":
        match = re.search(pig_pattern, image_id)
    elif species == "mouse":
        match = re.search(mouse_pattern, image_id)
    elif species == "rat":
        match = re.search(rat_pattern, image_id)
    elif species == "human":
        match = re.search(human_pattern, image_id)

    if match:
        subject = match.group(0)
    else:
        subject = None

    return ImageMetadata(
        image_id=image_id,
        subject=subject,
        species=species,
        protein=image.parent.name.lower(),
        negative="negative" in image_id.lower(),
    )


@dataclass
class ImageMetadata:
    """Metadata resolved from path information."""

    subject: str
    species: str
    protein: str
    negative: bool
    image_id: str


def filter_factory(
    subject: Optional[str] = None,
    species: Optional[str] = None,
    protein: Optional[str] = None,
    negative: bool = False
):
    """Create filter functions"""

    def f_filter(p: Path):
        """Filter all images for rat and CYP1A2"""
        md = image_metadata(p)

        keep = True
        if subject:
            keep = keep and md.subject == subject
        if species:
            keep = keep and md.species == species.lower()
        if protein:
            keep = keep and md.protein == protein.lower()
        keep = keep and md.negative == negative
        return keep

    return f_filter


if __name__ == "__main__":
    from zia import BASE_PATH
    from zia.config import read_config

    configuration = read_config(BASE_PATH / "configuration.ini")
    file_manager = FileManager(configuration)
    file_manager.info()

    # Rat, CYP1A2
    file_manager = FileManager(
        configuration,
        filter=filter_factory(species="rat", protein="cyp1a2")
    )
    file_manager.info()

    # Data for subject
    file_manager = FileManager(
            configuration,
            filter=filter_factory(subject="NOR-022")
    )
    file_manager.info()
    images = file_manager.get_images()
    console.print(images)