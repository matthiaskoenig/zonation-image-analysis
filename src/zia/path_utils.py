"""Module handling project paths."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, List, Dict

from zia.config import Configuration
from zia.console import console


class ResultsDirectories(str, Enum):
    """Reused directories."""

    LIVER_ROIS = "liver_rois"
    LIVER_ROIS_IMAGES = "liver_rois_images"
    LIVER_MASK = "liver_mask"
    MASKED_PNG_IMAGES = "masked_png_images"
    STAIN_SEPERATED_IMAGES = "stain_seperated_images"


@dataclass
class ImageInfo:
    path: Path
    zarr_path: Path
    annotations_path: Path
    metadata: ImageMetadata
    roi_path: Path


class FileManager:
    """Class for managing set of images."""

    def __init__(self, configuration: Configuration, filter: Optional[Callable] = None):
        """Initialize the file manager."""
        self._configuration: Configuration = configuration
        self.data_path: Path = configuration.data_path
        self.image_data_path: Path = configuration.image_data_path
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
                    roi_path=self.get_roi_path(p)
                )
            )
        return images

    def image_paths(self, extension="ndpi") -> List[Path]:
        """Get list of images for given extension."""
        paths = self.data_path.glob(f"**/*.{extension}")
        image_filter = self.filter if self.filter else (lambda x: True)
        return [p for p in paths if image_filter(p)]

    def image_info_grouped_by_subject(self) -> Dict[str, List[ImageInfo]]:
        image_info_dict = {}
        for image_info in self.get_images():
            subject = image_info.metadata.subject
            if subject not in image_info_dict:
                image_info_dict[subject] = []
            image_info_dict[subject].append(image_info)
        return image_info_dict

    def get_zarr_path(self, image: Path) -> Path:
        """Get Zarr Path for image."""
        zarr_path = self.image_data_path / "zarr"
        zarr_path.mkdir(parents=True, exist_ok=True)
        return zarr_path / f"{image.stem}.zarr"

    def get_roi_path(self, image: Path) -> Path:
        """Get ROI Path for image."""
        roi_path = self.image_data_path / "rois"
        roi_path.mkdir(parents=True, exist_ok=True)
        return roi_path / f"{image.stem}.geojson"

    def get_geojson_path(self, image: Path) -> Path:
        """Get geojson common path for image."""
        # resolve relative path
        directory = self.annotation_path / image.relative_to(
            self.data_path).parent.parent / "objectsjson"
        return directory / f"{image.stem}.geojson"

    def info(self):
        console.rule(style="white")
        image_paths = list(self.image_paths())
        console.print(f"FileManager: {len(image_paths)} images")
        console.print(f"filter: {self.filter}")
        console.rule(style="white")

        for k, p in enumerate(image_paths):
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
        negative=any([s in image_id.lower() for s in ["negative", "neg."]]),
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
