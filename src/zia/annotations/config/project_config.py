import configparser
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from zia.annotations.config import RESOURCE_PATH


class LabelStudioConfig(BaseModel):
    """Frog Objective."""

    api_token: str
    base_url: str


@dataclass
class SampleDataConfig:
    reference_stain_matrix: np.ndarray
    reference_max_conc: np.ndarray
    tile_coords_path: Path


def read_labelstudio_config(mode: Literal["Production", "Test"]) -> LabelStudioConfig:
    """Read configuration setting."""
    config = configparser.ConfigParser()
    config.read(Path(__file__).parent.parent / "configuration.ini")

    configuration = LabelStudioConfig(
        api_token=config[mode]["api_token"],
        base_url=config[mode]["base_url"]
    )

    return configuration


def read_sample_data_config() -> SampleDataConfig:
    config = configparser.ConfigParser()
    config.read(Path(__file__).parent.parent / "configuration.ini")

    configuration = SampleDataConfig(
        reference_stain_matrix=np.array(json.loads(config["StainSeparation"]["HERef"])),
        reference_max_conc=np.array(json.loads(config["StainSeparation"]["maxCRef"])),
        tile_coords_path=Path(config["TrainingData"]["tile_coords_path"])
    )

    return configuration


class ResourcePaths:
    def __init__(self, data_set: str):
        self.base_path = RESOURCE_PATH / data_set
        self.image_path = self.base_path / "image"
        self.mask_path = self.base_path / "mask"
        self.polygon_path = self.base_path / "polygons"
        self.predictions = self.base_path / "predictions"
        self.raw_masks = self.base_path / "raw_masks"

        self.initialize_paths()

    def initialize_paths(self):
        for p in [self.base_path, self.image_path, self.mask_path, self.polygon_path, self.predictions, self.raw_masks]:
            p.mkdir(exist_ok=True, parents=True)
