"""Module for handling project configuration."""
import configparser
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from zia import RESOURCES_PATH
from zia.console import console


class Configuration(BaseModel):
    """Frog Objective."""

    results_path: Path
    reports_path: Path
    data_path: Path
    annotations_path: Path
    zarr_path: Path
    openslide_path: Optional[Path]

    def __post_init_post_parse__(self):
        """Ensure paths exists."""
        for p in [self.results_path, self.zarr_path]:
            p.mkdir(exist_ok=True, parents=True)


def read_config(file_path: Path) -> Configuration:
    """Read configuration setting."""
    config = configparser.ConfigParser()
    config.read(file_path)

    openslide_path = config["OpenSlide"]["open_slide"]
    configuration = Configuration(
        results_path=Path(config["Paths"]["results_path"]),
        reports_path=Path(config["Paths"]["reports_path"]),
        data_path=Path(config["Paths"]["data_path"]),
        annotations_path=Path(config["Paths"]["annotation_path"]),
        zarr_path=Path(config["Paths"]["zarr_path"]),
        openslide_path=Path(openslide_path) if openslide_path else None,
    )
    console.print(configuration)
    return configuration


if __name__ == "__main__":
    configuration: Configuration = read_config(RESOURCES_PATH / "config.template")