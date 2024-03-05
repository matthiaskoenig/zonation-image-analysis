import configparser
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class Configuration(BaseModel):
    """Frog Objective."""

    reports_path: Path
    data_path: Path
    annotations_path: Path
    image_data_path: Path
    openslide_path: Optional[Path]
    libvips_path: Optional[Path]
    extension: str

    def __post_init_post_parse__(self):
        """Ensure paths exists."""
        if not self.data_path.exists():
            raise FileNotFoundError("The data directory does not exist.")
        if not self.annotations_path.exists():
            raise FileNotFoundError("The annotations directory does not exist.")
        for p in [self.image_data_path, self.reports_path]:
            p.mkdir(exist_ok=True, parents=True)


def read_config() -> Configuration:
    """Read configuration setting."""
    config = configparser.ConfigParser()
    config.read(Path(__file__).parent.parent / "configuration.ini")

    openslide_path = config["Dependency"]["open_slide"]
    libvips_path = config["Dependency"]["libvips"]
    configuration = Configuration(
        reports_path=Path(config["Paths"]["reports_path"]),
        data_path=Path(config["Paths"]["data_path"]),
        annotations_path=Path(config["Paths"]["annotations_path"]),
        image_data_path=Path(config["Paths"]["image_data_path"]),
        openslide_path=Path(openslide_path) if openslide_path else None,
        libvips_path=Path(libvips_path) if libvips_path else None,
        extension=config["Extensions"]["extension"]
    )

    return configuration


def get_project_config(project_name: str) -> Configuration:
    config = configparser.ConfigParser()

    config.read(Path(__file__).parent.parent / "configuration.ini")

    openslide_path = config["Dependency"]["open_slide"]
    libvips_path = config["Dependency"]["libvips"]
    configuration = Configuration(
        reports_path=Path(config["Paths"]["reports_path"]) / project_name,
        data_path=Path(config["Paths"]["data_path"]) / project_name,
        image_data_path=Path(config["Paths"]["image_data_path"]) / project_name,
        annotations_path=Path(config["Paths"]["annotations_path"]) / project_name,
        openslide_path=Path(openslide_path) if openslide_path else None,
        libvips_path=Path(libvips_path) if libvips_path else None,
        extension=config["Extensions"]["extension"]
    )

    return configuration
