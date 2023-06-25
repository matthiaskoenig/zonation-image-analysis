"""
zonation-image analysis - Python utilities for image analyses of zonation patterns.
"""
from pathlib import Path
import configparser

def read_config(file_path: Path) -> dict:
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


__author__ = "Matthias König"
__version__ = "0.0.2"

program_name = "zonation-image-analysis"

BASE_PATH = Path(__file__).parent.parent.parent

config = read_config(BASE_PATH / "path_config.ini")

RESULTS_PATH = Path(config["PipelinePaths"]["results_path"])
REPORT_PATH = Path(config["PipelinePaths"]["reports_path"])
DATA_PATH = Path(config["PipelinePaths"]["data_path"])
ZARR_PATH = Path(config["PipelinePaths"]["zarr_path"])

for p in [RESULTS_PATH, REPORT_PATH]:
    p.mkdir(parents=True, exist_ok=True)

OPENSLIDE_PATH = Path(config["OpenSlide"]["open_slide"])

RESOURCES_PATH = BASE_PATH / "src" / "zia" / "resources"
