"""
zonation-image analysis - Python utilities for image analyses of zonation patterns.
"""
from pathlib import Path

__author__ = "Matthias König"
__version__ = "0.0.2"

program_name = "zonation-image-analysis"


BASE_PATH = Path(__file__).parent.parent.parent
RESOURCES_PATH = BASE_PATH / "src" / "zia" / "resources"
RESULTS_PATH = BASE_PATH / "results"
REPORT_PATH = BASE_PATH / "reports"

for p in [RESULTS_PATH, REPORT_PATH]:
    p.mkdir(parents=True, exist_ok=True)


# FIXME: handle better
OPENSLIDE_PATH = r"C:\Program Files\OpenSlide\openslide-win64-20230414\bin"
prefix = "/media/jkuettner/Extreme Pro"
# prefix = "D:"
DATA_PATH = Path(f"{prefix}/data")
ZARR_PATH = Path(f"{prefix}/zarr")
