"""
zonation-image analysis - Python utilities for image analyses of zonation patterns.
"""
from pathlib import Path

__author__ = "Matthias König"
__version__ = "0.0.0"

program_name = "zonation-image-analysis"


BASE_PATH = Path(__file__).parent.parent.parent
RESOURCES_PATH = BASE_PATH / "src" / "zia" / "resources"


DATA_PATH = Path("D:/data")
ZARR_PATH = Path("D:/zarr")
RESULTS_PATH = BASE_PATH / "results"
REPORT_PATH = BASE_PATH / "reports"

CZI_IMAGES_INITIAL = sorted(
    [p for p in (BASE_PATH / "data" / "czi" / "Initial").glob("*.czi")]
)
CZI_IMAGES_AXIOS = sorted(
    [p for p in (BASE_PATH / "data" / "czi" / "AxioScanZeiss").glob("*.czi")]
)

# example_ometiff: Path = DATA_PATH / "ometiff" / "LuCa-7color_Scan1.ome.tiff"
# example_npdi: Path = DATA_PATH / "ndpi" / "LQF1_LM_HE_PVL.ndpi"
# example_czi: Path = DATA_PATH / "czi" / "Initial" / "RH0422_1x2.5.czi"
# example_czi_axios1: Path = DATA_PATH / "czi" / "AxioScanZeiss" / "2022_11_10__0001.czi"
# example_czi_axios2: Path = DATA_PATH / "czi" / "AxioScanZeiss" / "2022_11_10__0002.czi"
# example_czi_axios3: Path = DATA_PATH / "czi" / "AxioScanZeiss" / "2022_11_10__0003.czi"
# example_czi_axios4: Path = DATA_PATH / "czi" / "AxioScanZeiss" / "2022_11_10__0004.czi"

for p in [RESULTS_PATH, REPORT_PATH]:
    p.mkdir(parents=True, exist_ok=True)
