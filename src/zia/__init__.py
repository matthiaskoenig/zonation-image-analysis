from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent
RESOURCES_PATH = BASE_PATH / "src" / "zia" / "resources"
DATA_PATH = BASE_PATH / "data"

CZI_IMAGES_INITIAL = sorted([p for p in (BASE_PATH / "data" / "czi" / "Initial").glob("*.czi")])
CZI_IMAGES_AXIOS = sorted([p for p in (BASE_PATH / "data" / "czi" / "AxioScanZeiss").glob("*.czi")])

CZI_EXAMPLE = BASE_PATH / "src" / "zia" / "resources" / "RH0422_1x2.5.czi"
OME_TIFF_EXAMPLE = BASE_PATH / "src" / "zia" / "resources" / "RH0422_1x2.5.tif"

RESULTS_PATH = BASE_PATH / "results"

file_path_luca = DATA_PATH / "LuCa-7color_Scan1.ome.tiff"
# file_path_npdi = DATA_PATH / "LQF2_LM_HE_PVL.ndpi"
file_path_npdi = DATA_PATH / "ndpi" / "LQF1_LM_HE_PVL.ndpi"

file_path_lqf2 = DATA_PATH / "LQF2_RM_HE_PVL2.ome.tif"
file_path_czi = DATA_PATH / "RH0422_1x2.5.czi"
file_path_czi2 = DATA_PATH / "2022_11_10__0003.czi"
file_path_tiff_exported2 = DATA_PATH / "2022_11_10__0003.ome.tif"

file_path_tiff_test1 = DATA_PATH / "test_001.tif"
file_path_tiff_test1_exported = DATA_PATH / "test_001_exported.ome.tif"
file_path_tiff_test1_exported2 = DATA_PATH / "test_001_exported2.ome.tif"

TEST_FILES_PATHS = [
    file_path_luca,
    file_path_npdi,
    file_path_lqf2,
    file_path_czi,
    file_path_tiff_test1,
    file_path_czi,
    file_path_czi2,
    file_path_tiff_test1_exported,
    file_path_tiff_test1_exported2]

for p in [RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)
