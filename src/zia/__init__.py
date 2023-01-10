from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent

CZI_IMAGES_INITIAL = sorted([p for p in (BASE_PATH / "data" / "czi" / "Initial").glob("*.czi")])
CZI_IMAGES_AXIOS = sorted([p for p in (BASE_PATH / "data" / "czi" / "AxioScanZeiss").glob("*.czi")])

CZI_EXAMPLE = BASE_PATH / "src" / "zia" / "resources" / "RH0422_1x2.5.czi"

RESULTS_PATH = BASE_PATH / "results"

for p in [RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)
