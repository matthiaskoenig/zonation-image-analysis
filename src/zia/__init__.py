from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent

CZI_PATH = BASE_PATH / "data" / "czi"
CZI_IMAGES = sorted([p for p in CZI_PATH.glob("*.czi")])

CZI_EXAMPLE = BASE_PATH / "src" / "zia" / "resources" / "RH0422_1x2.5.czi"

RESULTS_PATH = BASE_PATH / "results"

for p in [CZI_PATH, RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)
