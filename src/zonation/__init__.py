from pathlib import Path

CZI_PATH = Path(__file__).parent.parent.parent / "data" / "czi"
CZI_IMAGES = sorted([p for p in CZI_PATH.glob("*.czi")])

IMAGE_PATH = Path(__file__).parent.parent.parent / "data" / "images"
IMAGES = sorted([p for p in IMAGE_PATH.glob("*.czi")])

RESULTS_PATH = Path(__file__).parent.parent.parent / "results"

for p in [CZI_PATH, IMAGE_PATH, RESULTS_PATH]:
    p.mkdir(parents=True, exist_ok=True)